import sys
import os
import json
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import config
from Component.hfLLM import hfLLM
from Component.hfRetriever import hfRetriever
import random
import faiss

from typing import *

# DEBUG and INFO messages, output to log file
import logging
logging.basicConfig(level=logging.INFO, filename="train_log", format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Training Log")
START_EPOCH = 1
BATCH_SIZE = 20
BLOCK_SIZE = 8
TOP_K = 20
LR = 1e-6

class fairFT:
    def __init__(self, retriever: hfRetriever, generator: hfLLM, lr: float = 1e-5) -> None:
        """
        Initialize the fairFT trainer.
        retriever: hfRetriever instance for document retrieval.
        generator: hfLLM instance for generation and likelihood computation.
        lr: Learning rate for optimizing the retriever.
        """
        self.retriever = retriever
        self.generator = generator
        self.llm_name = generator.model_name
        self.retriever.model.train()
        self.optimizer = optim.Adam(self.retriever.model.parameters(), lr=lr)  # Optimizer for the retriever
        self.gamma = 0.1  # Retrieval score scaling factor
        self.beta = 0.1  # LM score scaling factor

    def compute_kl_divergence(self, retriever_probs: torch.Tensor, lm_probs: torch.Tensor) -> torch.Tensor:
        """
        Compute the KL divergence between the retriever's probability distribution and the LM's probability distribution.
        retriever_probs: Probability distribution from the retriever (batch_size, num_docs).
        lm_probs: Probability distribution from the LM (batch_size, num_docs).
        return: KL divergence.
        """
        kl_div = F.kl_div(retriever_probs.log(), lm_probs, reduction='batchmean')  # KL divergence
        return kl_div

    def train_step(
        self, 
        input_queries: List[str], 
        ground_truths: List[str], 
        prompt: str, 
        top_k: int = 3
    ) -> float:
        """
        Perform a single training step (batch).
        input_queries: List of input queries (batch_size,).
        ground_truths: List of target outputs (true documents or sequences) (batch_size,).
        top_k: Number of documents to retrieve.
        return: Loss value for the single training step (float).
        """
        batch_size = len(input_queries)

        # Step 1: Batch retrieve relevant document blocks for each query using FaissIndex
        retrieved_docs, retriever_scores = self.retriever.batch_retrieve(input_queries, top_k=top_k)

        # Step 2: Compute the likelihood of the retriever
        retriever_probs = F.softmax(retriever_scores / self.gamma, dim=-1)  # Convert to probability distribution

        # Step 3: Batch compute LM scores for the retrieved documents
        expanded_prompts: List[str] = []
        expanded_truths: List[str] = []

        # Expand prompts and truths
        for query, doc_list, truth in zip(input_queries, retrieved_docs, ground_truths):
            # Combine query and each doc into a prompt
            for doc in doc_list:
                expanded_prompts.append(prompt.format(query_str=query, context_str=doc))
                expanded_truths.append(truth)  # Expand truth top_k times

        # Compute likelihood
        length = len(expanded_prompts)
        block_size = BLOCK_SIZE
        lm_probs: List[torch.Tensor] = []
        for i in range(0, length, block_size):
            torch.cuda.empty_cache()
            prompts = expanded_prompts[i:i+block_size]
            truths = expanded_truths[i:i+block_size]
            probs = self.generator.get_likelihood(prompts, truths)
            lm_probs.append(probs)

        lm_probs = torch.cat(lm_probs, dim=0)
        lm_probs = lm_probs.view(-1, top_k)
        # Convert to probability distribution
        lm_probs = F.softmax(lm_probs / self.beta, dim=-1)

        # Step 4: Compute KL divergence and accumulate loss over the batch
        retriever_probs = retriever_probs.to(self.retriever.device)
        lm_probs = lm_probs.to(self.retriever.device)

        kl_div_total = torch.tensor(0.0, device=self.retriever.device)
        for retriever_prob, lm_prob in zip(retriever_probs, lm_probs):
            kl_div_total += self.compute_kl_divergence(retriever_prob, lm_prob)

        # Step 5: Backpropagate and update the retriever
        loss = kl_div_total / batch_size
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()  # Return the average KL divergence per sample

    def async_update_faiss_index(self) -> None:
        """
        Asynchronously update the FAISS index.
        """
        # Temporarily release the generator's GPU memory
        logger.info("Releasing generator model...")
        self.generator.model.cpu()
        del self.generator.model
        self.generator.model = None
        torch.cuda.empty_cache()

        # Re-encode all documents and update the index
        self.retriever.model.eval()
        self.retriever.update_faiss_index()
        self.retriever.model.train()

        # Reload the generator model
        torch.cuda.empty_cache()
        self.generator.reload_model()

    def train(
        self, 
        training_data: List[Tuple[str, str]], 
        batch_size: int = 8, 
        epochs: int = 3, 
        async_update_interval: int = 100
    ) -> None:
        """
        Train the model.
        training_data: A list of (input_query, ground_truth) tuples.
        batch_size: Batch size.
        epochs: Number of training epochs.
        async_update_interval: Interval for asynchronously updating the FAISS index.
        """
        logger.info("Starting training...")

        prompt = config.fairness_gen

        for epoch in range(epochs):
            if epoch + 1 < START_EPOCH:
                continue
            print(f"Epoch {epoch+1}/{epochs}")
            logger.info(f"Epoch {epoch+1}/{epochs}")

            # Create DataLoader
            data_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
            total_kl_div = 0

            # Add progress bar to the training step loop
            with tqdm(total=len(data_loader), desc=f"Epoch {epoch+1}", unit="batch") as pbar:
                for step, batch in enumerate(data_loader):

                    input_queries, ground_truths = batch  # Unpack a batch of data
                    torch.cuda.empty_cache()

                    # Perform batch training step
                    kl_div = self.train_step(input_queries, ground_truths, prompt=prompt, top_k=TOP_K)
                    total_kl_div += kl_div

                    # Update progress bar
                    pbar.set_postfix({"Avg KL Divergence": kl_div})
                    pbar.update(1)  # Update progress bar after processing each batch

                    # Clear CUDA cache
                    torch.cuda.empty_cache()
                    # Asynchronously update FAISS index
                    if ((step + 1) * batch_size) % async_update_interval == 0 and step > 0:
                        try:
                            self.async_update_faiss_index()
                            if (step + 1) * batch_size % 1000 == 0:
                                save_path = f"epoch_{epoch+1}"
                                # Save retriever model
                                logger.info("Saving new retriever...")
                                self.retriever.model.save_pretrained(save_path + "_retriever")
                                # Save index
                                logger.info("Saving new index...")
                                faiss.write_index(self.retriever.index, save_path + "_index")

                        except Exception as e:
                            # Log error
                            logger.error(f"Error updating Faiss Index: {e}")
                            logger.info(f"current STEP {step}")
                            exit()

            print(f"Epoch {epoch+1} Avg KL Divergence: {total_kl_div/len(data_loader)}")
            logger.info(f"Epoch {epoch+1} Avg KL Divergence: {total_kl_div/len(data_loader)}")

if __name__ == "__main__":
    logger.info("================================================================")
    # Load training data
    data_path = "datasets/fairness/stereotype_recognition.json"
    training_data: List[Tuple[str, str]] = []
    with open(data_path, "r") as f:
        data = json.load(f)
    for item in data:
        query = item["prompt"][29:]
        if item["label"] == "stereotype":
            truth = "1. Yes, it's a stereotype"
        elif item["label"] == "anti-stereotype":
            truth = "2. No, it's an anti-stereotype"
        else:
            truth = "3. It's unrelated and meaningless"
        training_data.append((query, truth))

    # Initialize retriever and generator
    if START_EPOCH > 1:
        retriever = hfRetriever(
            model_path=f"epoch_{START_EPOCH-1}_retriever", 
            faiss_index_path=f"epoch_{START_EPOCH-1}_index",
            doc_path="train_doc.pkl"
        )
    else:
        retriever = hfRetriever(
            model_path="retrieve/facebook-contriever", 
            faiss_index_path="faiss_index_train",
            doc_path="train_doc.pkl"
        )

    torch.cuda.empty_cache()
    generator = hfLLM(model_name="/data/share_weight/Meta-Llama-3-8B-Instruct")

    # Initialize trainer and start training
    trainer = fairFT(retriever, generator, lr=LR)
    trainer.train(training_data, batch_size=BATCH_SIZE, epochs=5, async_update_interval=500)
