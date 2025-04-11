import sys
import os
import faiss
import pickle
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
import multiprocessing as mp
from tqdm import tqdm
from typing import *

import config

class hfRetriever:
    def __init__(
        self, 
        model_path: str = "retrieve/facebook-contriever",
        tokenizer_path: str = "retrieve/facebook-contriever",
        corpus_path: str = "corpus/web/txt",
        faiss_index_path: str = "faiss_index",
        doc_path: str = "doc.pkl",  # Ensure index and doc correspond one-to-one
        chunk_size: int = config.chunk_size,
        overlap_size: int = config.chunk_overlap
    ) -> None:
        """
        Load model and tokenizer
        """
        self.model = AutoModel.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.device = self.model.device
        self.corpus_path = corpus_path
        self.faiss_index_path = faiss_index_path
        self.doc_path = doc_path
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.batch_size = 512
        self.index: Optional[faiss.IndexFlatIP] = None
        self.nodes: Optional[List[str]] = None
        self.embedding_dim: Optional[int] = None
        mp.set_start_method('spawn', force=True)  # Set multiprocessing start method to spawn

        if not os.path.exists(faiss_index_path):
            self.build_faiss_index()
        else:
            self.load_faiss_index()

    def is_gibberish(self, text: str, threshold: float = 0.6) -> bool:
        """
        Calculate the ratio of printable characters. If the ratio is below the threshold, consider it gibberish.
        """
        import string
        printable_chars = set(string.printable)  # Set of printable characters
        printable_count = sum(1 for char in text if char in printable_chars)
        ratio = printable_count / len(text)
        return ratio < threshold

    def build_faiss_index(self) -> None:
        """
        Load documents, split them, encode, and store vectors in the FAISS index.
        """
        if os.path.exists(self.doc_path):
            self.nodes = pickle.load(open(self.doc_path, "rb"))
        else:
            documents: List[Document] = []
            for file in sorted(os.listdir(self.corpus_path)):  # Use sorted to ensure file order for later checks
                with open(os.path.join(self.corpus_path, file), "r", encoding="utf-8") as f:
                    lines = f.readlines()
                documents.extend([Document(text=line.strip()) for line in lines if len(line.strip()) > 1000 and not self.is_gibberish(line.strip())])
            
            parser = SentenceSplitter(chunk_size=self.chunk_size, chunk_overlap=self.overlap_size)
            nodes = parser.get_nodes_from_documents(documents, show_progress=True)
            self.nodes = list(set([node.text.strip() for node in nodes if len(node.text.strip()) > 1000 and not self.is_gibberish(node.text.strip())]))
            with open(self.doc_path, "wb") as f:
                pickle.dump(self.nodes, f)
        
        with torch.no_grad():
            embeddings = self._encode_with_multiple_gpus(self.nodes)
        assert len(self.nodes) == len(embeddings), "Mismatch between document chunks and embeddings."

        self.embedding_dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        faiss.normalize_L2(embeddings)  # Normalize inner product, equivalent to cosine similarity
        self.index.add(embeddings)  # Add all vectors to the index
        assert self.index.ntotal == len(self.nodes), "FAISS index size does not match the number of nodes."
        
        self.save_faiss_index()

    def _encode_with_multiple_gpus(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts using multiple GPUs.
        """
        device_count = torch.cuda.device_count()
        split_size = len(texts) // device_count
        splits = [texts[i * split_size: (i + 1) * split_size] for i in range(device_count)]
        if len(texts) % device_count != 0:
            splits[-1].extend(texts[device_count * split_size:])

        with mp.Pool(device_count) as pool:
            results = pool.starmap(self._encode_on_single_gpu, [(i, splits[i]) for i in range(device_count)])

        embeddings = np.vstack(results)
        return embeddings

    def _encode_on_single_gpu(self, gpu_id: int, texts: List[str]) -> np.ndarray:
        """
        Encode texts on a single GPU, processing in batches.
        """
        device = f'cuda:{gpu_id}'
        torch.cuda.set_device(gpu_id)  # Set the GPU device for the current process
        model = self.model.to(device)  # Load the model onto the specified GPU

        embeddings: List[np.ndarray] = []
        batch_size = self.batch_size
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i+batch_size]
            encoded_inputs = self.tokenizer.batch_encode_plus(
                batch_texts,
                max_length=512,
                truncation=True,
                padding=True,
                add_special_tokens=True,
                return_tensors="pt",
            ).to(device)
            attention_mask = encoded_inputs['attention_mask']
            with torch.no_grad():
                last_hidden = model(**encoded_inputs)["last_hidden_state"]
                last_hidden = last_hidden.masked_fill(~attention_mask[..., None].bool(), 0.0)
                batch_embeddings = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
                batch_embeddings = batch_embeddings.cpu().numpy()
                embeddings.append(batch_embeddings)
        
        return np.concatenate(embeddings, axis=0)

    def update_faiss_index(self) -> None:
        """
        Update the FAISS index.
        """
        self.index.reset()
        torch.cuda.empty_cache()
        with torch.no_grad():
            embeddings = self._encode_with_multiple_gpus(self.nodes)
        assert len(self.nodes) == len(embeddings), "Mismatch between document chunks and embeddings."
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        assert self.index.ntotal == len(self.nodes), "FAISS index size does not match the number of nodes."

    def save_faiss_index(self) -> None:
        """
        Save the FAISS index to disk.
        """
        if self.index is None or self.nodes is None:
            raise ValueError("Index or nodes are not built yet. Build the FAISS index first.")
        
        faiss.write_index(self.index, self.faiss_index_path)
        # with open(self.faiss_index_path + ".pkl", "wb") as f:
        #     pickle.dump(self.nodes, f)

    def load_faiss_index(self) -> None:
        """
        Load the FAISS index from disk.
        """
        self.index = faiss.read_index(self.faiss_index_path)
        with open(self.doc_path, "rb") as f:
            self.nodes = pickle.load(f)
        self.embedding_dim = self.index.d

    def batch_retrieve(self, queries: List[str], top_k: int = 3) -> Tuple[List[List[str]], torch.Tensor]:
        """
        Search the FAISS index and return the original document chunks based on vectors.
        """
        if self.index is None or self.nodes is None:
            raise ValueError("Index or nodes are not loaded. Load or build the FAISS index first.")

        encoded_inputs = self.tokenizer.batch_encode_plus(
            queries,
            max_length=128,
            truncation=True,
            padding=True,
            add_special_tokens=True,
            return_tensors="pt",
        ).to(self.device)
        attention_mask = encoded_inputs['attention_mask']
        last_hidden = self.model(**encoded_inputs)["last_hidden_state"]
        last_hidden = last_hidden.masked_fill(~attention_mask[..., None].bool(), 0.0)
        query_embeddings = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=-1)

        query_embeddings_np = query_embeddings.detach().numpy()
        query_embeddings_np = np.ascontiguousarray(query_embeddings_np)

        _, indices = self.index.search(query_embeddings_np, top_k)
        top_k_embeddings = np.array([self.index.reconstruct(int(i)) for i in indices.flatten()]).reshape(indices.shape[0], indices.shape[1], -1)
        top_k_embeddings = torch.tensor(top_k_embeddings, device=self.device)
        cosine_similarities = torch.matmul(query_embeddings.unsqueeze(1), top_k_embeddings.transpose(1, 2)).squeeze(1)

        results = [[self.nodes[i] for i in idxs] for idxs in indices]
        return results, cosine_similarities


if __name__ == "__main__":
    faiss_index = hfRetriever(corpus_path="corpus/wiki/txt", faiss_index_path="faiss_index_wiki", doc_path="wiki_doc.pkl")

    query = [
        "where did the ceo of salesforce previously work?",
        "who is the ceo of salesforce?",
    ]
    results, scores = faiss_index.batch_retrieve(query)
    print(results[0][0])
    print(scores[0][0])
    print("====================================")
    print(results[0][1])
    print(scores[0][1])
