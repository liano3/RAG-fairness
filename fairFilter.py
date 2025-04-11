import os
import sys
from tqdm import tqdm
import json
import torch
from typing import *

import config
from Component.hfRetriever import hfRetriever
from Component.bm25Retriever import bm25Retriever
from Component.hfLLM import hfLLM
from Component.gptLLM import GLM

class RAG:
    def __init__(self, llm: hfLLM, retriever: Optional[hfRetriever] = None, filter: Optional[bool] = False) -> None:
        """
        Initialize the RAG class.

        Args:
            llm (hfLLM): Language model object for text generation.
            retriever (Optional[hfRetriever]): Retriever object for fetching relevant texts. Can be None.
            filter (Optional[bool]): Whether to apply filtering. Default is False.
        """
        self.llm = llm
        self.retriever = retriever
        self.filter = filter

    def direct_gen(self, queries: List[str], prompt: str, queries_time: Optional[List[str]] = None) -> List[str]:
        """
        Directly generate text.

        Args:
            queries (List[str]): List of query strings.
            prompt (str): Template for generation prompt.
            queries_time (Optional[List[str]]): List of query times, optional.

        Returns:
            List[str]: List of generated texts.
        """
        if queries_time is not None:
            input_texts = [prompt.format(query_str=query, query_time=query_time) for query, query_time in zip(queries, queries_time)]
        else:
            input_texts = [prompt.format(query_str=query) for query in queries]
        return self.llm.generate(input_texts)
    
    def rag_gen(self, queries: List[str], prompt: str, top_k: int = 3, queries_time: Optional[List[str]] = None) -> Tuple[List[str], List[List[str]]]:
        """
        Generate text using RAG.

        Args:
            queries (List[str]): List of query strings.
            prompt (str): Template for generation prompt.
            top_k (int): Number of top relevant documents to retrieve. Default is 3.
            queries_time (Optional[List[str]]): List of query times, optional.

        Returns:
            Tuple[List[str], List[List[str]]]: List of generated texts and retrieved contexts.
        """
        # Retrieve relevant texts
        retrieved_texts, _ = self.retriever.batch_retrieve(queries, top_k=top_k)
        if self.filter:
            retrieved_texts = [self.llm.batch_filter(query, texts) for query, texts in zip(queries, retrieved_texts)]
        if queries_time is not None:
            input_texts = [prompt.format(query_str=query, query_time=query_time, context_str="\n\n".join(retrieved_text)) for query, query_time, retrieved_text in zip(queries, queries_time, retrieved_texts)]
        else:
            input_texts = [prompt.format(query_str=query, context_str="\n\n".join(retrieved_text)) for query, retrieved_text in zip(queries, retrieved_texts)]
        return self.llm.generate(input_texts), retrieved_texts
    
    def gen_fairness(self, data_path: str, save_path: str, direct_gen: bool, batch_size: int, count: int) -> None:
        """
        Generate fairness data and save results.

        Args:
            data_path (str): Path to input data file.
            save_path (str): Path to save generated data.
            direct_gen (bool): Whether to generate results directly (without retrieval).
            batch_size (int): Size of each generation batch.
            count (int): Current generation count for file naming.

        Returns:
            None
        """
        with open(data_path, "r", encoding='utf-8') as f:
            data = json.load(f)
        
        if direct_gen:
            prompt = "{query_str}"
        else:
            prompt = config.fairness_gen
        
        for i in tqdm(range(0, len(data), batch_size)):
            batch_data = data[i:i+batch_size]
            if direct_gen:
                queries = [d['prompt'] for d in batch_data]
                res = self.direct_gen(queries, prompt)
            else:
                queries = [d['prompt'][29:] for d in batch_data]
                res, docs = self.rag_gen(queries, prompt, top_k=config.top_k)
                for d, doc in zip(batch_data, docs):
                    d['docs'] = doc

            for d, r in zip(batch_data, res):
                d['res'] = r

        os.makedirs(save_path, exist_ok=True)
        save_file = os.path.join(save_path, "stereotype_agreement" + f"_{count}.json")
        with open(save_file, "w", encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Results saved to {save_file}")

    def gen_crag(self, data_path: str, save_path: str, direct_gen: bool, batch_size: int, count: int) -> None:
        """
        Generate CRAG data and save results.

        Args:
            data_path (str): Path to input data file.
            save_path (str): Path to save generated data.
            direct_gen (bool): Whether to generate results directly (without retrieval).
            batch_size (int): Size of each generation batch.
            count (int): Current generation count for file naming.

        Returns:
            None
        """
        with open(data_path, "r", encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
        
        if direct_gen:
            prompt = config.direct_gen
        else:
            prompt = config.crag_gen
        
        for i in tqdm(range(0, len(data), batch_size)):
            batch_data = data[i:i+batch_size]
            queries = [d['query'] for d in batch_data]
            queries_time = [d['query_time'] for d in batch_data]

            if direct_gen:
                res = self.direct_gen(queries, prompt, queries_time=queries_time)
            else:
                res, docs = self.rag_gen(queries, prompt, top_k=config.top_k, queries_time=queries_time)
                for d, doc in zip(batch_data, docs):
                    d['docs'] = doc

            for d, r in zip(batch_data, res):
                d['res'] = r

        os.makedirs(save_path, exist_ok=True)
        save_file = os.path.join(save_path, "crag" + f"_{count}.json")
        with open(save_file, "w", encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Results saved to {save_file}")

    def cache_gen_fair(self, data_path: str, save_path: str, batch_size: int, count: int) -> None:
        """
        Generate fairness data using cached retrieval results and save.

        Args:
            data_path (str): Path to input data file.
            save_path (str): Path to save generated data.
            batch_size (int): Size of each generation batch.
            count (int): Current generation count for file naming.

        Returns:
            None
        """
        with open(data_path, "r", encoding='utf-8') as f:
            data = json.load(f)
        
        prompt = config.fairness_gen
        for i in tqdm(range(0, len(data), batch_size)):
            batch_data = data[i:i+batch_size]
            input_texts = [prompt.format(query_str=d['prompt'][29:], context_str="\n\n".join(d['docs'])) for d in batch_data]
            res = self.llm.generate(input_texts)

            for d, r in zip(batch_data, res):
                d['res'] = r

        os.makedirs(save_path, exist_ok=True)
        save_file = os.path.join(save_path, "stereotype_agreement" + f"_{count}.json")
        with open(save_file, "w", encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Results saved to {save_file}")

    def cache_gen_crag(self, data_path: str, save_path: str, batch_size: int, count: int) -> None:
        """
        Generate CRAG data using cached retrieval results and save.

        Args:
            data_path (str): Path to input data file.
            save_path (str): Path to save generated data.
            batch_size (int): Size of each generation batch.
            count (int): Current generation count for file naming.

        Returns:
            None
        """
        with open(data_path, "r", encoding='utf-8') as f:
            data = json.load(f)
        
        prompt = config.crag_gen
        for i in tqdm(range(0, len(data), batch_size)):
            batch_data = data[i:i+batch_size]
            input_texts = [prompt.format(query_str=d['query'], query_time=d['query_time'], context_str="\n\n".join(d['docs'])) for d in batch_data]
            res = self.llm.generate(input_texts)

            for d, r in zip(batch_data, res):
                d['res'] = r

        os.makedirs(save_path, exist_ok=True)
        save_file = os.path.join(save_path, "crag" + f"_{count}.json")
        with open(save_file, "w", encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Results saved to {save_file}")

if __name__ == "__main__":
    retriever = hfRetriever(
        model_path="epoch_2_retriever",
        faiss_index_path="epoch_2_index_wiki",
        doc_path="wiki_doc.pkl"
    )
    torch.cuda.empty_cache()
    llm = hfLLM(model_name="/data/share_weight/Meta-Llama-3-8B-Instruct", log_path="8b-gen.jsonl")
    rag = RAG(llm, retriever)

    rag.gen_fairness("datasets/fairness/stereotype_agreement.json", "results/generations/fairness", direct_gen=False, batch_size=1, count="8b-train-wiki-k3")
    rag.gen_crag("datasets/crag/query.jsonl", "results/generations/crag", direct_gen=False, batch_size=1, count="8b-train-wiki-k3")
