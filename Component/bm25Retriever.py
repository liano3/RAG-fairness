import bm25s
import Stemmer
import sys
import os
import pickle
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from tqdm import tqdm
from typing import *

import config

class bm25Retriever:
    def __init__(
        self, 
        corpus_path: str = "corpus/web/txt", 
        index_path: str = "bm25_index", 
        doc_path: str = "doc.pkl",
        chunk_size: int = config.chunk_size,
        overlap_size: int = config.chunk_overlap
    ) -> None:
        """
        Initialize the bm25Retriever class and load or create the BM25 index.
        """
        self.stemmer = Stemmer.Stemmer("english")
        
        # Load the BM25 model if it exists
        if os.path.exists(index_path):
            self.retriever = bm25s.BM25.load(index_path)
            self.nodes: List[str] = pickle.load(open(doc_path, "rb"))
            return
        
        if os.path.exists(doc_path):
            self.nodes = pickle.load(open(doc_path, "rb"))
        else:
            # Load and split documents
            documents: List[Document] = []
            for file in tqdm(sorted(os.listdir(corpus_path))):  # Use sorted to ensure file order for subsequent checks
                with open(os.path.join(corpus_path, file), "r", encoding="utf-8") as f:
                    lines = f.readlines()
                # Treat each line as a document, remove too short documents, and remove gibberish documents
                documents.extend([Document(text=line.strip()) for line in lines if len(line.strip()) > 1000 and not self.is_gibberish(line.strip())])
            
            parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=overlap_size)
            nodes = parser.get_nodes_from_documents(documents, show_progress=True)
            # Keep only the text, remove duplicates
            self.nodes = list(set([node.text.strip() for node in nodes if len(node.text.strip()) > 1000 and not self.is_gibberish(node.text.strip())]))
            with open(doc_path, "wb") as f:
                pickle.dump(self.nodes, f)

        # Tokenize the corpus and only keep the ids (faster and saves memory)
        corpus_tokens = bm25s.tokenize(self.nodes, stopwords="en", stemmer=self.stemmer)

        # Create the BM25 model and index the corpus
        self.retriever = bm25s.BM25()
        self.retriever.index(corpus_tokens)
        self.retriever.save(index_path)

    def is_gibberish(self, text: str, threshold: float = 0.6) -> bool:
        """
        Determine if the given text is gibberish.

        Args:
            text (str): The text string to check.
            threshold (float): The threshold ratio of printable characters, below which the text is considered gibberish.

        Returns:
            bool: True if the text is gibberish, False otherwise.
        """
        import string
        # Calculate the ratio of printable characters, if the ratio is below the threshold, consider it gibberish
        printable_chars = set(string.printable)  # Set of printable characters
        printable_count = sum(1 for char in text if char in printable_chars)
        # Calculate the ratio of printable characters
        ratio = printable_count / len(text)
        # If the ratio of printable characters is below the threshold, consider it gibberish
        return ratio < threshold

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """
        Retrieve the most relevant documents based on the query.

        Args:
            query (str): The user's query.
            top_k (int): The number of most relevant documents to return.

        Returns:
            List[str]: The list of retrieved documents.
        """
        # Query the corpus
        query_tokens = bm25s.tokenize(query, stopwords="en", stemmer=self.stemmer, show_progress=False)

        # Get top-k results as a tuple of (doc ids, scores). Both are arrays of shape (n_queries, k)
        results, scores = self.retriever.retrieve(query_tokens, corpus=self.nodes, k=top_k, show_progress=False)
        # print(scores)
        return results.tolist()[0]

    def batch_retrieve(self, queries: List[str], top_k: int = 3) -> List[List[str]]:
        """
        Batch retrieve the most relevant documents for multiple queries.

        Args:
            queries (List[str]): The list of user queries.
            top_k (int): The number of most relevant documents to return.

        Returns:
            List[List[str]]: The list of retrieved documents for each query.
        """
        return [self.retrieve(query, top_k) for query in queries], [[0] * top_k] * len(queries)


if __name__ == "__main__":
    retriever = bm25Retriever(corpus_path="corpus/wiki/txt", index_path="bm25_wiki", doc_path="wiki_doc.pkl")
    
    # Test search
    query = [
        "where did the ceo of salesforce previously work?",
        "who is the ceo of salesforce?",
    ]
    results, scores = retriever.batch_retrieve(query)
    print(results[0][0])
    print(scores[0][0])
    print("====================================")
    print(results[0][1])
    print(scores[0][1])
