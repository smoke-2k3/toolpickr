# Placeholder for FlatRetriever
from typing import List
from toolpickr.vectorstores.base import VectorStore, SearchResult
from toolpickr.embeddings.base import EmbeddingProvider

class FlatRetriever:
    def __init__(self, embedding_provider: EmbeddingProvider, vector_store: VectorStore):
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        Embeds a user query and searches the vector store for the best tools.
        """
        query_vector = self.embedding_provider.embed_text(query)
        return self.vector_store.search(query_vector, k=top_k)

    # For testing purposes
    def retrieve_batch(self, queries: List[str], top_k: int = 5) -> List[List[SearchResult]]:
        """
        Embeds multiple user queries and searches the vector store for the best tools.
        """
        query_vectors = self.embedding_provider.embed_batch(queries)
        return self.vector_store.search_batch(query_vectors, k=top_k)
