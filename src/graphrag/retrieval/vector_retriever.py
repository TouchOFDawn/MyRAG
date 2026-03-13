import logging
from typing import List
from graphrag.retrieval.base import BaseRetriever, RetrieverResult
from graphrag.db.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)

class VectorRetriever(BaseRetriever):
    """
    Semantic search retriever extending BaseRetriever. 
    Can be backed by Chroma or FAISS managed by VectorStoreManager.
    """
    
    def __init__(self, vector_store: VectorStoreManager):
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 4) -> List[RetrieverResult]:
        logger.info(f"VectorRetriever executing search for: '{query}'")
        
        # We rely on the unified interface of vector_store manager
        raw_results = self.vector_store.similarity_search(query, k=top_k)
        
        results = []
        for doc in raw_results:
            results.append(
                RetrieverResult(
                    content=doc.page_content,
                    metadata=doc.metadata,
                    # Fallback score logic if underlying DB doesn't return scores by default
                    score=doc.metadata.get("score", 1.0) 
                )
            )
        return results
