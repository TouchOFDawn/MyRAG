import logging
from typing import List
from graphrag.retrieval.base import BaseRetriever

logger = logging.getLogger(__name__)

class HybridRetriever:
    """
    Combines results from multiple underlying retrievers (Vector, Graph, etc.)
    """
    
    def __init__(self, retrievers: List[BaseRetriever]):
        """
        Initializes the HybridRetriever with a list of backend retrievers.
        """
        self.retrievers = retrievers

    def retrieve(self, query: str, top_k: int = 4) -> str:
        """
        Executes parallel or sequential retrieval across all configured backend 
        retrievers and formats the combined results into a single context string.
        """
        logger.info(f"Executing Hybrid Retrieval for: '{query}'")
        
        all_results = []
        for retriever in self.retrievers:
            try:
                # Top_k can be divided or configured per retriever in the future
                results = retriever.retrieve(query, top_k=top_k)
                all_results.extend([(retriever.__class__.__name__, res) for res in results])
            except Exception as e:
                logger.error(f"Error during retrieval in {retriever.__class__.__name__}: {e}")
        
        return self._format_results(all_results)
        
    def _format_results(self, combined_results: List[tuple]) -> str:
        """Formats the unified retriever results for LLM consumption."""
        if not combined_results:
            return "No relevant context found."
            
        formatted = "### Retrievers Context ###\n"
        
        # We can group by retriever type for clarity
        for source_name, res in combined_results:
            formatted += f"\n--- Source: {source_name} ---\n"
            formatted += f"Content: {res.content}\n"
            
            # Optionally format metadata if useful for the agent
            if res.metadata:
                metadata_str = " | ".join([f"{k}: {v}" for k, v in res.metadata.items()])
                formatted += f"Metadata: {metadata_str}\n"
                
        return formatted
