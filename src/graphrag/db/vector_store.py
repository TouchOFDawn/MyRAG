import os
import logging
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from graphrag.models.factory import ModelFactory

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """
    Manages the Vector Database (Chroma) for semantic search over text chunks and nodes.
    """
    def __init__(self, persist_directory: str = "./data/chroma_db"):
        self.persist_directory = persist_directory
        self.embeddings = ModelFactory.get_embeddings()#no fc，simple sentenceTransformers
        
        # Ensure persistence directory exists
        os.makedirs(self.persist_directory, exist_ok=True)
        
        self.vector_store = Chroma(
            collection_name="graphrag_collection",
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )

    def ingest_chunks(self, chunks: List[Document]):
        """
        Adds text chunks into the vector store.
        """
        if not chunks:
            return
            
        logger.info(f"Ingesting {len(chunks)} document chunks into Vector Store.")
        self.vector_store.add_documents(chunks)
        # In langchain_community Chroma, it persists automatically or explicitly depending on version.
        # Calling persist just to be safe if it's the older API style.
        if hasattr(self.vector_store, "persist"):
            self.vector_store.persist()

    def ingest_nodes(self, node_texts: List[str], metadatas: List[Dict[str, Any]], ids: List[str]):
        """
        Adds node descriptions into the vector store for entity-based retrieval.
        """
        if not node_texts:
            return
            
        logger.info(f"Ingesting {len(node_texts)} node definitions into Vector Store.")
        self.vector_store.add_texts(texts=node_texts, metadatas=metadatas, ids=ids)
        if hasattr(self.vector_store, "persist"):
            self.vector_store.persist()

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Performs standard semantic search.
        """
        logger.info(f"Performing vector search for query: {query}")
        return self.vector_store.similarity_search(query, k=k)
