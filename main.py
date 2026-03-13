#Set HF
import os
os.environ["HF_HOME"] = "D:/huggingface_cache" 
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" 

import logging
from graphrag.data.loader import MultimodalDataLoader
from graphrag.data.directory import DirectoryProcessor
from graphrag.splitters.markdown import SemanticMarkdownSplitter
from graphrag.graph.builder import KnowledgeGraphBuilder
from graphrag.db.neo4j_manager import Neo4jManager
from graphrag.db.vector_store import VectorStoreManager
from graphrag.retrieval.vector_retriever import VectorRetriever
from graphrag.retrieval.graph_retriever import Neo4jRetriever
from graphrag.retrieval.router import QueryRouter
from graphrag.generation.generator import GraphRAGGenerator
from graphrag.config import settings
from graphrag.utils.state import IndexStateTracker



# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting MultiGraph-RAG System Demo")
    logger.info(f"Using Provider: {settings.model_type.upper()}")
    
    # --- Check Index State Cache ---
    input_dir = "./data/input"
    state_tracker = IndexStateTracker()
    current_hash = state_tracker.compute_directory_hash(input_dir)
    last_hash = state_tracker.get_last_hash()
    
    neo4j_manager = Neo4jManager()
    vector_store = VectorStoreManager()
    
    if current_hash and current_hash == last_hash:
        logger.info(f"Index Cache Hit! The contents of '{input_dir}' haven't changed.")
        logger.info("Skipping Data Loading, Chunking, Graph Building, and Database Ingestion.")
    else:
        logger.info(f"Index Cache Miss! Processing new data from '{input_dir}'.")
        
        # --- 1. Load Documents from Directory ---
        logger.info("=== Phase 1: Data Loading ===")
        dir_processor = DirectoryProcessor(input_dir=input_dir, output_dir="./data/output")
        documents = dir_processor.process()
        
        if not documents:
            logger.error("No valid documents found in ./data/input. Exiting.")
            return

        # --- 2. Semantic Chunking ---
        logger.info("\n=== Phase 2: Semantic Chunking ===")
        splitter = SemanticMarkdownSplitter(chunk_size=500, chunk_overlap=50)
        all_chunks = []
        for doc in documents:
            chunks = splitter.split_document(doc)
            all_chunks.extend(chunks)
            
        logger.info(f"Generated a total of {len(all_chunks)} chunks from {len(documents)} documents.")

        # --- 3. Build Knowledge Graph ---
        logger.info("\n=== Phase 3: Graph Building & Vectors ===")
        builder = KnowledgeGraphBuilder()
        
        try:
            graph_extraction, node_embeddings = builder.process_chunks(all_chunks)
            logger.info(f"Extracted {len(graph_extraction.nodes)} nodes and {len(graph_extraction.edges)} edges.")
        except Exception as e:
            logger.error(f"Failed to build graph (is API key configured?): {e}")
            return

        # --- 4. Database Ingestion ---
        logger.info("\n=== Phase 4: Database Ingestion ===")
        
        # Store text chunks into Chroma
        vector_store.ingest_chunks(all_chunks)
        
        # Store graph into Neo4j
        neo4j_manager.ingest_graph_extraction(graph_extraction)
        
        # Store nodes into Chroma
        if node_embeddings:
            node_texts = [f"{n.id}: {n.description}" for n in graph_extraction.nodes]
            node_ids = [n.id for n in graph_extraction.nodes]
            node_metadatas = [{"id": n.id, "type": n.type} for n in graph_extraction.nodes]
            vector_store.ingest_nodes(node_texts, node_metadatas, node_ids)
            
        # Update Index State Cache
        if current_hash:
            state_tracker.save_hash(current_hash)

    # --- 5. Hybrid Retrieval & Generation (Interactive) ---
    logger.info("\n=== Phase 5: Augmented Generation (Interactive Mode) ===")

    # Initialize router with retrievers
    vector_retriever = VectorRetriever(vector_store)
    graph_retriever = Neo4jRetriever(neo4j_manager)

    router = QueryRouter(retrievers=[vector_retriever, graph_retriever])
    generator = GraphRAGGenerator(router)
    
    # --- Mute verbose loggers before interactive session ---
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logger.setLevel(logging.WARNING) # Mute our own info logs during chat
    
    print("\n" + "="*50)
    print("MultiGraph-RAG System Ready!")
    print("Type your questions below. Type 'exit' or 'quit' or press Ctrl+C to stop.")
    print("="*50 + "\n")
    
    while True:
        try:
            query = input("\nUser Query: ")
            if query.lower() in ["exit", "quit"]:
                logger.info("Exiting interactive mode. Goodbye!")
                break
                
            if not query.strip():
                continue
                
            answer = generator.generate(query)
            print(f"\nModel Answer:\n{answer}\n")
            
        except KeyboardInterrupt:
            print("\nExiting interactive mode. Goodbye!")
            break
        except Exception as e:
            print(f"Error during generation: {e}")

if __name__ == "__main__":
    main()
