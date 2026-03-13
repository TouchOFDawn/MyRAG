import base64
import logging
from typing import List, Optional, Tuple
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from graphrag.config import settings
from graphrag.models.factory import ModelFactory

logger = logging.getLogger(__name__)

# --- Structured Output Schemas ---
class NodeSchema(BaseModel):
    id: str = Field(description="Unique identifier for the node (usually its name)")
    type: str = Field(description="Type of the node (e.g., PERSON, ORGANIZATION, CONCEPT, LOCATION)")
    description: str = Field(description="A brief description of the node extracted from context")

class EdgeSchema(BaseModel):
    source: str = Field(description="ID of the source node")
    target: str = Field(description="ID of the target node")
    relation: str = Field(description="Type of relationship between source and target (e.g., WORKS_FOR, IS_A)")
    description: str = Field(description="Explanation of why this relationship exists based on the text")

class GraphExtraction(BaseModel):
    nodes: List[NodeSchema] = Field(description="List of extracted nodes/entities", default_factory=list)
    edges: List[EdgeSchema] = Field(description="List of extracted edges/relationships", default_factory=list)

# --- Graph Builder Logic ---
class KnowledgeGraphBuilder:
    """
    Extracts knowledge graphs (entities and relations) and generates vector embeddings.
    """
    def __init__(self):
        # Instantiate Models
        self.llm = ModelFactory.get_llm(temperature=0.0)
        
        # If multimodal is enabled, we create a vision model proxy
        # Fallback to standard LLM if a separate vision model is not configured properly
        self.use_multimodal = settings.use_multimodal
        
        self.embeddings = ModelFactory.get_embeddings()##no fc，simple sentenceTransformers
        
        # Use LangChain 1.x structured output
        try:
            self.extractor = self.llm.with_structured_output(GraphExtraction)
        except NotImplementedError:
             logger.warning(f"LLM {self.llm.__class__.__name__} might not fully support with_structured_output. " 
                            "Using generic prompt-based fallback if needed.")
             self.extractor = self.llm.with_structured_output(GraphExtraction, method="json_mode")

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def extract_graph_from_document(self, doc: Document) -> Tuple[GraphExtraction, dict]:
        """
        Extracts Node/Edge schema from a document chunk, utilizing VLM if images are present.
        Returns the extraction and a dictionary of vector embeddings for the chunk's nodes.
        """
        images = doc.metadata.get("images", [])
        
        sys_msg = SystemMessage(
            content="""You are a top-tier Knowledge Graph extractor.
Your task is to extract entities (nodes) and their relationships (edges) from the provided context.
Nodes should have an ID, type, and description. 
Edges should clearly connect two extracted node IDs with a specific relation type.
Output exactly to the requested schema."""
        )

        user_content = [{"type": "text", "text": f"Context chunk:\n{doc.page_content}\n\nExtract the knowledge graph now."}]
        
        if self.use_multimodal and images:
            logger.info(f"Extracting graph using VLM for chunk with {len(images)} associated images.")
            for img_path in images:
                try:
                    img_b64 = self._encode_image(img_path)
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                    })
                except Exception as e:
                    logger.warning(f"Failed to encode image {img_path}: {e}")
                    
        human_msg = HumanMessage(content=user_content)
        
        # 1. LLM/VLM Extraction
        logger.info("Executing LLM pipeline for graph extraction...")
        extraction: GraphExtraction = self.extractor.invoke([sys_msg, human_msg])
        
        # 2. Vector Embedding Generation
        node_embeddings = {}
        if extraction and extraction.nodes:
            logger.info("Generating embeddings for extracted nodes.")
            node_texts = [f"{n.id}: {n.description}" for n in extraction.nodes]
            vectors = self.embeddings.embed_documents(node_texts)
            for node, vector in zip(extraction.nodes, vectors):
                node_embeddings[node.id] = vector
                
        return extraction, node_embeddings

    def process_chunks(self, chunks: List[Document]) -> Tuple[GraphExtraction, dict]:
        """
        Processes a list of partitioned chunks, aggregating the global graph and vector map.
        """
        global_nodes = {}
        global_edges = []
        global_embeddings = {}
        
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}")
            extr, embeds = self.extract_graph_from_document(chunk)
            
            if not extr: continue
            
            # Deduplicate nodes by ID
            for node in extr.nodes:
                if node.id not in global_nodes:
                    global_nodes[node.id] = node
                    
            global_edges.extend(extr.edges)
            global_embeddings.update(embeds)
            
        final_graph = GraphExtraction(
            nodes=list(global_nodes.values()),
            edges=global_edges
        )
        return final_graph, global_embeddings
