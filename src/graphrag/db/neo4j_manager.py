import logging
from typing import List
from langchain_community.graphs.neo4j_graph import Neo4jGraph
from graphrag.config import settings
from graphrag.graph.builder import GraphExtraction

logger = logging.getLogger(__name__)

class Neo4jManager:
    """
    Manages connections and transactions with the Neo4j Graph Database using LangChain integration.
    """
    def __init__(self):
        logger.info(f"Connecting to Neo4j at {settings.neo4j_uri}")
        try:
            self.graph = Neo4jGraph(
                url=settings.neo4j_uri,
                username=settings.neo4j_username,
                password=settings.neo4j_password,
            )
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}. Please ensure the Neo4j instance is running.")
            self.graph = None

    def ingest_graph_extraction(self, extraction: GraphExtraction):
        """
        Inserts structured nodes and edges into Neo4j using efficient Cypher queries.
        """
        if not self.graph:
            logger.warning("Neo4j graph not initialized. Skipping graph ingestion.")
            return

        logger.info(f"Ingesting {len(extraction.nodes)} nodes and {len(extraction.edges)} edges.")

        # 1. Ingest Nodes
        # Use APOC or basic Cypher UNWIND for batch insert
        nodes_data = [
            {"id": n.id, "type": n.type.upper().replace(" ", "_"), "desc": n.description}
            for n in extraction.nodes
        ]
        
        node_query = """
        UNWIND $data AS row
        CALL apoc.create.node([row.type, 'Entity'], {id: row.id, description: row.desc})
        YIELD node
        RETURN count(node)
        """
        
        # Fallback if APOC is not available
        node_query_fallback = """
        UNWIND $data AS row
        MERGE (n:Entity {id: row.id})
        SET n.description = row.desc
        // Dynamic labels are tricky in raw cypher without APOC, we set 'Entity' and save type as property
        SET n.type = row.type
        """

        try:
            self.graph.query(node_query_fallback, params={"data": nodes_data})
        except Exception as e:
            logger.error(f"Error executing Neo4j node query: {e}")

        # 2. Ingest Edges
        edges_data = [
            {"source": e.source, "target": e.target, "rel": e.relation.upper().replace(" ", "_"), "desc": e.description}
            for e in extraction.edges
        ]
        
        edge_query_fallback = """
        UNWIND $data AS row
        MERGE (s:Entity {id: row.source})
        MERGE (t:Entity {id: row.target})
        // As dynamic relationships also require APOC, we use a generic 'RELATED_TO' 
        // and set the specific relation type as an edge property
        MERGE (s)-[r:RELATED_TO]->(t)
        SET r.type = row.rel
        SET r.description = row.desc
        """
        
        try:
            self.graph.query(edge_query_fallback, params={"data": edges_data})
        except Exception as e:
            logger.error(f"Error executing Neo4j edge query: {e}")
            
        # Refresh LangChain Schema
        self.graph.refresh_schema()

    def get_schema(self) -> str:
        """Returns the current Neo4j schema formatted by LangChain."""
        if self.graph:
            return self.graph.schema
        return ""
