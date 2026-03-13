import logging
from typing import List, Optional
from graphrag.retrieval.base import BaseRetriever, RetrieverResult
from graphrag.db.neo4j_manager import Neo4jManager
from graphrag.models.factory import ModelFactory

logger = logging.getLogger(__name__)

class Neo4jRetriever(BaseRetriever):
    """
    Graph traversal retriever extending BaseRetriever.
    Backended by Neo4jManager. Utilizes an LLM to extract starting entities from the query.
    """
    
    def __init__(self, neo4j_manager: Neo4jManager, llm: Optional[object] = None):
        self.neo4j_manager = neo4j_manager
        # If no LLM passed, initialize a default one for entity extraction
        if not llm:
            factory = ModelFactory()
            self.llm = factory.get_llm()
        else:
            self.llm = llm

    def _extract_entities_from_query(self, query: str) -> List[str]:
        """Simple LLM call to extract key entities from the user query."""
        prompt = f"""
        Extract the key entities (names, places, concepts) from the following query. 
        Return ONLY a comma-separated list of entities.
        
        Query: {query}
        Entities:
        """
        response = self.llm.invoke(prompt)
        entities = [e.strip() for e in response.content.split(",") if e.strip()]
        logger.info(f"Extracted entities for graph search: {entities}")
        return entities

    def retrieve(self, query: str, top_k: int = 4) -> List[RetrieverResult]:
        logger.info(f"Neo4jRetriever executing search for: '{query}'")
        entities = self._extract_entities_from_query(query)
        
        if not entities:
            return []
            
        results = []
        for entity in entities:
            # Query the 1-hop neighborhood for each entity
            cypher_query = """
            MATCH (n)-[r]->(m)
            WHERE toLower(n.id) CONTAINS toLower($entity) 
               OR toLower(m.id) CONTAINS toLower($entity)
            RETURN n.id AS source, type(r) AS relation, r.description AS description, m.id AS target
            LIMIT $limit
            """
            
            records = self.neo4j_manager.graph.query(
                cypher_query, 
                params={"entity": entity, "limit": top_k}
            )
            
            for record in records:
                desc = record.get("description", "")
                rel_str = f"({record['source']}) -[{record['relation']}]-> ({record['target']}) | Desc: {desc}"
                results.append(RetrieverResult(
                    content=rel_str,
                    metadata={"source_entity": entity, "type": "graph_relation"}
                ))
                
        return results
