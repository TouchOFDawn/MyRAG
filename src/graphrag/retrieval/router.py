import logging
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from graphrag.retrieval.base import BaseRetriever

logger = logging.getLogger(__name__)

class QueryRouter:
    """使用本地小模型决定检索策略并执行检索"""

    def __init__(self, retrievers: List[BaseRetriever]):
        self.vector_retriever = retrievers[0]
        self.graph_retriever = retrievers[1]

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-0.5B-Instruct",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True,
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

    def _decide_route(self, query: str) -> str:
        """
        返回: 'vector', 'graph', 或 'hybrid'
        """
        prompt = f"""Classify the query type. Reply with ONE word only.

Rules:
- vector: simple fact lookup, "what is X", "define X", single entity
- graph: relationships, "how X relates to Y", "connection between", "impact of X on Y"
- hybrid: complex multi-aspect questions

Examples:
Q: What is machine learning? A: vector
Q: How does AI impact healthcare? A: graph
Q: Explain neural networks and their applications in medicine A: hybrid

Query: {query}
Answer:"""

        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=10, temperature=0.1)

        result = self.tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip().lower()

        # 提取关键词
        if "vector" in result:
            return "vector"
        elif "graph" in result:
            return "graph"
        else:
            return "hybrid"

    def retrieve(self, query: str, top_k: int = 4) -> str:
        """根据路由决策执行检索"""
        route = self._decide_route(query)
        logger.info(f"Router decision: {route}")

        if route == "vector":
            results = self.vector_retriever.retrieve(query, top_k=top_k)
            return self._format_results([("VectorRetriever", r) for r in results])
        elif route == "graph":
            results = self.graph_retriever.retrieve(query, top_k=top_k)
            return self._format_results([("Neo4jRetriever", r) for r in results])
        else:
            vec_results = self.vector_retriever.retrieve(query, top_k=top_k)
            graph_results = self.graph_retriever.retrieve(query, top_k=top_k)
            combined = [("VectorRetriever", r) for r in vec_results] + [("Neo4jRetriever", r) for r in graph_results]
            return self._format_results(combined)

    def _format_results(self, combined_results: List[tuple]) -> str:
        """格式化检索结果"""
        if not combined_results:
            return "No relevant context found."

        formatted = "### Retrievers Context ###\n"
        for source_name, res in combined_results:
            formatted += f"\n--- Source: {source_name} ---\n"
            formatted += f"Content: {res.content}\n"
            if res.metadata:
                metadata_str = " | ".join([f"{k}: {v}" for k, v in res.metadata.items()])
                formatted += f"Metadata: {metadata_str}\n"

        return formatted
