import logging
import time
from typing import Any
from langchain.agents import create_agent
from langchain_core.tools import tool

from graphrag.models.factory import ModelFactory
from graphrag.retrieval.router import QueryRouter
from graphrag.generation.tool_predictor import ToolPredictor


logger = logging.getLogger(__name__)

class GraphRAGGenerator:
    """
    Agentic Generator using tool calling to retrieve context and generate answers,
    now supporting multi-turn conversations via chat history.
    """
    def __init__(self, router: QueryRouter):
        self.router = router
        self.llm = ModelFactory.get_llm(temperature=0.0)
        self.predictor = ToolPredictor()
        self.agent_executor = self._build_agent()
        self.chat_history = []

    def _build_agent(self) -> Any:
        """
        Builds the LangChain Agent using the modern create_agent API.
        """
        # Expose the retriever as a tool
        @tool
        def query_hybrid_graph(query: str) -> str:
            """
            Useful for querying the knowledge graph and vector store to find specific information,
            relationships, and context related to the user's query.
            """
            return self.router.retrieve(query, top_k=4)
        
        from graphrag.tools.weather_tool import get_weather_tool, set_predictor as set_weather_predictor, _get_weather_impl
        from graphrag.tools.web_fetcher import fetch_url_tool, set_predictor as set_web_predictor, _fetch_url_impl

        # 注入预测器到工具模块
        set_weather_predictor(self.predictor)
        set_web_predictor(self.predictor)

        tools = [query_hybrid_graph, get_weather_tool, fetch_url_tool]

        # 保存工具实现函数供预测器使用
        self.tool_impls = {
            'fetch_url': _fetch_url_impl,
            'get_weather': _get_weather_impl
        }
        
        system_prompt = (
            "你是一个智能助手，可以使用多种工具回答问题：\n"
            "- query_hybrid_graph: 查询知识图谱获取相关信息\n"
            "- get_weather_tool: 获取城市天气信息\n"
            "- fetch_url_tool: 抓取网页内容\n"
            "根据用户问题选择合适的工具，如果需要知识库信息则调用 query_hybrid_graph。"
        )

        # Create the modern tool calling agent
        logger.info("Initializing tool calling agent...")
        agent_graph = create_agent(
            model=self.llm,
            tools=tools,
            system_prompt=system_prompt,
            debug=True
        )
        
        return agent_graph

    def generate(self, query: str) -> str:
        """
        Runs the agent loop to answer a query, preserving context across turns.
        """
        logger.info(f"Generating answer for query: '{query}'")
        start_time = time.time()

        try:
            # 【非阻塞】启动小模型预测 + 异步工具调用
            if True:
                self.predictor.predict_and_execute_async(query, self.tool_impls)

            # Append the user's new message to the existing history
            from langchain_core.messages import HumanMessage
            self.chat_history.append(HumanMessage(content=query))

            inputs = {"messages": self.chat_history}
            response = self.agent_executor.invoke(inputs)

            elapsed = time.time() - start_time
            logger.info(f"总耗时: {elapsed:.2f}秒 (预测: False)")

            # The agent returns the full updated message history in response["messages"]
            messages = response.get("messages", [])
            if messages:
                # Update our class history with the agent's full execution trace
                self.chat_history = messages
                # The final answer is the content of the last message
                return messages[-1].content

            return "No response generated."
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return f"An error occurred during generation: {str(e)}"
