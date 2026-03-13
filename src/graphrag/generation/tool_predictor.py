import logging
import re
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor, Future
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

logger = logging.getLogger(__name__)

class ToolPredictor:
    """使用小模型预测工具调用并异步执行"""

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-0.5B-Instruct",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True,
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        self.executor = ThreadPoolExecutor(max_workers=3)
        self.cache: Dict[str, Future] = {}

    def _parse_prediction(self, text: str) -> List[Dict[str, Any]]:
        """解析小模型输出"""
        predictions = []
        lines = text.strip().split('\n')

        for line in lines:
            if 'TOOL:' in line and 'PARAM:' in line:
                match = re.search(r'TOOL:\s*(\w+)\s*\|\s*PARAM:\s*(.+)', line)
                if match:
                    tool_name = match.group(1).strip()
                    param_str = match.group(2).strip()
                    predictions.append({'tool': tool_name, 'params': param_str})

        return predictions

    def predict(self, query: str) -> List[Dict[str, Any]]:
        """使用小模型预测需要的工具"""
        prompt = f"""Analyze the query and predict which tools are needed. Reply in this format:
- If need to fetch a webpage: TOOL: fetch_url | PARAM: <url>
- If need weather info: TOOL: get_weather | PARAM: city=<city>, date=<date>
- If no tools needed: NONE

Examples:
Q: 帮我看看这个网页 https://example.com 讲了什么
A: TOOL: fetch_url | PARAM: https://example.com

Q: 明天北京天气怎么样
A: TOOL: get_weather | PARAM: city=北京, date=明天

Q: 什么是机器学习
A: NONE

Query: {query}
Answer:"""

        try:
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=50, temperature=0.1)

            content = self.tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip()
            return self._parse_prediction(content)
        except Exception as e:
            logger.warning(f"预测失败: {e}")
            return []

    def _execute_predictions(self, query: str, tools: Dict[str, Any]):
        """在后台线程中执行预测和工具调用"""
        predictions = self.predict(query)

        for pred in predictions:
            tool_name = pred['tool']
            params = pred['params']

            if tool_name == 'fetch_url':
                url = params
                cache_key = f"fetch_url:{url}"
                if cache_key not in self.cache:
                    future = self.executor.submit(tools['fetch_url'], url)
                    self.cache[cache_key] = future
                    logger.info(f"异步执行: fetch_url({url})")

            elif tool_name == 'get_weather':
                match = re.search(r'city=([^,]+)(?:,\s*date=(.+))?', params)
                if match:
                    city = match.group(1).strip()
                    date = match.group(2).strip() if match.group(2) else "today"
                    cache_key = f"get_weather:{city}:{date}"
                    if cache_key not in self.cache:
                        future = self.executor.submit(tools['get_weather'], city, date)
                        self.cache[cache_key] = future
                        logger.info(f"异步执行: get_weather({city}, {date})")

    def predict_and_execute_async(self, query: str, tools: Dict[str, Any]):
        """预测并异步执行工具调用（非阻塞）"""
        self.executor.submit(self._execute_predictions, query, tools)
