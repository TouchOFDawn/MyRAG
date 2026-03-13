import requests
from bs4 import BeautifulSoup
import html2text
import logging
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# 全局预测器引用（由 Generator 注入）
_predictor = None

def set_predictor(predictor):
    global _predictor
    _predictor = predictor

def _fetch_url_impl(url: str) -> str:
    """实际的网页抓取逻辑"""
    try:
        # Added realistic headers to bypass basic bot protection
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
            
        # Convert to Markdown
        h = html2text.HTML2Text()
        h.ignore_links = False
        h.ignore_images = True
        markdown_text = h.handle(str(soup))
        
        # Truncate if too long (e.g., arbitrarily at 15000 chars to avoid prompt bloat)
        max_chars = 15000
        if len(markdown_text) > max_chars:
            markdown_text = markdown_text[:max_chars] + "\n...[truncated]"
            
        return markdown_text
    except requests.exceptions.Timeout:
         return f"Error: Request to {url} timed out after 15 seconds."
    except Exception as e:
        return f"Error fetching URL: {str(e)}"

@tool("fetch_url")
def fetch_url_tool(url: str) -> str:
    """Fetches a URL and returns its textual representation (cleaned Markdown) to save tokens."""
    cache_key = f"fetch_url:{url}"

    if _predictor and cache_key in _predictor.cache:
        future = _predictor.cache[cache_key]
        try:
            logger.info(f"使用预测缓存: {cache_key}")
            return future.result(timeout=10)
        except Exception as e:
            logger.warning(f"预测工具失败: {e}")

    return _fetch_url_impl(url)
