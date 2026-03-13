import logging
import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from urllib.parse import urlparse
from graphrag.data.loaders.base import BaseLoader, DocumentBase

logger = logging.getLogger(__name__)

class WebLoader(BaseLoader):
    """
    Fetches HTML from URLs and converts them to formatted Markdown.
    """
    
    def load(self, url: str) -> DocumentBase:
        parsed_url = urlparse(url)
        if not parsed_url.scheme:
            raise ValueError(f"Invalid URL: {url}")
            
        logger.info(f"Fetching content from: {url}")
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Optional: Extract title and clean up common junk tags
            for junk in soup(["script", "style", "nav", "footer", "aside"]):
                junk.decompose()
            
            title = soup.title.string if soup.title else url
            html_content = str(soup)
            
            # Convert to Markdown
            markdown_content = md(html_content, heading_style="ATX")
            
            return DocumentBase(
                content=markdown_content.strip(),
                metadata={"source": url, "title": title.strip()}
            )
            
        except Exception as e:
            logger.error(f"Failed to load web page {url}: {e}")
            raise
