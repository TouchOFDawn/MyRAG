import logging
from pathlib import Path
from urllib.parse import urlparse
from graphrag.data.loaders.base import DocumentBase
from graphrag.data.loaders.pdf_loader import PDFLoader
from graphrag.data.loaders.web_loader import WebLoader
from graphrag.data.loaders.text_loader import TextLoader

logger = logging.getLogger(__name__)

class MultimodalDataLoader:
    """
    DataLoader Factory that routes to the appropriate parser for PDF, Web, and Text documents.
    """
    
    def __init__(self, output_dir: str = "./data/output"):
        self.output_dir = output_dir
        
        # Initialize specialized loaders
        self.pdf_loader = PDFLoader(output_dir)
        self.web_loader = WebLoader(output_dir)
        self.text_loader = TextLoader(output_dir)

    def _is_url(self, path: str) -> bool:
        try:
            result = urlparse(path)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False

    def load_document(self, file_path_or_url: str) -> DocumentBase:
        """
        Loads and parses a document based on its extension or protocol, returning normalized Markdown.
        """
        logger.info(f"Processing input: {file_path_or_url}")
        
        if self._is_url(file_path_or_url):
            return self.web_loader.load(file_path_or_url)
            
        path = Path(file_path_or_url)
        if not path.exists():
            raise FileNotFoundError(f"Input not found: {file_path_or_url}")

        if path.suffix.lower() == '.pdf':
            return self.pdf_loader.load(file_path_or_url)
        elif path.suffix.lower() in ['.txt', '.md', '.csv']:
            return self.text_loader.load(file_path_or_url)
        else:
            logger.warning(f"Unsupported extension {path.suffix}. Falling back to TextLoader.")
            return self.text_loader.load(file_path_or_url)
