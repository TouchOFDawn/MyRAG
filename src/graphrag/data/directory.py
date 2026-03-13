import os
import logging
from pathlib import Path
from typing import List
from graphrag.data.loaders.base import DocumentBase
from graphrag.data.loader import MultimodalDataLoader

logger = logging.getLogger(__name__)

class DirectoryProcessor:
    """
    Scans a directory for readable data files and URLs, parsing them 
    into a unified collection of DocumentBase objects.
    """
    
    def __init__(self, input_dir: str = "./data/input", output_dir: str = "./data/output"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.loader_factory = MultimodalDataLoader(output_dir=str(self.output_dir))
        
        # Ensure input directory exists
        self.input_dir.mkdir(parents=True, exist_ok=True)

    def _extract_urls_from_file(self, file_path: Path) -> List[str]:
        """Reads a text file and yields valid-looking URLs."""
        urls = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    clean_line = line.strip()
                    if clean_line.startswith("http://") or clean_line.startswith("https://"):
                        urls.append(clean_line)
        except Exception as e:
            logger.error(f"Failed to read URLs from {file_path}: {e}")
        return urls

    def process(self) -> List[DocumentBase]:
        """
        Walks the input directory and delegates parsing to the DataLoaderFactory.
        Special handling for `url.txt` to parse contained links.
        """
        documents = []
        
        logger.info(f"Scanning directory: {self.input_dir}")
        for root, _, files in os.walk(self.input_dir):
            for file in files:
                # Ignore hidden system files (like macOS ._ prefixed files)
                if file.startswith('.'):
                    continue
                    
                file_path = Path(root) / file
                
                # Special handling for lists of URLs
                if file.lower() == "url.txt":
                    logger.info(f"Found URL list: {file_path}")
                    urls = self._extract_urls_from_file(file_path)
                    for url in urls:
                        try:
                            doc = self.loader_factory.load_document(url)
                            documents.append(doc)
                        except Exception as e:
                            logger.error(f"Failed to load URL {url}: {e}")
                    continue
                
                # Filter valid extensions for standard processing
                if file_path.suffix.lower() in ['.txt', '.md', '.csv', '.pdf', '.html']:
                    try:
                        doc = self.loader_factory.load_document(str(file_path))
                        documents.append(doc)
                    except Exception as e:
                        logger.error(f"Failed to load document {file_path}: {e}")
                else:
                    logger.debug(f"Skipping unsupported file type: {file_path}")
                    
        logger.info(f"Successfully processed {len(documents)} documents from directory.")
        return documents
