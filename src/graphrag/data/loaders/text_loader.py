import logging
from pathlib import Path
from graphrag.data.loaders.base import BaseLoader, DocumentBase

logger = logging.getLogger(__name__)

class TextLoader(BaseLoader):
    """
    Loads raw text and markdown files directly.
    """
    
    def load(self, file_path: str) -> DocumentBase:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info(f"Processing Text document: {file_path}")
        content = path.read_text(encoding="utf-8")
        
        return DocumentBase(
            content=content,
            metadata={"source": str(path), "parsed_by": "TextLoader"}
        )
