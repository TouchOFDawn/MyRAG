from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from pydantic import BaseModel

class DocumentBase(BaseModel):
    """Normalized document representation across all loaders."""
    content: str
    metadata: Dict[str, str] = {}
    images: List[str] = []

class BaseLoader(ABC):
    """Abstract base class for all Document Loaders."""
    
    def __init__(self, output_dir: str = "./data/output"):
        self.output_dir = output_dir

    @abstractmethod
    def load(self, file_path_or_url: str) -> DocumentBase:
        """Loads data from a path or URL and returns normalized DocumentBase."""
        pass
