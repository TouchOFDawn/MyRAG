import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

class IndexStateTracker:
    """
    Tracks the state of the input directory to prevent redundant database ingestion.
    It calculates a hash based on the file names, sizes, and modification times
    of all files in the input directory.
    """

    def __init__(self, state_file: str = ".index_state.json"):
        self.state_file = Path(state_file)

    def compute_directory_hash(self, directory_path: str) -> str:
        """
        Computes an MD5 hash of the directory contents based on file metadata.
        """
        dir_path = Path(directory_path)
        if not dir_path.exists() or not dir_path.is_dir():
            return ""

        hasher = hashlib.md5()
        
        # Sort files to ensure consistent hashing order
        files = sorted([f for f in dir_path.rglob('*') if f.is_file()])
        
        for file_path in files:
            # Skip hidden files or specific extensions if needed
            if file_path.name.startswith('.'):
                continue
                
            try:
                stat = file_path.stat()
                # Include relative path, size, and modification time in hash
                file_info = f"{file_path.relative_to(dir_path)}|{stat.st_size}|{stat.st_mtime}"
                hasher.update(file_info.encode('utf-8'))
            except OSError as e:
                logger.warning(f"Failed to stat {file_path}: {e}")

        return hasher.hexdigest()

    def get_last_hash(self) -> Optional[str]:
        """Reads the last saved hash from the state file."""
        if not self.state_file.exists():
            return None
            
        try:
            with open(self.state_file, 'r', encoding='utf-8') as f:
                state = json.load(f)
                return state.get('input_dir_hash')
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to read index state file: {e}")
            return None

    def save_hash(self, current_hash: str) -> None:
        """Saves the current hash to the state file."""
        try:
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump({'input_dir_hash': current_hash}, f, indent=2)
            logger.info(f"Saved new index state hash: {current_hash}")
        except OSError as e:
            logger.warning(f"Failed to save index state file: {e}")
