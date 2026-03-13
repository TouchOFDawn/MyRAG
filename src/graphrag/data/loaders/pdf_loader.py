import logging
import os
from pathlib import Path
from graphrag.data.loaders.base import BaseLoader, DocumentBase
import pymupdf4llm

logger = logging.getLogger(__name__)

class PDFLoader(BaseLoader):
    """
    Loads and parses PDF documents using PyMuPDF4LLM.
    Extracts Markdown and saves embedded images locally.
    """
    
    def __init__(self, output_dir: str = "./data/output"):
        super().__init__(output_dir)

    def load(self, file_path: str) -> DocumentBase:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {file_path}")

        logger.info(f"Processing PDF with PyMuPDF4LLM: {file_path}")
        
        doc_name = path.stem
        img_dir = Path(self.output_dir) / doc_name / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        
        # PyMuPDF4LLM extracts the document to Markdown.
        # Setting write_images=True automatically saves images to image_path.
        # It also naturally embeds the relative paths into the Markdown.
        markdown_content = pymupdf4llm.to_markdown(
            doc=str(path),
            write_images=True,
            image_path=str(img_dir),
            image_format="png"
        )
        
        # Collect saved images
        doc_images = []
        if img_dir.exists():
            for img_file in img_dir.glob("*.png"):
                doc_images.append(str(img_file.absolute()))
                
        # Save Markdown file to output directory
        md_file_path = Path(self.output_dir) / doc_name / f"{doc_name}.md"
        with open(md_file_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        logger.info(f"Saved extracted markdown to: {md_file_path}")
            
        return DocumentBase(
            content=markdown_content,
            metadata={"source": str(path), "parsed_by": "PyMuPDF4LLM", "saved_path": str(md_file_path)},
            images=doc_images
        )