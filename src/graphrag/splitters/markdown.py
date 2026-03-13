import logging
from typing import List
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from graphrag.data.loader import DocumentBase

logger = logging.getLogger(__name__)

class SemanticMarkdownSplitter:
    """
    Splits normalized Markdown documents into semantic chunks while preserving headers
    and image relationships.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        # We split on standard markdown headers
        self.headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
        ]
        
        self.md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on,
            strip_headers=False
        )
        
        # After splitting by headers, we enforce a max chunk size
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def split_document(self, doc: DocumentBase) -> List[Document]:
        """
        Splits a DocumentBase into LangChain Documents with preserved metadata.
        """
        logger.info("Splitting markdown document semantically...")
        
        # 1. Split by Markdown headers first
        header_splits = self.md_splitter.split_text(doc.content)
        
        # 2. Add document-level metadata to each split
        for split in header_splits:
            split.metadata.update(doc.metadata)
            
            # Find which images are mentioned in this chunk
            chunk_images = []
            for img_path in doc.images:
                if img_path in split.page_content:
                    chunk_images.append(img_path)
            if chunk_images:
                split.metadata["images"] = chunk_images
                
        # 3. Recursively split large chunks ensuring they fit the chunk size
        final_chunks = self.text_splitter.split_documents(header_splits)
        
        logger.info(f"Generated {len(final_chunks)} chunks from document.")
        return final_chunks
