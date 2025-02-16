import datetime
import logging
import os
from pathlib import Path
from typing import Optional, List, Dict

import tqdm

from document_processor import DocumentProcessor
from docling.document_converter import DocumentConverter
from langchain.docstore.document import Document

from milvus_manager import MilvusManager

class KnowledgeBaseBuilder:
    """
    Builds and maintains a knowledge base with context-aware document chunking.

    Args:
        milvus_manager: Instance of MilvusManager for vector store operations
        doc_processor: Instance of DocumentProcessor for document handling
        doc_converter: Optional custom document converter
        logging_enabled: Whether to enable detailed logging
    """
    
    def __init__(self,
                 milvus_manager: MilvusManager,
                 doc_processor: DocumentProcessor,
                 doc_converter: Optional[DocumentConverter] = None,
                 logging_enabled: bool = True):
        self.milvus_manager = milvus_manager
        self.doc_processor = doc_processor
        self.doc_converter = doc_converter
        
        # Setup logging
        if logging_enabled:
            self._setup_logging()
        
        # Mount cloud storage
        self._mount_cloud_storage()
    
    def _setup_logging(self):
        """Configure logging for the builder."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"kb_builder_{datetime.now().strftime('%Y%m%d')}.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _mount_cloud_storage(self):
        """Mount cloud storage with error handling."""
        try:
            if 'google.colab' in str(get_ipython()): # type: ignore
                from google.colab import drive # type: ignore
                drive.mount('/content/drive', force_remount=True)
                self.logger.info("Successfully mounted Google Drive")
            else:
                self.logger.info("No cloud storage to mount")
        except Exception as e:
            self.logger.error(f"Failed to mount cloud storage: {str(e)}")
            raise
    
    def _validate_document(self, file_path: str) -> bool:
        """
        Validate document before processing.
        
        Args:
            file_path: Path to the document
            
        Returns:
            bool: Whether document is valid for processing
        """
        try:
            if not os.path.exists(file_path):
                self.logger.error(f"File not found: {file_path}")
                return False
            
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                self.logger.error(f"Empty file: {file_path}")
                return False
                
            # Add additional validation as needed
            return True
        except Exception as e:
            self.logger.error(f"Validation error for {file_path}: {str(e)}")
            return False
    
    def process_document(self, file_path: str) -> List[Document]:
        """
        Process a single document with context-aware chunking.
        
        Args:
            file_path: Path to the document
            
        Returns:
            List of processed document chunks with context
        """
        if not self._validate_document(file_path):
            return []
            
        try:
            self.logger.info(f"Processing document: {file_path}")
            
            # Get processed chunks with context
            chunks = self.doc_processor.process_document(file_path)
            
            if not chunks:
                self.logger.info(f"No new chunks generated for {file_path}")
                return []
                
            self.logger.info(f"Successfully processed {file_path}: {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {str(e)}")
            return []
    
    def build_or_update_knowledge_base(self, 
                                     file_types: List[str] = ['.pdf'],
                                     batch_size: int = 100) -> Dict:
        """
        Build or update the knowledge base with new/modified documents.
        
        Args:
            file_types: List of file extensions to process
            batch_size: Number of chunks to process in each batch
            
        Returns:
            Dict containing build statistics
        """
        try:
            files = []
            for file_type in file_types:
                files.extend([
                    f for f in os.listdir(self.doc_processor.folder_path)
                    if f.endswith(file_type)
                ])
            
            if not files:
                self.logger.warning(f"No files found in {self.doc_processor.folder_path}")
                return {"status": "No files found", "processed_files": 0, "total_chunks": 0}
            
            self.logger.info(f"Found {len(files)} files to process")
            
            # Process documents and collect chunks
            all_chunks = []
            processed_count = 0
            
            for file in tqdm(files, desc="Processing documents"):
                file_path = os.path.join(self.doc_processor.folder_path, file)
                chunks = self.process_document(file_path)
                
                if chunks:
                    all_chunks.extend(chunks)
                    processed_count += 1
                
                # Batch processing to manage memory
                if len(all_chunks) >= batch_size:
                    self._update_vector_store(all_chunks)
                    all_chunks = []
            
            # Process any remaining chunks
            if all_chunks:
                self._update_vector_store(all_chunks)
            
            build_stats = self.get_build_stats()
            build_stats.update({
                "status": "Success",
                "processed_files": processed_count,
                "total_input_files": len(files)
            })
            
            self.logger.info("Knowledge base update completed successfully")
            return build_stats
            
        except Exception as e:
            self.logger.error(f"Error building knowledge base: {str(e)}")
            raise
    
    def _update_vector_store(self, chunks: List[Document]):
        """
        Update vector store with new chunks.
        
        Args:
            chunks: List of document chunks to add
        """
        try:
            if self.milvus_manager.vectorstore is None:
                self.logger.info("Creating new vector store")
                self.milvus_manager.create_vectorstore(chunks)
            else:
                self.logger.info(f"Adding {len(chunks)} chunks to existing vector store")
                self.milvus_manager.add_documents(chunks)
        except Exception as e:
            self.logger.error(f"Error updating vector store: {str(e)}")
            raise
    
    # TODO: Implement this method. For now, return an empty dict.
    def get_build_stats(self) -> Dict:
        """
        Get comprehensive statistics about the knowledge base.
        
        Returns:
            Dict containing various statistics about the knowledge base
        """
        return {}
        