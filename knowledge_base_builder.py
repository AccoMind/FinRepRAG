import os
from pathlib import Path
from typing import List, Dict
import re
from datetime import datetime

from langchain_docling.loader import DoclingLoader, ExportType
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from docling.chunking import HybridChunker

class KnowledgeBaseBuilder:
    """Class responsible for building and storing the knowledge base."""
    
    def __init__(self, folder_path: str, milvus_uri: str):
        """
        Initialize the Knowledge Base Builder.
        
        Args:
            drive_folder_path: Path to Google Drive folder containing annual reports
            milvus_uri: URI for Milvus/Zilliz Cloud connection
        """
        self.folder_path = folder_path
        self.milvus_uri = milvus_uri
        
        # Initialize embedding model
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    def extract_metadata(self, filename: str) -> Dict:
        """Extract company name and year from filename."""
        pattern = r"(.+)_Annual_Report_(\d{4})\.pdf"
        match = re.match(pattern, filename)
        if match:
            company_name, year = match.groups()
            return {
                "company_name": company_name.replace("_", " "),  # Convert underscores to spaces
                "year": int(year),
                "source": filename,
                "processed_date": datetime.now().isoformat()
            }
        raise ValueError(f"Filename {filename} doesn't match expected pattern")

    def process_document(self, file_path: str) -> List:
        """Process a single document and return chunks with metadata."""
        loader = DoclingLoader(
            file_path=file_path,
            export_type=ExportType.DOC_CHUNKS,
            chunker=HybridChunker(tokenizer="sentence-transformers/all-MiniLM-L6-v2")
        )
        
        docs = loader.load()
        metadata = self.extract_metadata(Path(file_path).name)
        
        # Add metadata to each chunk
        for doc in docs:
            doc.metadata.update(metadata)
        
        return docs

    def build_knowledge_base(self, collection_name: str = "cse_annual_reports") -> Milvus:
        """
        Build the knowledge base from all documents in the drive folder.
        
        Args:
            collection_name: Name for the Milvus collection
            
        Returns:
            Milvus vector store instance
        """
        # Get all PDF files in the folder
        pdf_files = [f for f in os.listdir(self.folder_path) if f.endswith('.pdf')]
        
        if not pdf_files:
            raise ValueError(f"No PDF files found in {self.folder_path}")
        
        print(f"Found {len(pdf_files)} PDF files to process")
        
        all_docs = []
        for pdf_file in pdf_files:
            print(f"Processing {pdf_file}...")
            file_path = os.path.join(self.folder_path, pdf_file)
            try:
                chunks = self.process_document(file_path)
                all_docs.extend(chunks)
                print(f"Successfully processed {pdf_file} into {len(chunks)} chunks")
            except Exception as e:
                print(f"Error processing {pdf_file}: {str(e)}")
        
        print(f"Creating vector store with {len(all_docs)} total chunks...")
        
        # Create vector store
        vectorstore = Milvus.from_documents(
            documents=all_docs,
            embedding=self.embedding_model,
            collection_name=collection_name,
            connection_args={"uri": self.milvus_uri},
            index_params={"index_type": "FLAT"},
        )
        
        print("Knowledge base built successfully!")
        return vectorstore

    def get_build_stats(self, vectorstore: Milvus) -> Dict:
        """Get statistics about the built knowledge base."""
        collection = vectorstore.col
        stats = collection.get_statistics()
        
        return {
            "total_chunks": stats["row_count"],
            "embedding_dimension": stats["dim"],
            "build_date": datetime.now().isoformat()
        }
    