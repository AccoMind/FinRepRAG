import os
from pathlib import Path
import re
from typing import List, Dict
import json
from datetime import datetime
import hashlib

from langchain_docling.loader import DoclingLoader, ExportType
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from docling.chunking import HybridChunker

# Configuration
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
GEN_MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"
EXPORT_TYPE = ExportType.DOC_CHUNKS

class PersistentKnowledgeBaseBuilder:
    def __init__(self,
                 drive_folder_path: str,
                 milvus_uri: str,
                 milvus_user: str,
                 milvus_password: str,
                 embed_model_id: str,
                 collection_name: str = "cse_annual_reports"):
        """
        Initialize the Knowledge Base Builder with persistence support.

        Args:
            drive_folder_path: Path to Google Drive folder containing annual reports
            milvus_uri: Zilliz Cloud URI
            milvus_user: Zilliz Cloud username
            milvus_password: Zilliz Cloud password
            embed_model_id: HuggingFace model ID for embeddings
            collection_name: Name for the Milvus collection
        """
        self.drive_folder_path = drive_folder_path
        self.collection_name = collection_name

        # Mount Google Drive
        # drive.mount('/content/drive', force_remount=True)

        # Initialize embedding model
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=embed_model_id
        )

        # Setup Milvus connection
        self.connection_args = {
            "uri": milvus_uri,
            "user": milvus_user,
            "password": milvus_password,
            "secure": True
        }

        # Initialize/connect to vector store
        self._initialize_vectorstore()

        # Setup processing history tracking
        self.history_file = Path(drive_folder_path) / "processing_history.json"
        self.processed_files = self._load_processing_history()

    def _initialize_vectorstore(self):
        """Initialize or connect to existing vector store."""
        try:
            self.vectorstore = Milvus(
                embedding_function=self.embedding_model,
                collection_name=self.collection_name,
                connection_args=self.connection_args,
                auto_id=True
            )
            print(f"Connected to existing collection: {self.collection_name}")
        except Exception as e:
            print(f"Creating new collection: {self.collection_name}")
            self.vectorstore = None

    def _load_processing_history(self) -> Dict:
        """Load processing history from file."""
        if self.history_file.exists():
            with open(self.history_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_processing_history(self):
        """Save processing history to file."""
        with open(self.history_file, 'w') as f:
            json.dump(self.processed_files, f, indent=2)

    def _compute_file_hash(self, file_path: str) -> str:
        """Compute SHA-256 hash of file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def extract_metadata(self, filename: str, file_hash: str) -> Dict:
        """Extract metadata from filename and add processing info."""
        pattern = r"(.+)_Annual_Report_(\d{4})\.pdf"
        match = re.match(pattern, filename)
        if match:
            company_name, year = match.groups()
            return {
                "company_name": company_name,
                "year": int(year),
                "file_hash": file_hash,
                "processed_date": datetime.now().isoformat(),
                "collection": self.collection_name
            }
        raise ValueError(f"Filename {filename} doesn't match expected pattern")

    def process_document(self, file_path: str) -> List:
        """Process a single document and return chunks with metadata."""
        file_hash = self._compute_file_hash(file_path)

        # Check if file was already processed and hasn't changed
        filename = Path(file_path).name
        if filename in self.processed_files and self.processed_files[filename]["hash"] == file_hash:
            print(f"Skipping {filename} - already processed and unchanged")
            return []

        loader = DoclingLoader(
            file_path=file_path,
            export_type=EXPORT_TYPE,
            chunker=HybridChunker(tokenizer=self.embedding_model.model_name)  # Use the actual model name
        )

        docs = loader.load()
        metadata = self.extract_metadata(filename, file_hash)

        # Add metadata to each chunk
        for doc in docs:
            doc.metadata.update(metadata)

        # Update processing history
        self.processed_files[filename] = {
            "hash": file_hash,
            "processed_date": metadata["processed_date"],
            "chunk_count": len(docs)
        }

        return docs

    def build_or_update_knowledge_base(self) -> Milvus:
        """Build or update the knowledge base with new/modified documents."""
        # Get all PDF files in the folder
        pdf_files = [f for f in os.listdir(self.drive_folder_path) if f.endswith('.pdf')]

        if not pdf_files:
            raise ValueError(f"No PDF files found in {self.drive_folder_path}")

        print(f"Found {len(pdf_files)} PDF files to process")

        all_docs = []
        for pdf_file in pdf_files:
            print(f"Processing {pdf_file}...")
            file_path = os.path.join(self.drive_folder_path, pdf_file)
            try:
                chunks = self.process_document(file_path)
                all_docs.extend(chunks)
                if chunks:
                    print(f"Successfully processed {pdf_file} into {len(chunks)} chunks")
            except Exception as e:
                print(f"Error processing {pdf_file}: {str(e)}")

        if all_docs:
            print(f"Adding {len(all_docs)} new chunks to vector store...")
            if self.vectorstore is None:
                # Create new vector store if it doesn't exist
                self.vectorstore = Milvus.from_documents(
                    documents=all_docs,
                    embedding=self.embedding_model,
                    collection_name=self.collection_name,
                    connection_args=self.connection_args,
                    index_params={"index_type": "FLAT"},
                )
            else:
                # Add new documents to existing vector store
                self.vectorstore.add_documents(all_docs)

            # Save processing history
            self._save_processing_history()
            print("Knowledge base updated successfully!")
        else:
            print("No new or modified documents to process")

        return self.vectorstore

    def get_build_stats(self) -> Dict:
        """Get statistics about the knowledge base."""
        if self.vectorstore is None:
            return {"status": "Not initialized"}

        collection = self.vectorstore.col
        stats = collection.get_statistics()

        # Get processing history stats
        processed_files = len(self.processed_files)
        total_chunks = sum(info["chunk_count"] for info in self.processed_files.values())

        return {
            "total_documents": processed_files,
            "total_chunks": total_chunks,
            "vector_store_chunks": stats["row_count"],
            "embedding_dimension": stats["dim"],
            "last_update": max(info["processed_date"]
                             for info in self.processed_files.values()) if self.processed_files else None
        }
