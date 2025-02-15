from pathlib import Path
from typing import List, Dict, Optional
from langchain_core.embeddings import Embeddings
from docling.document_converter import DocumentConverter
from langchain_milvus import Milvus

class KnowledgeBaseBuilder:
    def __init__(self, 
                 folder_path: str,
                 milvus_uri: str,
                 milvus_user: str, 
                 milvus_password: str,
                 embedding_model: Embeddings,
                 collection_name: str,
                 doc_converter: Optional[DocumentConverter] = None):
        self.folder_path = folder_path
        self.milvus_uri = milvus_uri
        self.milvus_user = milvus_user
        self.milvus_password = milvus_password
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.doc_converter = doc_converter

        # mount google drive only if it's colab
        if 'google.colab' in str(get_ipython()): # type: ignore
            from google.colab import drive # type: ignore
            drive.mount('/content/drive', force_remount=True)

        self.history_file = Path(folder_path) / "processing_history.json"
        self.processed_files = self._load_processing_history()

        self.vectorstore = self._initialize_vectorstore()
        self.connection_args = {
            "uri": self.milvus_uri, 
            "user": self.milvus_user, 
            "password": self.milvus_password,
            "secure": True
        }

    def _initialize_vectorstore(self):
        pass

    def _load_processing_history(self) -> Dict:
        pass

    def _save_processing_history(self):
        pass

    def _compute_file_hash(self, file_path: str) -> str:
        pass

    def extract_metadata(self, filename: str, file_hash: str) -> Dict:
        pass

    def process_document(self, file_path: str) -> List:
        pass

    def build_or_update_knowledge_base(self) -> Milvus:
        pass

    def get_build_stats(self) -> Dict:
        pass