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
    def __init__(self):
        pass

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