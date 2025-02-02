from langchain_docling.loader import DoclingLoader, ExportType
from docling.chunking import HybridChunker

loader = DoclingLoader(
    file_path="test/data/DocLayNet.pdf",
    chunker=HybridChunker(),
)

docs = loader.load()

for doc in docs:
    metadata = doc.metadata
    metadata.update({"export_type": "pdf"})
    updated_metadata = metadata