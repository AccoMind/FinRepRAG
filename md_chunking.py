from langchain_docling.loader import DoclingLoader, ExportType
from docling.chunking import HybridChunker
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from langchain_text_splitters import MarkdownHeaderTextSplitter

pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = False

doc_converter = DocumentConverter(
    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
)

file_path = "test/data/Docling Technical Report-4.pdf"

# for MARKDOWN
loader = DoclingLoader(
    converter=doc_converter,
    file_path=file_path,
    export_type=ExportType.MARKDOWN,
)

print("Loading documents...")

docs = loader.load()

print(f"Loaded {len(docs)} documents")

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
md_header_splits = markdown_splitter.split_text(docs[0].page_content)

print(f"Loaded {len(md_header_splits)} documents")

for i, doc in enumerate(md_header_splits):
    print(f"\n\nDocument {i + 1}:")
    print(doc.page_content)
    doc.metadata.update({"source": file_path})
    print(f"\nMetadata")
    print(doc.metadata)


# metadata_example = {'Header 2': '3.4 Extensibility', 'source': 'test/data/Docling Technical Report-4.pdf'}
