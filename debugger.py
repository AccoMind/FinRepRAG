from langchain_docling.loader import DoclingLoader, ExportType
from docling.chunking import HybridChunker
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat

pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = False

doc_converter = DocumentConverter(
    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
)

loader = DoclingLoader(
    converter=doc_converter,
    file_path="test/data/Docling Technical Report-5.pdf",
    chunker=HybridChunker(),
    export_type=ExportType.DOC_CHUNKS,
)

print("Loading documents...")

docs = loader.load()

print(f"Loaded {len(docs)} documents")

for i, doc in enumerate(docs):
    print(f"\n\nDocument {i + 1}:")
    # print(doc.page_content)
    print(doc.metadata)

# metadata_example = {
#     'source': 'test/data/DocLayNet.pdf', 
#     'dl_meta': {
#         'schema_name': 'docling_core.transforms.chunker.DocMeta', 
#         'version': '1.0.0', 
#         'doc_items': [
#             {'self_ref': '#/texts/2', 'parent': {'$ref': '#/body'}, 'children': [], 'label': 'text', 'prov': [{'page_no': 1, 'bbox': {'l': 90.96700286865234, 't': 658.3280029296875, 'r': 193.7310028076172, 'b': 611.760009765625, 'coord_origin': 'BOTTOMLEFT'}, 'charspan': [0, 74]}]}, 
#             {'self_ref': '#/texts/3', 'parent': {'$ref': '#/body'}, 'children': [], 'label': 'text', 'prov': [{'page_no': 1, 'bbox': {'l': 255.11599731445312, 't': 658.3280029296875, 'r': 357.8800048828125, 'b': 611.760009765625, 'coord_origin': 'BOTTOMLEFT'}, 'charspan': [0, 71]}]}, 
#             {'self_ref': '#/texts/4', 'parent': {'$ref': '#/body'}, 'children': [], 'label': 'text', 'prov': [{'page_no': 1, 'bbox': {'l': 419.2650146484375, 't': 658.3280029296875, 'r': 522.0289916992188, 'b': 611.760009765625, 'coord_origin': 'BOTTOMLEFT'}, 'charspan': [0, 71]}]}, 
#             {'self_ref': '#/texts/5', 'parent': {'$ref': '#/body'}, 'children': [], 'label': 'text', 'prov': [{'page_no': 1, 'bbox': {'l': 172.54299926757812, 't': 599.9429931640625, 'r': 275.3070068359375, 'b': 553.375, 'coord_origin': 'BOTTOMLEFT'}, 'charspan': [0, 72]}]}, 
#             {'self_ref': '#/texts/6', 'parent': {'$ref': '#/body'}, 'children': [], 'label': 'text', 'prov': [{'page_no': 1, 'bbox': {'l': 336.6929931640625, 't': 599.9429931640625, 'r': 439.4570007324219, 'b': 553.375, 'coord_origin': 'BOTTOMLEFT'}, 'charspan': [0, 68]}]}
#         ], 
#         'headings': ['DocLayNet: A Large Human-Annotated Dataset for Document-Layout Analysis'], 
#         'origin': {
#             'mimetype': 'application/pdf', 
#             'binary_hash': 7156212269791437020, 
#             'filename': 'DocLayNet.pdf'
#         }
#     }
# }
