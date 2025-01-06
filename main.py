from docling.document_converter import DocumentConverter

source = "2408.09869v5-5.pdf"  # PDF path or URL
converter = DocumentConverter()
result = converter.convert(source)
print(result.document.export_to_markdown())  # output: "### Docling Technical Report[...]"