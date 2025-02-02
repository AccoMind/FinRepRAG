from docling.document_converter import DocumentConverter

source = "data\HATTON_NATIONAL_BANK_PLC 2023-48.pdf"  # document per local path or URL
converter = DocumentConverter()
result = converter.convert(source)
printable = result.document.export_to_markdown()
print(printable)