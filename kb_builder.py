import json
import logging
import time
from pathlib import Path
from typing import Iterable

from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.document import ConversionResult
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions

# Set up logging
_log = logging.getLogger(__name__)

def export_documents(
    conv_results: Iterable[ConversionResult],
    output_dir: Path,
):
    """
    Export converted documents to markdown format.
    
    Args:
        conv_results: Iterable of conversion results
        output_dir: Directory where markdown files will be stored
    
    Returns:
        Tuple of (success_count, partial_success_count, failure_count)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    failure_count = 0
    partial_success_count = 0

    for conv_res in conv_results:
        if conv_res.status == ConversionStatus.SUCCESS:
            success_count += 1
            doc_filename = conv_res.input.file.stem
            
            # Export only to markdown format
            with (output_dir / f"{doc_filename}.md").open("w", encoding="utf-8") as fp:
                fp.write(conv_res.document.export_to_markdown())

        elif conv_res.status == ConversionStatus.PARTIAL_SUCCESS:
            _log.info(
                f"Document {conv_res.input.file} was partially converted with the following errors:"
            )
            for item in conv_res.errors:
                _log.info(f"\t{item.error_message}")
            partial_success_count += 1
        else:
            _log.info(f"Document {conv_res.input.file} failed to convert.")
            failure_count += 1

    _log.info(
        f"Processed {success_count + partial_success_count + failure_count} docs, "
        f"of which {failure_count} failed "
        f"and {partial_success_count} were partially converted."
    )
    return success_count, partial_success_count, failure_count

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Configure input and output paths
    input_dir = Path("data")
    output_dir = Path("output")
    
    # Get all PDF files from the data directory
    input_doc_paths = list(input_dir.glob("*.pdf"))
    
    if not input_doc_paths:
        _log.error(f"No PDF files found in {input_dir}")
        return

    # Configure pipeline options
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True

    # Initialize document converter with custom options
    doc_converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
            ),
        },
    )

    # Convert documents
    start_time = time.time()
    
    try:
        conv_results = doc_converter.convert_all(
            input_doc_paths,
            raises_on_error=False,
        )
        
        success_count, partial_success_count, failure_count = export_documents(
            conv_results, output_dir=output_dir
        )

        end_time = time.time() - start_time
        _log.info(f"Document conversion complete in {end_time:.2f} seconds.")

        if failure_count > 0:
            raise RuntimeError(
                f"Failed to convert {failure_count} out of {len(input_doc_paths)} documents."
            )

    except Exception as e:
        _log.error(f"An error occurred during conversion: {str(e)}")
        raise

if __name__ == "__main__":
    main()