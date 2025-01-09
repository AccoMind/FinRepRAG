import os
import concurrent.futures
import logging
from pathlib import Path
from docling.document_converter import DocumentConverter

# Setup logging
logging.basicConfig(
    filename="pdf_conversion.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def process_pdf(pdf_path, output_path):
    try:
        # Initialize the converter
        converter = DocumentConverter(pdf_path)

        # Convert PDF to text
        text_content = converter.to_text()

        # Save the output
        with open(output_path, "w", encoding="utf-8") as output_file:
            output_file.write(text_content)

        logging.info(f"Successfully processed: {pdf_path}")

    except Exception as e:
        logging.error(f"Failed to process {pdf_path}: {e}")


def get_output_path(input_path, base_input_dir, base_output_dir):
    relative_path = os.path.relpath(input_path, base_input_dir)
    return os.path.join(base_output_dir, relative_path.replace(".pdf", ".txt"))


def process_directory(base_input_dir, base_output_dir):
    # Ensure the base output directory exists
    Path(base_output_dir).mkdir(parents=True, exist_ok=True)

    pdf_files = []
    for root, _, files in os.walk(base_input_dir):
        for file in files:
            if file.endswith(".pdf"):
                pdf_files.append(os.path.join(root, file))

    # Use concurrency to speed up the processing
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_pdf = {}
        for pdf_file in pdf_files:
            output_file = get_output_path(pdf_file, base_input_dir, base_output_dir)

            # Ensure the output directory exists
            Path(os.path.dirname(output_file)).mkdir(parents=True, exist_ok=True)

            # Submit the task
            future = executor.submit(process_pdf, pdf_file, output_file)
            future_to_pdf[future] = pdf_file

        # Handle completed tasks
        for future in concurrent.futures.as_completed(future_to_pdf):
            pdf_file = future_to_pdf[future]
            try:
                future.result()
            except Exception as e:
                logging.error(f"Error processing {pdf_file}: {e}")

if __name__ == "__main__":
    base_input_dir = "annual_reports"  # Input directory containing PDFs
    base_output_dir = "annual_reports"  # Same directory structure for outputs

    process_directory(base_input_dir, base_output_dir)
