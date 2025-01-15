import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Dict
import uuid
from dataclasses import dataclass, field

from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.document import ConversionResult
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions

logger = logging.getLogger(__name__)

@dataclass
class ChunkInfo:
    """Information about a specific chunk of text"""
    chunk_id: str
    content: str
    start_index: int
    end_index: int
    page_number: Optional[int]
    section_title: Optional[str]
    
@dataclass
class DocumentChunks:
    """Container for document chunks with metadata"""
    document_id: str
    source_path: Path
    title: str
    chunks: List[ChunkInfo]
    metadata: Dict

@dataclass
class ProcessingResult:
    """Enhanced processing result with chunk information"""
    markdown_content: str
    source_path: Path
    metadata: Dict
    success: bool
    document_id: str
    chunks: List[ChunkInfo] = field(default_factory=list)
    error_messages: List[str] = field(default_factory=list)

class PDFKnowledgeProcessor:
    """
    Enhanced processor class with chunk tracking capabilities.
    """
    
    def __init__(
        self,
        enable_ocr: bool = False,
        enable_table_structure: bool = True,
        enable_cell_matching: bool = True,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        custom_pipeline_options: Optional[PdfPipelineOptions] = None,
    ):
        """
        Initialize the PDF processor with custom options.
        
        Args:
            enable_ocr: Whether to enable OCR processing
            enable_table_structure: Whether to process table structures
            enable_cell_matching: Whether to enable table cell matching
            chunk_size: Target size for text chunks
            chunk_overlap: Number of characters to overlap between chunks
            custom_pipeline_options: Optional custom pipeline options
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.pipeline_options = custom_pipeline_options or self._create_pipeline_options(
            enable_ocr,
            enable_table_structure,
            enable_cell_matching
        )
        
        self.converter = DocumentConverter(
            allowed_formats=[InputFormat.PDF],
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=self.pipeline_options,
                ),
            },
        )

    def _create_pipeline_options(self, enable_ocr: bool, enable_table_structure: bool, 
                               enable_cell_matching: bool) -> PdfPipelineOptions:
        """Create pipeline options with specified settings."""
        options = PdfPipelineOptions()
        options.do_ocr = enable_ocr
        options.do_table_structure = enable_table_structure
        options.table_structure_options.do_cell_matching = enable_cell_matching
        return options

    def _create_chunks(self, text: str, metadata: Dict) -> List[ChunkInfo]:
        """
        Create chunks from text while preserving semantic boundaries and tracking source.
        """
        chunks = []
        current_pos = 0
        
        while current_pos < len(text):
            # Find a good breaking point near chunk_size
            end_pos = min(current_pos + self.chunk_size, len(text))
            
            # Try to break at paragraph or sentence boundary
            if end_pos < len(text):
                # Look for paragraph break
                next_para = text.find('\n\n', end_pos - self.chunk_size//2, end_pos + self.chunk_size//2)
                if next_para != -1:
                    end_pos = next_para
                else:
                    # Look for sentence boundary
                    next_sentence = text.find('. ', end_pos - self.chunk_size//2, end_pos + self.chunk_size//2)
                    if next_sentence != -1:
                        end_pos = next_sentence + 1

            chunk_text = text[current_pos:end_pos].strip()
            
            # Create chunk with metadata
            chunk = ChunkInfo(
                chunk_id=str(uuid.uuid4()),
                content=chunk_text,
                start_index=current_pos,
                end_index=end_pos,
                page_number=self._estimate_page_number(current_pos, metadata),
                section_title=self._find_section_title(current_pos, text, metadata)
            )
            chunks.append(chunk)
            
            # Move position for next chunk, accounting for overlap
            current_pos = max(current_pos + 1, end_pos - self.chunk_overlap)
            
        return chunks

    def _estimate_page_number(self, position: int, metadata: Dict) -> Optional[int]:
        """Estimate page number based on position in text (implement based on your PDF library)"""
        # This would need to be implemented based on your specific PDF processing library
        return None

    def _find_section_title(self, position: int, text: str, metadata: Dict) -> Optional[str]:
        """Find the section title for the given position (implement based on your document structure)"""
        # This would need to be implemented based on your document structure
        return None

    def process_single_document(self, pdf_path: Path) -> ProcessingResult:
        """
        Process a single PDF document and return its content with chunk information.
        """
        try:
            conv_results = self.converter.convert_all(
                [pdf_path],
                raises_on_error=False,
            )
            
            conv_result = next(iter(conv_results))
            document_id = str(uuid.uuid4())
            
            if conv_result.status == ConversionStatus.SUCCESS:
                markdown_content = conv_result.document.export_to_markdown()
                metadata = {
                    "title": pdf_path.stem,
                    "original_path": str(pdf_path),
                    "conversion_status": "success",
                    "document_id": document_id
                }
                
                # Create chunks with source tracking
                chunks = self._create_chunks(markdown_content, metadata)
                
                return ProcessingResult(
                    markdown_content=markdown_content,
                    source_path=pdf_path,
                    metadata=metadata,
                    success=True,
                    document_id=document_id,
                    chunks=chunks
                )
            else:
                error_messages = [
                    err.error_message for err in conv_result.errors
                ] if conv_result.errors else ["Unknown conversion error"]
                
                return ProcessingResult(
                    markdown_content="",
                    source_path=pdf_path,
                    metadata={
                        "title": pdf_path.stem,
                        "original_path": str(pdf_path),
                        "conversion_status": "failure",
                        "document_id": document_id,
                        "errors": error_messages
                    },
                    success=False,
                    document_id=document_id,
                    error_messages=error_messages
                )
                
        except Exception as e:
            document_id = str(uuid.uuid4())
            logger.error(f"Error processing document {pdf_path}: {str(e)}")
            return ProcessingResult(
                markdown_content="",
                source_path=pdf_path,
                metadata={
                    "title": pdf_path.stem,
                    "original_path": str(pdf_path),
                    "conversion_status": "error",
                    "document_id": document_id,
                    "error": str(e)
                },
                success=False,
                document_id=document_id,
                error_messages=[str(e)]
            )

    def get_chunk_source_info(self, chunk_id: str, results: List[ProcessingResult]) -> Dict:
        """
        Retrieve source information for a specific chunk.
        """
        for result in results:
            for chunk in result.chunks:
                if chunk.chunk_id == chunk_id:
                    return {
                        "document_title": result.metadata["title"],
                        "document_path": str(result.source_path),
                        "page_number": chunk.page_number,
                        "section_title": chunk.section_title,
                        "chunk_content": chunk.content,
                        "document_id": result.document_id
                    }
        return None

    def format_source_reference(self, chunk_info: Dict) -> str:
        """
        Format source information for user display.
        """
        ref = f"Source: {chunk_info['document_title']}"
        if chunk_info['page_number']:
            ref += f", Page {chunk_info['page_number']}"
        if chunk_info['section_title']:
            ref += f", Section: {chunk_info['section_title']}"
        return ref