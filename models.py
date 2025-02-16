from dataclasses import dataclass
from typing import List


@dataclass
class ChunkMetadata:
    previous_chunks: List[str]  # Content of previous chunks
    next_chunks: List[str]     # Content of next chunks
    original_metadata: dict    # Original document metadata