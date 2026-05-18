"""PDF layout extraction using Unstructured library"""

import os
from pathlib import Path
from typing import List
from unstructured.partition.pdf import partition_pdf


def extract_catalogue_layout(
    pdf_path: str,
    output_image_dir: str,
    strategy: str = "hi_res",
    extract_images: bool = True,
    extract_table_structure: bool = True,
    languages: List[str] = ["eng"],
    chunk_strategy: str = "by_title",
    max_chars: int = 1000,
    overlap: int = 100
) -> List:
    """
    Extracts EVERYTHING from a complex catalogue:
    - Text blocks with their role (title, body, list, caption...)
    - Tables with full HTML structure
    - Images saved to disk
    - Reading order preserved
    - Coordinates / bounding boxes retained
    """
    os.makedirs(output_image_dir, exist_ok=True)
    
    print(f"🔍 Extracting layout from: {pdf_path}")
    
    elements = partition_pdf(
        filename=pdf_path,
        
        # Layout & reading order
        strategy=strategy,
        
        # Image handling
        extract_images_in_pdf=extract_images,
        extract_image_block_output_dir=output_image_dir,
        extract_image_block_types=["Image", "Table"] if extract_images else [],
        
        # Table handling
        infer_table_structure=extract_table_structure,
        
        # OCR (for scanned or image-heavy catalogues)
        languages=languages,
        
        # Chunking hints
        chunking_strategy=chunk_strategy,
        max_characters=max_chars,
        overlap=overlap,
        overlap_all=True,
        
        # Coordinate metadata
        include_page_breaks=True,
    )
    
    print(f"✅ Extracted {len(elements)} elements")
    return elements