#!/usr/bin/env python3
"""Complete RAG pipeline: extract → chunk → store → query"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import (
    RAW_DATA_DIR, IMAGES_DIR, VECTORDB_DIR,
    EXTRACTION_STRATEGY, EMBEDDING_MODEL, COLLECTION_NAME
)
from src.extraction.pdf_extractor import extract_catalogue_layout
from src.chunking.chunk_processor import elements_to_structured_chunks
from src.embedding.vector_store import VectorStore


def run_pipeline():
    """Execute full extraction and storage pipeline"""
    
    # Find all PDFs in raw data directory
    pdf_paths = list(RAW_DATA_DIR.glob("*.pdf"))
    
    if not pdf_paths:
        print(f"❌ No PDFs found in {RAW_DATA_DIR}")
        print("   Please add PDF catalogues to data/raw/ folder")
        return
    
    print(f"📁 Found {len(pdf_paths)} PDFs to process\n")
    
    all_chunks = []
    
    for pdf_path in pdf_paths:
        # Extract
        elements = extract_catalogue_layout(
            pdf_path=str(pdf_path),
            output_image_dir=str(IMAGES_DIR / f"images_{pdf_path.stem}"),
            strategy=EXTRACTION_STRATEGY
        )
        
        # Chunk
        chunks = elements_to_structured_chunks(elements, source_name=pdf_path.name)
        all_chunks.extend(chunks)
    
    print(f"\n✅ Total chunks across all catalogues: {len(all_chunks)}")
    
    # Store in vector DB
    vector_store = VectorStore(
        persist_path=str(VECTORDB_DIR),
        collection_name=COLLECTION_NAME,
        embedding_model=EMBEDDING_MODEL
    )
    
    vector_store.store_chunks(all_chunks)
    
    # Print stats
    stats = vector_store.get_collection_stats()
    print(f"\n📊 Collection stats: {stats}")
    
    return vector_store, all_chunks


if __name__ == "__main__":
    run_pipeline()