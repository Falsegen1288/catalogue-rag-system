"""Configuration settings for the Catalogue RAG System"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
IMAGES_DIR = DATA_DIR / "extracted_images"
VECTORDB_DIR = DATA_DIR / "vectordb"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
for dir_path in [DATA_DIR, RAW_DATA_DIR, IMAGES_DIR, VECTORDB_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Extraction settings
EXTRACTION_STRATEGY = "hi_res"  # hi_res, fast, ocr_only, auto
EXTRACT_IMAGES = True
EXTRACT_TABLE_STRUCTURE = True
LANGUAGES = ["eng"]

# Chunking settings
CHUNK_STRATEGY = "by_title"
MAX_CHARS = 1000
OVERLAP = 100

# Embedding settings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "catalogues"
RETRIEVAL_TOP_K = 6
SIMILARITY_METRIC = "cosine"

# LLM settings
LLM_MODEL = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0.2
MAX_TOKENS = 1500

# Evaluation
EVAL_QUESTIONS_COUNT = 5