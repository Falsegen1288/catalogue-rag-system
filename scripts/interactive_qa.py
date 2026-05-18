#!/usr/bin/env python3
"""Interactive Q&A CLI for the catalogue system"""

import sys
from pathlib import Path
import os
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import VECTORDB_DIR, EMBEDDING_MODEL, COLLECTION_NAME, RETRIEVAL_TOP_K
from src.embedding.vector_store import VectorStore
from src.llm.groq_client import GroqClient

# Load environment variables
load_dotenv()


def main():
    # Initialize
    print("🗂️  Initializing Catalogue RAG System...")
    
    vector_store = VectorStore(
        persist_path=str(VECTORDB_DIR),
        collection_name=COLLECTION_NAME,
        embedding_model=EMBEDDING_MODEL
    )
    
    llm_client = GroqClient()
    
    print(f"✅ Ready! Collection contains {vector_store.collection.count()} chunks\n")
    print("Type 'exit' to quit, 'stats' for collection info\n")
    
    while True:
        question = input("\n❓ Ask about the catalogues: ").strip()
        
        if question.lower() == "exit":
            print("👋 Goodbye!")
            break
        
        if question.lower() == "stats":
            stats = vector_store.get_collection_stats()
            print(f"\n📊 {stats}")
            continue
        
        if not question:
            continue
        
        # Retrieve
        chunks = vector_store.retrieve(question, top_k=RETRIEVAL_TOP_K)
        
        if not chunks:
            print("⚠️  No relevant chunks found. Try a different question.")
            continue
        
        # Answer
        answer = llm_client.ask_catalogue(question, chunks)
        
        # Display
        print(f"\n{'='*60}")
        print(f"💬 {answer}")
        print(f"\n📚 Sources:")
        for c in chunks:
            print(f"  [{c['type']}] {c['source']} — Page {c['page']} (score: {c['score']})")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()