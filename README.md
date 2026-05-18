\# 🗂️ Catalogue RAG System



A production-ready RAG (Retrieval-Augmented Generation) system for extracting and querying product catalogues. Supports text, tables, and image extraction with semantic search capabilities.



\## ✨ Features



\- \*\*Complete PDF extraction\*\* – Text, tables (with HTML), images, and layout metadata

\- \*\*Vector-based retrieval\*\* – ChromaDB with SentenceTransformers embeddings

\- \*\*LLM-powered Q\&A\*\* – Groq (Llama 3.3 70B) for accurate, cited answers

\- \*\*Multi-catalogue support\*\* – Process multiple PDFs together

\- \*\*Evaluation framework\*\* – ROUGE, BERTScore, and fact-based scoring



\## 🚀 Quick Start



\### 1. Clone \& Setup



```bash

git clone https://github.com/yourusername/catalogue-rag-system.git

cd catalogue-rag-system

python -m venv venv

source venv/bin/activate  # On Windows: venv\\Scripts\\activate

pip install -r requirements.txt

