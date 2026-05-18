"""Vector database operations with ChromaDB and SentenceTransformers"""

import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
from tqdm import tqdm


class VectorStore:
    def __init__(self, persist_path: str, collection_name: str, embedding_model: str):
        self.persist_path = persist_path
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        
        self.embedder = SentenceTransformer(embedding_model)
        self.client = chromadb.PersistentClient(path=persist_path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def store_chunks(self, chunks: List[Dict], batch_size: int = 500):
        """Store chunks with embeddings into ChromaDB"""
        texts = [c["content"] for c in chunks]
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        metadatas = [{
            "source": c.get("source", ""),
            "page": str(c.get("page", "")),
            "type": c.get("type", ""),
            "image_path": c.get("image_path", ""),
            "has_table": str("TABLE" in c.get("content", ""))
        } for c in chunks]
        
        print("🔢 Generating embeddings...")
        embeddings = self.embedder.encode(texts, show_progress_bar=True).tolist()
        
        for i in tqdm(range(0, len(chunks), batch_size), desc="Storing batches"):
            self.collection.add(
                ids=ids[i:i+batch_size],
                embeddings=embeddings[i:i+batch_size],
                documents=texts[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size]
            )
        
        print(f"✅ {len(chunks)} chunks stored in ChromaDB")
        return len(chunks)
    
    def retrieve(self, query: str, top_k: int = 6, filter_type: Optional[str] = None) -> List[Dict]:
        """Retrieve relevant chunks from vector DB"""
        query_vec = self.embedder.encode([query]).tolist()
        
        where_filter = {"type": filter_type} if filter_type else None
        
        results = self.collection.query(
            query_embeddings=query_vec,
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )
        
        retrieved = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            retrieved.append({
                "content": doc,
                "source": meta["source"],
                "page": meta["page"],
                "type": meta["type"],
                "image_path": meta.get("image_path", ""),
                "score": round(1 - dist, 3)
            })
        
        return retrieved
    
    def get_collection_stats(self) -> Dict:
        """Get basic stats about the collection"""
        count = self.collection.count()
        return {"collection": self.collection_name, "total_chunks": count}