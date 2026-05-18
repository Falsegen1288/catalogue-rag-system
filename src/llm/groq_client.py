"""Groq LLM client wrapper for catalogue Q&A"""

from groq import Groq
from typing import List, Dict
import os


class GroqClient:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found! Set it via environment variable.")
        self.client = Groq(api_key=self.api_key)
    
    def ask_catalogue(self, question: str, chunks: List[Dict], 
                      model: str = "llama-3.3-70b-versatile",
                      temperature: float = 0.2,
                      max_tokens: int = 1500) -> str:
        """Generate answer from retrieved chunks"""
        
        # Build type-tagged context
        context = ""
        for c in chunks:
            tag = c["type"].upper()
            context += f"\n[{tag} | {c['source']} | Page {c['page']}]\n{c['content']}\n"
        
        # System prompt
        system_prompt = """You are an expert assistant for product catalogue queries.
You have access to extracted content from complex product catalogues including
text descriptions, tables (with specs/pricing), section titles, and image captions.

Rules:
- Answer ONLY from the provided catalogue context
- For specs/pricing questions, prioritize TABLE content
- For product descriptions, use NarrativeText
- Always cite: catalogue name + page number
- If info is missing, say so clearly — do NOT hallucinate
- Format answers cleanly with bullet points where helpful"""
        
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Catalogue Context:\n{context}\n\nQuestion: {question}"}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        return response.choices[0].message.content