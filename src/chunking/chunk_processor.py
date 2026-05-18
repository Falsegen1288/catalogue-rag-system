"""Convert Unstructured elements to structured chunks"""

from typing import List, Dict, Any


def elements_to_structured_chunks(elements: List, source_name: str) -> List[Dict]:
    """
    Converts Unstructured elements into clean dicts
    ready for embedding + storage.
    """
    chunks = []
    
    for el in elements:
        el_type = el.category  # Title, NarrativeText, Table, Image, etc.
        text = el.text.strip() if el.text else ""
        metadata = el.metadata
        
        # Skip empty / junk
        if not text and el_type not in ["Image"]:
            continue
        if len(text) < 10 and el_type == "NarrativeText":
            continue
        
        chunk = {
            "type": el_type,
            "content": text,
            "source": source_name,
            "page": getattr(metadata, "page_number", None),
            "coordinates": str(getattr(metadata, "coordinates", "")),
        }
        
        # Tables: also store HTML for structured display
        if el_type == "Table":
            chunk["content_html"] = getattr(metadata, "text_as_html", text)
            chunk["content"] = f"[TABLE]\n{text}"
        
        # Images: store file path
        elif el_type == "Image":
            img_path = getattr(metadata, "image_path", "")
            chunk["image_path"] = img_path
            chunk["content"] = f"[IMAGE on page {chunk['page']}] {text}"
        
        # Titles: tag them clearly for better retrieval
        elif el_type == "Title":
            chunk["content"] = f"[SECTION: {text}]"
        
        chunks.append(chunk)
    
    print(f"📦 {len(chunks)} structured chunks from {source_name}")
    return chunks