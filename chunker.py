import fitz

def extract_chunks(pdf_path, chunk_size=500, overlap=100):
    doc = fitz.open(pdf_path)
    full_text = ""
    
    for page in doc:
        full_text += page.get_text()
    
    words = full_text.split()
    chunks = []
    
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap  # step forward by 400, not 500
    
    return chunks