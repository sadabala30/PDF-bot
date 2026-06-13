from sentence_transformers import SentenceTransformer
import chromadb
from chunker import extract_chunks

# load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# load chromadb
client = chromadb.Client()
collection = client.create_collection("pdf_chunks")

# get all 523 chunks from pdf
print("Reading and chunking PDF...")
chunks = extract_chunks("ml.pdf")
print(f"Total chunks: {len(chunks)}")

# embed all chunks and store in chromadb
print("Embedding all chunks... (this takes a few minutes)")
for i, chunk in enumerate(chunks):
    embedding = model.encode(chunk).tolist()
    collection.add(
        documents=[chunk],
        embeddings=[embedding],
        ids=[f"chunk_{i}"]
    )
    if i % 50 == 0:
        print(f"Progress: {i}/{len(chunks)} chunks done")

print("All chunks stored in ChromaDB!")

# test search
query = "What is gradient descent?"
query_embedding = model.encode(query).tolist()

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)

print(f"\nTop 3 chunks for: '{query}'")
for i, doc in enumerate(results['documents'][0]):
    print(f"\n--- Chunk {i+1} ---")
    print(doc[:300])