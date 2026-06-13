import streamlit as st
import anthropic
from dotenv import load_dotenv
import os
load_dotenv()
from sentence_transformers import SentenceTransformer
import chromadb
import fitz
import tempfile
import os

# page config
st.set_page_config(page_title="PDF Bot", page_icon="📄")
st.title("📄 PDF Q&A Bot")
st.caption("Upload any PDF and ask questions about it")

# load models once
@st.cache_resource
def load_models():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return model, anthropic_client

model, anthropic_client = load_models()

# chunk text
def extract_chunks(text, chunk_size=500, overlap=100):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

# sidebar — pdf upload
with st.sidebar:
    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")
    
    if uploaded_file:
        if st.button("Process PDF"):
            with st.spinner("Reading and indexing PDF..."):
                # read pdf
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
                
                doc = fitz.open(tmp_path)
                full_text = ""
                for page in doc:
                    full_text += page.get_text()
                doc.close()
                os.unlink(tmp_path)
                
                # chunk and embed
                chunks = extract_chunks(full_text)
                
                chroma_client = chromadb.Client()
                
                # reset collection
                try:
                    chroma_client.delete_collection("pdf_chunks")
                except:
                    pass
                
                collection = chroma_client.create_collection("pdf_chunks")
                
                for i, chunk in enumerate(chunks):
                    embedding = model.encode(chunk).tolist()
                    collection.add(
                        documents=[chunk],
                        embeddings=[embedding],
                        ids=[f"chunk_{i}"]
                    )
                
                st.session_state.collection = collection
                st.session_state.messages = []
                st.success(f"Done! {len(chunks)} chunks indexed.")

# main chat
if "collection" not in st.session_state:
    st.info("Upload a PDF from the sidebar to get started.")
else:
    # display chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    question = st.chat_input("Ask a question about your PDF...")
    
    if question:
        with st.chat_message("user"):
            st.write(question)
        
        # retrieve chunks
        question_embedding = model.encode(question).tolist()
        results = st.session_state.collection.query(
            query_embeddings=[question_embedding],
            n_results=3
        )
        context = "\n\n".join(results['documents'][0])
        
        st.session_state.messages.append({
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}"
        })
        
        # get response
        with st.spinner("Thinking..."):
            response = anthropic_client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1024,
                system="You are a helpful assistant and an expert in machine learning. Answer questions based on the provided context. If the exact term isn't found, look for related concepts in the context and explain those. Only say 'I don't find that in the document' if there is absolutely nothing related in the context.",
                messages=st.session_state.messages
            )
        
        reply = response.content[0].text
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": reply
        })
        
        with st.chat_message("assistant"):
            st.write(reply)