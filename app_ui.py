import streamlit as st
import anthropic
from sentence_transformers import SentenceTransformer
import fitz
import tempfile
import os
import base64
import numpy as np
import faiss
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="PDF Bot", page_icon="📄", layout="wide")

# custom CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}

.main-title {
    text-align: center;
    font-size: 3rem;
    font-weight: 700;
    background: linear-gradient(90deg, #f953c6, #b91d73, #f953c6);
    background-size: 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 3s infinite linear;
    margin-bottom: 0;
}

.subtitle {
    text-align: center;
    color: #a0a0c0;
    font-size: 1.1rem;
    margin-bottom: 2rem;
}

@keyframes shimmer {
    0% { background-position: 0% }
    100% { background-position: 200% }
}

.stChatMessage {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 16px !important;
    backdrop-filter: blur(10px) !important;
    margin-bottom: 1rem !important;
}

.stButton > button {
    background: linear-gradient(90deg, #f953c6, #b91d73) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.5rem 2rem !important;
    font-weight: 600 !important;
    transition: transform 0.2s !important;
    width: 100% !important;
}

.stButton > button:hover {
    transform: scale(1.05) !important;
}

.stFileUploader {
    background: rgba(255,255,255,0.05) !important;
    border: 2px dashed rgba(249,83,198,0.5) !important;
    border-radius: 16px !important;
    padding: 1rem !important;
}

.stChatInput > div {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(249,83,198,0.5) !important;
    border-radius: 16px !important;
}

.stSidebar {
    background: rgba(255,255,255,0.03) !important;
    border-right: 1px solid rgba(255,255,255,0.1) !important;
}

.stSuccess {
    background: rgba(0,255,150,0.1) !important;
    border: 1px solid rgba(0,255,150,0.3) !important;
    border-radius: 12px !important;
}

.stInfo {
    background: rgba(249,83,198,0.1) !important;
    border: 1px solid rgba(249,83,198,0.3) !important;
    border-radius: 12px !important;
}

.pulse {
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}
</style>
""", unsafe_allow_html=True)

# animated title
st.markdown('<p class="main-title">✨ PDF Q&A Bot</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload any PDF — ask anything — get instant answers</p>', unsafe_allow_html=True)

@st.cache_resource
def load_models():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return model, anthropic_client

model, anthropic_client = load_models()

def extract_chunks(text, chunk_size=500, overlap=100):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

with st.sidebar:
    st.markdown("## 📂 Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf", label_visibility="collapsed")
    
    if uploaded_file:
        st.markdown(f"**📄 {uploaded_file.name}**")
        if st.button("⚡ Process PDF"):
            with st.spinner("🔍 Reading and indexing..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
                
                doc = fitz.open(tmp_path)
                full_text = ""
                progress = st.progress(0)
                total_pages = len(doc)
                
                for page_num, page in enumerate(doc):
                    full_text += page.get_text()
                    
                    image_list = page.get_images()
                    for img in image_list:
                        try:
                            xref = img[0]
                            base_image = doc.extract_image(xref)
                            image_bytes = base_image["image"]
                            image_b64 = base64.b64encode(image_bytes).decode()
                            media_type = "image/png" if base_image["ext"] == "png" else "image/jpeg"
                            
                            vision_response = anthropic_client.messages.create(
                                model="claude-haiku-4-5-20251001",
                                max_tokens=300,
                                messages=[{
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "image",
                                            "source": {
                                                "type": "base64",
                                                "media_type": media_type,
                                                "data": image_b64
                                            }
                                        },
                                        {
                                            "type": "text",
                                            "text": "Describe this image in detail. If it's a diagram, chart, or graph, explain what it shows."
                                        }
                                    ]
                                }]
                            )
                            
                            image_description = f"[Image on page {page_num + 1}]: {vision_response.content[0].text}"
                            full_text += "\n" + image_description + "\n"
                            
                        except Exception:
                            continue
                    
                    progress.progress((page_num + 1) / total_pages)
                
                doc.close()
                os.unlink(tmp_path)
                
                chunks = extract_chunks(full_text)
                embeddings = model.encode(chunks)
                embeddings = np.array(embeddings).astype('float32')
                
                index = faiss.IndexFlatL2(embeddings.shape[1])
                index.add(embeddings)
                
                st.session_state.index = index
                st.session_state.chunks = chunks
                st.session_state.messages = []
                st.success(f"✅ Done! {len(chunks)} chunks indexed.")
    
    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.markdown("- Ask specific questions")
    st.markdown("- Reference page numbers")
    st.markdown("- Ask for summaries")
    st.markdown("- Ask about diagrams")

if "index" not in st.session_state:
    st.markdown("""
    <div style='text-align:center; padding: 4rem 2rem;'>
        <div style='font-size:5rem;'>📄</div>
        <h2 style='color:white;'>Upload a PDF to get started</h2>
        <p style='color:#a0a0c0;'>Your document stays private and secure</p>
    </div>
    """, unsafe_allow_html=True)
else:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    question = st.chat_input("✨ Ask anything about your PDF...")
    
    if question:
        with st.chat_message("user"):
            st.write(question)
        
        question_embedding = np.array([model.encode(question)]).astype('float32')
        _, indices = st.session_state.index.search(question_embedding, 3)
        relevant_chunks = [st.session_state.chunks[i] for i in indices[0]]
        context = "\n\n".join(relevant_chunks)
        
        st.session_state.messages.append({
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}"
        })
        
        with st.spinner("✨ Thinking..."):
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