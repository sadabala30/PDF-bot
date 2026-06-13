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
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf",