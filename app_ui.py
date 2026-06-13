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
                
                for page_num, page in enumerate(doc):
                    # extract text
                    full_text += page.get_text()
                    
                    # extract images
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
                
                doc.close()
                os.unlink(tmp_path)
                
                # chunk
                chunks = extract_chunks(full_text)
                
                # embed all chunks
                embeddings = model.encode(chunks)
                embeddings = np.array(embeddings).astype('float32')
                
                # build faiss index
                index = faiss.IndexFlatL2(embeddings.shape[1])
                index.add(embeddings)
                
                st.session_state.index = index
                st.session_state.chunks = chunks
                st.session_state.messages = []
                st.success(f"Done! {len(chunks)} chunks indexed.")

# main chat
if "index" not in st.session_state:
    st.info("Upload a PDF from the sidebar to get started.")
else:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    question = st.chat_input("Ask a question about your PDF...")
    
    if question:
        with st.chat_message("user"):
            st.write(question)
        
        # embed question and search
        question_embedding = np.array(
            [model.encode(question)]
        ).astype('float32')
        
        _, indices = st.session_state.index.search(question_embedding, 3)
        relevant_chunks = [st.session_state.chunks[i] for i in indices[0]]
        context = "\n\n".join(relevant_chunks)
        
        st.session_state.messages.append({
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}"
        })
        
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