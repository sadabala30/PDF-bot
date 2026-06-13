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

st.set_page_config(page_title="PDF Bot", page_icon="✦", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;700&family=Space+Mono:wght@400;700&display=swap');

* { font-family: 'Space Grotesk', sans-serif; box-sizing: border-box; }

/* ── KILL ALL STREAMLIT CHROME ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
[data-testid="stToolbar"] { display: none; }

/* ── ROOT BG ── */
.stApp {
    background: #080810;
    color: #e8e6f0;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: #09090f !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 0 !important;
}

/* sidebar canvas hero injected via component */
.sidebar-brand {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.18em;
    color: rgba(249,83,198,0.6);
    text-transform: uppercase;
    padding: 10px 16px 6px;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    margin-bottom: 12px;
}

/* file uploader */
[data-testid="stFileUploader"] {
    background: rgba(249,83,198,0.04) !important;
    border: 1.5px dashed rgba(249,83,198,0.35) !important;
    border-radius: 10px !important;
    padding: 8px !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(249,83,198,0.7) !important;
}
[data-testid="stFileUploader"] label { color: #c0bdd8 !important; font-size: 0.82rem !important; }

/* process button */
.stButton > button {
    background: transparent !important;
    border: 1px solid rgba(249,83,198,0.5) !important;
    color: #f953c6 !important;
    border-radius: 6px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.1em !important;
    width: 100% !important;
    padding: 8px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: rgba(249,83,198,0.12) !important;
    border-color: #f953c6 !important;
}

/* progress bar */
.stProgress > div > div { background: #f953c6 !important; }

/* success / info banners */
.stSuccess {
    background: rgba(0,255,150,0.06) !important;
    border: 1px solid rgba(0,255,150,0.2) !important;
    border-radius: 8px !important;
    color: #6effc8 !important;
}
.stInfo {
    background: rgba(249,83,198,0.06) !important;
    border: 1px solid rgba(249,83,198,0.2) !important;
    border-radius: 8px !important;
}

/* sidebar tips */
.tip-block {
    margin-top: 4px;
    padding: 10px 14px;
    border-left: 2px solid rgba(249,83,198,0.4);
    font-size: 0.78rem;
    color: #7a78a0;
    line-height: 1.7;
}

/* ── MAIN AREA ── */
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ── HERO CANVAS STRIP ── */
#hero-wrap {
    width: 100%;
    position: relative;
    overflow: hidden;
}

/* ── CHAT MESSAGES ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0 32px !important;
    margin-bottom: 6px !important;
}

/* user bubble */
[data-testid="stChatMessage"][data-testid*="user"] .stMarkdown,
.stChatMessage:has([data-testid="chatAvatarIcon-user"]) {
    background: rgba(249,83,198,0.1) !important;
}

/* override chat bubble backgrounds */
[data-testid="stChatMessage"] > div:last-child {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 12px !important;
    padding: 12px 16px !important;
    font-size: 0.88rem !important;
    line-height: 1.65 !important;
    color: #d0cee8 !important;
}

/* user msg distinct */
[data-testid="stChatMessage"]:has([aria-label="user avatar"]) > div:last-child,
[data-testid="stChatMessage"]:nth-child(odd) > div:last-child {
    background: rgba(249,83,198,0.08) !important;
    border-color: rgba(249,83,198,0.2) !important;
    color: #eee8f8 !important;
}

/* avatar icons */
[data-testid="stChatMessage"] img,
[data-testid="stChatMessage"] [data-testid="chatAvatarIcon-user"],
[data-testid="stChatMessage"] [data-testid="chatAvatarIcon-assistant"] {
    background: #1a1828 !important;
    border: 1px solid rgba(249,83,198,0.3) !important;
    border-radius: 50% !important;
    color: #f953c6 !important;
}

/* ── CHAT INPUT ── */
[data-testid="stChatInput"] {
    background: #0d0c18 !important;
    border-top: 1px solid rgba(255,255,255,0.06) !important;
    padding: 14px 32px !important;
}
[data-testid="stChatInput"] textarea {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(249,83,198,0.3) !important;
    border-radius: 10px !important;
    color: #e8e6f0 !important;
    font-size: 0.88rem !important;
    font-family: 'Space Grotesk', sans-serif !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: rgba(249,83,198,0.7) !important;
    box-shadow: 0 0 0 2px rgba(249,83,198,0.12) !important;
}
[data-testid="stChatInput"] button {
    background: rgba(249,83,198,0.15) !important;
    border: 1px solid rgba(249,83,198,0.4) !important;
    border-radius: 8px !important;
    color: #f953c6 !important;
}

/* ── EMPTY STATE ── */
.empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 60px 24px;
    text-align: center;
    color: #3d3a58;
}
.empty-state .glyph {
    font-size: 3.5rem;
    margin-bottom: 20px;
    opacity: 0.4;
}
.empty-state h2 {
    font-size: 1.3rem;
    font-weight: 500;
    color: #5a5878;
    margin-bottom: 8px;
}
.empty-state p {
    font-size: 0.8rem;
    color: #38364f;
    font-family: 'Space Mono', monospace;
    letter-spacing: 0.05em;
}

/* scrollbar */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(249,83,198,0.25); border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: rgba(249,83,198,0.5); }

/* spinner */
.stSpinner > div { border-top-color: #f953c6 !important; }

</style>
""", unsafe_allow_html=True)

# ── HERO CANVAS (injected at top of main area) ──────────────────────────────
HERO_HTML = """
<div id="hero-wrap" style="width:100%;position:relative;overflow:hidden;height:220px;background:#080810;">
  <canvas id="hc" style="position:absolute;inset:0;width:100%;height:100%;"></canvas>
  <canvas id="gc" style="position:absolute;inset:0;width:100%;height:100%;opacity:0;pointer-events:none;"></canvas>

  <!-- scanlines -->
  <div style="position:absolute;inset:0;pointer-events:none;z-index:4;
    background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,0,0,0.07) 2px,rgba(0,0,0,0.07) 4px);">
  </div>

  <!-- corner tags -->
  <div style="position:absolute;top:14px;left:18px;z-index:10;
    font-family:'Space Mono',monospace;font-size:10px;letter-spacing:.15em;
    color:rgba(249,83,198,.65);text-transform:uppercase;
    border:1px solid rgba(249,83,198,.25);padding:3px 8px;border-radius:2px;">
    ✦ PDF BOT
  </div>
  <div style="position:absolute;top:14px;right:18px;z-index:10;
    font-family:'Space Mono',monospace;font-size:10px;letter-spacing:.15em;
    color:rgba(255,255,255,.2);">
    RAG / v2.0
  </div>

  <!-- title -->
  <div style="position:absolute;bottom:22px;left:22px;z-index:10;pointer-events:none;">
    <div style="font-family:'Space Grotesk',sans-serif;font-size:2.4rem;font-weight:700;
      letter-spacing:-.03em;color:#fff;line-height:1;mix-blend-mode:difference;">
      Ask your&nbsp;<span style="color:#f953c6;">PDF</span>
    </div>
    <div style="font-family:'Space Mono',monospace;font-size:.68rem;color:rgba(255,255,255,.35);
      margin-top:6px;letter-spacing:.1em;text-transform:uppercase;">
      hover to glitch · upload · ask anything
    </div>
  </div>
</div>

<script>
(function(){
  const wrap = document.getElementById('hero-wrap');
  const hc   = document.getElementById('hc');
  const gc   = document.getElementById('gc');
  const ctx  = hc.getContext('2d');
  const gctx = gc.getContext('2d');
  let W,H,sy=0,glitching=false,gtimer=null;

  function resize(){
    W=wrap.offsetWidth; H=wrap.offsetHeight;
    hc.width=W; hc.height=H; gc.width=W; gc.height=H;
  }

  /* procedural noise texture */
  function makeTex(){
    const s=256,o=document.createElement('canvas');
    o.width=s;o.height=s;
    const ox=o.getContext('2d'),id=ox.createImageData(s,s),d=id.data;
    for(let i=0;i<s*s;i++){
      const v=Math.random();
      const r=v>.97?249:v>.93?120:Math.floor(v*22+3);
      const g=v>.97?83 :v>.93?20 :Math.floor(v*10+1);
      const b=v>.97?198:v>.93?80 :Math.floor(v*40+8);
      d[i*4]=r;d[i*4+1]=g;d[i*4+2]=b;d[i*4+3]=255;
    }
    ox.putImageData(id,0,0);return o;
  }
  const tex=makeTex();
  let pat;
  function buildPat(){ pat=ctx.createPattern(tex,'repeat'); }

  function draw(){
    ctx.clearRect(0,0,W,H);
    ctx.fillStyle='#080810';ctx.fillRect(0,0,W,H);
    ctx.save();
    ctx.translate(0,-(sy%tex.height));
    ctx.fillStyle=pat;ctx.globalAlpha=.5;
    ctx.fillRect(0,0,W,H+tex.height);
    ctx.restore();
    /* vignette */
    const vg=ctx.createRadialGradient(W/2,H/2,H*.05,W/2,H/2,H*.9);
    vg.addColorStop(0,'rgba(0,0,0,0)');vg.addColorStop(1,'rgba(0,0,0,.82)');
    ctx.fillStyle=vg;ctx.globalAlpha=1;ctx.fillRect(0,0,W,H);
    /* pink streaks */
    ctx.globalAlpha=.1;
    for(let i=0;i<4;i++){
      const y=(((sy*.35+i*83))%H+H)%H;
      ctx.fillStyle='#f953c6';ctx.fillRect(0,y,W,1);
    }
    ctx.globalAlpha=1;
  }

  function glitch(){
    if(!glitching)return;
    gctx.clearRect(0,0,W,H);
    gc.style.opacity='1';
    /* slice shift */
    const n=Math.floor(Math.random()*10)+4;
    for(let i=0;i<n;i++){
      const gy=Math.floor(Math.random()*H);
      const gh=Math.floor(Math.random()*14)+2;
      const dx=(Math.random()-.5)*50;
      gctx.drawImage(hc,0,gy,W,gh,dx,gy,W,gh);
    }
    /* channel blob */
    gctx.globalCompositeOperation='screen';
    gctx.globalAlpha=.18;
    const colors=['#f953c6','#00ffb3','#4444ff'];
    gctx.fillStyle=colors[Math.floor(Math.random()*colors.length)];
    gctx.fillRect(Math.random()*W*.7,Math.random()*H,Math.random()*120+20,Math.random()*24+4);
    gctx.globalCompositeOperation='source-over';
    gctx.globalAlpha=1;
    gtimer=setTimeout(()=>{
      gc.style.opacity='0';
      gctx.clearRect(0,0,W,H);
      if(glitching)setTimeout(glitch,Math.random()*100+30);
    },55);
  }

  wrap.addEventListener('mouseenter',()=>{glitching=true;glitch();});
  wrap.addEventListener('mouseleave',()=>{
    glitching=false;clearTimeout(gtimer);
    gc.style.opacity='0';gctx.clearRect(0,0,W,H);
  });
  wrap.addEventListener('mousemove',(e)=>{
    gctx.globalAlpha=.06;gctx.fillStyle='#b91d73';
    gctx.fillRect(0,Math.floor(e.offsetY-1),W,2);
    gctx.globalAlpha=1;
  });

  function loop(){ sy+=.55; draw(); requestAnimationFrame(loop); }
  resize(); buildPat(); loop();
  window.addEventListener('resize',()=>{resize();buildPat();});
})();
</script>
"""

# ── MODELS ──────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return model, client

model, anthropic_client = load_models()

def extract_chunks(text, chunk_size=500, overlap=100):
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i:i+chunk_size]))
        i += chunk_size - overlap
    return chunks

# ── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-brand">✦ &nbsp;PDF Bot &nbsp;·&nbsp; RAG engine</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Drop a PDF", type="pdf", label_visibility="collapsed")

    if uploaded_file:
        st.markdown(f"<p style='font-family:Space Mono,monospace;font-size:0.72rem;color:#7a78a0;padding:4px 0 8px;'>📄 {uploaded_file.name}</p>", unsafe_allow_html=True)
        if st.button("⚡ Process PDF"):
            with st.spinner("Indexing…"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                doc = fitz.open(tmp_path)
                full_text = ""
                progress = st.progress(0)
                total = len(doc)

                for pnum, page in enumerate(doc):
                    full_text += page.get_text()
                    for img in page.get_images():
                        try:
                            xref = img[0]
                            bi = doc.extract_image(xref)
                            b64 = base64.b64encode(bi["image"]).decode()
                            mt = "image/png" if bi["ext"] == "png" else "image/jpeg"
                            resp = anthropic_client.messages.create(
                                model="claude-haiku-4-5-20251001",
                                max_tokens=300,
                                messages=[{"role":"user","content":[
                                    {"type":"image","source":{"type":"base64","media_type":mt,"data":b64}},
                                    {"type":"text","text":"Describe this image. If it's a diagram or chart, explain what it shows."}
                                ]}]
                            )
                            full_text += f"\n[Image p{pnum+1}]: {resp.content[0].text}\n"
                        except Exception:
                            continue
                    progress.progress((pnum+1)/total)

                doc.close()
                os.unlink(tmp_path)

                chunks = extract_chunks(full_text)
                embs = np.array(model.encode(chunks)).astype('float32')
                index = faiss.IndexFlatL2(embs.shape[1])
                index.add(embs)

                st.session_state.index  = index
                st.session_state.chunks = chunks
                st.session_state.messages = []

            st.success(f"✅ {len(chunks)} chunks indexed")

    st.markdown("""
    <div class="tip-block">
        Ask specific questions<br>
        Reference page numbers<br>
        Ask for summaries<br>
        Ask about diagrams
    </div>
    """, unsafe_allow_html=True)

    # mini noise art in sidebar footer
    st.markdown("""
    <div style="position:absolute;bottom:16px;left:16px;right:16px;
      font-family:'Space Mono',monospace;font-size:9px;
      color:rgba(249,83,198,.2);letter-spacing:.08em;">
      ✦ powered by claude · faiss · sentence-transformers
    </div>
    """, unsafe_allow_html=True)

# ── MAIN ─────────────────────────────────────────────────────────────────────
# inject hero canvas at very top
st.components.v1.html(HERO_HTML, height=220, scrolling=False)

if "index" not in st.session_state:
    st.markdown("""
    <div class="empty-state">
      <div class="glyph">⬡</div>
      <h2>Upload a PDF to begin</h2>
      <p>your document stays local · embeddings live in memory</p>
    </div>
    """, unsafe_allow_html=True)
else:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # chat history
    for msg in st.session_state.messages:
        # only show assistant messages (user ones have context prepended)
        if msg["role"] == "assistant":
            with st.chat_message("assistant"):
                st.write(msg["content"])
        elif msg["role"] == "user":
            # strip context prefix for display
            display = msg["content"]
            if "Question: " in display:
                display = display.split("Question: ")[-1]
            with st.chat_message("user"):
                st.write(display)

    question = st.chat_input("✦  Ask anything about your PDF…")

    if question:
        with st.chat_message("user"):
            st.write(question)

        q_emb = np.array([model.encode(question)]).astype('float32')
        _, idxs = st.session_state.index.search(q_emb, 3)
        context = "\n\n".join(st.session_state.chunks[i] for i in idxs[0])

        st.session_state.messages.append({
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}"
        })

        with st.spinner("✦  thinking…"):
            response = anthropic_client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1024,
                system="You are a helpful assistant and expert at reading documents. Answer questions based on the provided context. If the exact term isn't found, look for related concepts. Only say 'I don't find that in the document' if there's absolutely nothing related.",
                messages=st.session_state.messages
            )

        reply = response.content[0].text
        st.session_state.messages.append({"role": "assistant", "content": reply})

        with st.chat_message("assistant"):
            st.write(reply)