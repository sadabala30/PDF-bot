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

st.set_page_config(page_title="PDF BOT", page_icon="⬡", layout="wide")

# ── GLOBAL STYLES ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&family=Inter:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

#MainMenu, footer, header, [data-testid="stToolbar"], .stDeployButton { display:none !important; visibility:hidden !important; }

html, body, .stApp, [data-testid="stAppViewContainer"] {
    background: #020208 !important;
    color: #c8e0f0 !important;
    font-family: 'Inter', sans-serif !important;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: rgba(0,4,18,0.95) !important;
    border-right: 1px solid rgba(0,200,255,0.12) !important;
    backdrop-filter: blur(12px) !important;
}
[data-testid="stSidebar"] > div { padding-top: 0 !important; }

/* file uploader */
[data-testid="stFileUploader"] {
    background: rgba(0,200,255,0.03) !important;
    border: 1px solid rgba(0,200,255,0.2) !important;
    border-radius: 6px !important;
}
[data-testid="stFileUploader"]:hover { border-color: rgba(0,200,255,0.6) !important; }
[data-testid="stFileUploader"] label { color: rgba(0,200,255,0.6) !important; font-family:'Share Tech Mono',monospace !important; font-size:0.72rem !important; }
[data-testid="stFileUploader"] section { border: none !important; background: transparent !important; }
[data-testid="stFileUploader"] button { border-color: rgba(0,200,255,0.4) !important; color: #00dcff !important; background: transparent !important; }

/* process button */
.stButton > button {
    background: transparent !important;
    border: 1px solid rgba(0,200,255,0.4) !important;
    color: #00dcff !important;
    border-radius: 4px !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.12em !important;
    width: 100% !important;
    padding: 10px !important;
    text-transform: uppercase !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: rgba(0,200,255,0.1) !important;
    border-color: #00dcff !important;
    box-shadow: 0 0 20px rgba(0,200,255,0.25) !important;
}

/* progress */
.stProgress > div > div { background: #00dcff !important; }
.stProgress > div { background: rgba(0,200,255,0.1) !important; border-radius: 0 !important; }

/* alerts */
.stSuccess { background: rgba(0,255,136,0.06) !important; border: 1px solid rgba(0,255,136,0.2) !important; border-radius: 4px !important; color: #00ff88 !important; font-family:'Share Tech Mono',monospace !important; font-size:0.75rem !important; }
.stInfo    { background: rgba(0,200,255,0.06) !important; border: 1px solid rgba(0,200,255,0.2) !important; border-radius: 4px !important; }

/* spinner */
.stSpinner > div { border-top-color: #00dcff !important; }

/* ── MAIN ── */
.block-container { padding: 0 !important; max-width: 100% !important; }

/* ── CHAT MESSAGES ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 4px 28px !important;
    margin-bottom: 4px !important;
}
[data-testid="stChatMessage"] > div:last-child {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 6px 6px 6px 2px !important;
    padding: 12px 16px !important;
    font-size: 0.85rem !important;
    line-height: 1.65 !important;
    color: #b8cce0 !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stChatMessage"]:has([aria-label="user avatar"]) > div:last-child {
    background: rgba(0,200,255,0.08) !important;
    border-color: rgba(0,200,255,0.2) !important;
    border-radius: 6px 6px 2px 6px !important;
    color: #a0d8f0 !important;
}
[data-testid="stChatMessage"] [data-testid*="Avatar"],
[data-testid="stChatMessage"] img {
    background: rgba(0,200,255,0.1) !important;
    border: 1px solid rgba(0,200,255,0.3) !important;
    border-radius: 50% !important;
}

/* ── CHAT INPUT ── */
[data-testid="stChatInput"] {
    background: rgba(0,2,15,0.9) !important;
    border-top: 1px solid rgba(0,200,255,0.12) !important;
    padding: 14px 28px !important;
}
[data-testid="stChatInput"] textarea {
    background: rgba(0,200,255,0.04) !important;
    border: 1px solid rgba(0,200,255,0.2) !important;
    border-radius: 4px !important;
    color: #a0d8f0 !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.82rem !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: rgba(0,200,255,0.7) !important;
    box-shadow: 0 0 16px rgba(0,200,255,0.15) !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: rgba(0,200,255,0.25) !important; }
[data-testid="stChatInput"] button {
    background: rgba(0,200,255,0.12) !important;
    border: 1px solid rgba(0,200,255,0.35) !important;
    color: #00dcff !important;
    border-radius: 4px !important;
}

/* scrollbar */
::-webkit-scrollbar { width: 3px; height: 3px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(0,200,255,0.3); border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: rgba(0,200,255,0.6); }

/* metric sidebar cards */
.metric-grid { display:grid; grid-template-columns:1fr 1fr; gap:8px; }
.metric-card {
    background: rgba(0,200,255,0.04);
    border: 1px solid rgba(0,200,255,0.12);
    border-radius:4px; padding:10px 8px; text-align:center;
}
.metric-val { font-family:'Orbitron',monospace; font-size:0.9rem; font-weight:700; color:#00dcff; }
.metric-lbl { font-family:'Share Tech Mono',monospace; font-size:0.6rem; color:rgba(0,200,255,0.4); margin-top:2px; letter-spacing:.06em; }

.sys-tips { font-family:'Share Tech Mono',monospace; font-size:0.65rem; color:rgba(0,200,255,0.3); line-height:2; letter-spacing:.04em; }

.s-divider { border:none; border-top:1px solid rgba(0,200,255,0.08); margin:6px 0; }
.s-label { font-family:'Share Tech Mono',monospace; font-size:0.65rem; letter-spacing:.18em; color:rgba(0,200,255,0.35); text-transform:uppercase; margin-bottom:6px; display:block; }

.empty-state { text-align:center; padding:80px 32px; }
.empty-state .glyph { font-size:3rem; color:rgba(0,200,255,0.15); margin-bottom:20px; display:block; font-family:'Orbitron',monospace; }
.empty-state h2 { font-family:'Orbitron',monospace; font-size:1rem; font-weight:400; color:rgba(0,200,255,0.25); letter-spacing:.1em; }
.empty-state p { font-family:'Share Tech Mono',monospace; font-size:0.65rem; color:rgba(0,200,255,0.15); margin-top:10px; letter-spacing:.08em; }
</style>
""", unsafe_allow_html=True)

# ── HERO HTML ─────────────────────────────────────────────────────────────────
HERO_HTML = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&display=swap');
*{box-sizing:border-box;margin:0;padding:0;}
#hero{position:relative;width:100%;height:260px;background:#020208;overflow:hidden;border-bottom:1px solid rgba(0,200,255,0.12);}
#matrix-canvas{position:absolute;inset:0;width:100%;height:100%;}
#particle-canvas{position:absolute;inset:0;width:100%;height:100%;pointer-events:none;}
.grid-bg{position:absolute;inset:0;pointer-events:none;
  background-image:linear-gradient(rgba(0,200,255,0.04) 1px,transparent 1px),
                   linear-gradient(90deg,rgba(0,200,255,0.04) 1px,transparent 1px);
  background-size:44px 44px;}
.scanlines{position:absolute;inset:0;pointer-events:none;
  background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,0,0,0.06) 2px,rgba(0,0,0,0.06) 4px);}
.vignette{position:absolute;inset:0;pointer-events:none;
  background:radial-gradient(ellipse at 50% 50%,transparent 30%,rgba(2,2,8,0.85) 100%);}
.corner-tl,.corner-tr{position:absolute;top:0;width:40px;height:40px;pointer-events:none;}
.corner-tl{left:0;border-top:1px solid rgba(0,200,255,0.4);border-left:1px solid rgba(0,200,255,0.4);}
.corner-tr{right:0;border-top:1px solid rgba(0,200,255,0.4);border-right:1px solid rgba(0,200,255,0.4);}
.corner-bl,.corner-br{position:absolute;bottom:0;width:40px;height:40px;pointer-events:none;}
.corner-bl{left:0;border-bottom:1px solid rgba(0,200,255,0.4);border-left:1px solid rgba(0,200,255,0.4);}
.corner-br{right:0;border-bottom:1px solid rgba(0,200,255,0.4);border-right:1px solid rgba(0,200,255,0.4);}
.status{position:absolute;top:16px;right:20px;display:flex;gap:10px;align-items:center;z-index:10;}
.dot-live{width:7px;height:7px;border-radius:50%;background:#00ff88;box-shadow:0 0 8px #00ff88;animation:bp 1.8s infinite;}
@keyframes bp{0%,100%{opacity:1;}50%{opacity:.2;}}
.stat-txt{font-family:'Share Tech Mono',monospace;font-size:9px;color:rgba(0,200,255,0.5);letter-spacing:.12em;}
.hero-content{position:absolute;bottom:24px;left:28px;z-index:10;}
.hero-tag{font-family:'Share Tech Mono',monospace;font-size:9px;letter-spacing:.2em;color:rgba(0,220,255,0.6);
  border:1px solid rgba(0,220,255,0.25);padding:3px 10px;border-radius:2px;display:inline-block;margin-bottom:12px;
  position:relative;}
.hero-tag::before{content:'';position:absolute;left:-1px;top:-1px;width:5px;height:5px;border-top:1px solid #00dcff;border-left:1px solid #00dcff;}
.hero-tag::after{content:'';position:absolute;right:-1px;bottom:-1px;width:5px;height:5px;border-bottom:1px solid #00dcff;border-right:1px solid #00dcff;}
.hero-h1{font-family:'Orbitron',monospace;font-size:2.5rem;font-weight:900;line-height:1.05;color:#fff;letter-spacing:-.01em;text-shadow:0 0 50px rgba(0,200,255,0.35);}
.hero-h1 span{color:#00dcff;display:inline-block;animation:glow 2.5s ease-in-out infinite alternate;}
@keyframes glow{from{text-shadow:0 0 10px rgba(0,200,255,0.3);}to{text-shadow:0 0 30px rgba(0,200,255,0.8),0 0 60px rgba(0,200,255,0.3);}}
.hero-sub{font-family:'Share Tech Mono',monospace;font-size:.65rem;color:rgba(0,200,255,0.35);letter-spacing:.12em;margin-top:8px;text-transform:uppercase;}
</style>

<div id="hero">
  <canvas id="matrix-canvas"></canvas>
  <canvas id="particle-canvas"></canvas>
  <div class="grid-bg"></div>
  <div class="scanlines"></div>
  <div class="vignette"></div>
  <div class="corner-tl"></div><div class="corner-tr"></div>
  <div class="corner-bl"></div><div class="corner-br"></div>
  <div class="status">
    <div class="dot-live"></div>
    <span class="stat-txt">SYSTEM ONLINE</span>
    <span class="stat-txt" id="hclock">--:--:--</span>
  </div>
  <div class="hero-content">
    <div class="hero-tag">✦ PDF INTELLIGENCE · RAG v2.0</div>
    <div class="hero-h1">QUERY YOUR<br><span>DOCUMENT</span></div>
    <div class="hero-sub">// claude ai · faiss vector search · sentence transformers</div>
  </div>
</div>

<script>
(function(){
  function tick(){
    var n=new Date(),e=document.getElementById('hclock');
    if(e)e.textContent=[n.getHours(),n.getMinutes(),n.getSeconds()].map(function(x){return String(x).padStart(2,'0');}).join(':');
  }
  setInterval(tick,1000);tick();

  var mc=document.getElementById('matrix-canvas');
  var mctx=mc.getContext('2d');
  var hero=document.getElementById('hero');
  var W,H,cols=[];
  var chars='01アイウエオカキクサシスタチツテトナニネノハヒフヘホ';
  var FS=13;

  function resizeM(){
    W=mc.width=hero.offsetWidth;
    H=mc.height=hero.offsetHeight;
    cols=[];
    var n=Math.floor(W/FS);
    for(var i=0;i<n;i++)cols.push({y:Math.random()*H,speed:Math.random()*1.4+0.3,bright:Math.random()>.82,trail:Math.floor(Math.random()*8)+4});
  }

  function drawMatrix(){
    mctx.fillStyle='rgba(2,2,8,0.16)';
    mctx.fillRect(0,0,W,H);
    mctx.font=FS+'px monospace';
    cols.forEach(function(c,i){
      var ch=chars[Math.floor(Math.random()*chars.length)];
      if(c.bright){mctx.fillStyle='rgba(200,240,255,0.95)';}
      else{mctx.fillStyle='rgba(0,180,255,0.15)';}
      mctx.fillText(ch,i*FS,c.y);
      c.y+=c.speed;
      if(c.y>H+FS){c.y=-FS*c.trail;c.bright=Math.random()>.82;}
    });
  }

  var pc=document.getElementById('particle-canvas');
  var pctx=pc.getContext('2d');
  var PW,PH,particles=[];

  function resizeP(){PW=pc.width=hero.offsetWidth;PH=pc.height=hero.offsetHeight;}

  function initParticles(){
    particles=[];
    for(var i=0;i<55;i++)particles.push({
      x:Math.random()*PW,y:Math.random()*PH,
      vx:(Math.random()-.5)*.5,vy:(Math.random()-.5)*.5,
      r:Math.random()*1.8+.4,a:Math.random()*.5+.15
    });
  }

  function drawParticles(){
    pctx.clearRect(0,0,PW,PH);
    particles.forEach(function(p){
      p.x+=p.vx;p.y+=p.vy;
      if(p.x<0)p.x=PW;if(p.x>PW)p.x=0;
      if(p.y<0)p.y=PH;if(p.y>PH)p.y=0;
      pctx.beginPath();pctx.arc(p.x,p.y,p.r,0,Math.PI*2);
      pctx.fillStyle='rgba(0,220,255,'+p.a+')';pctx.fill();
    });
    for(var i=0;i<particles.length;i++){
      for(var j=i+1;j<particles.length;j++){
        var d=Math.hypot(particles[i].x-particles[j].x,particles[i].y-particles[j].y);
        if(d<90){
          pctx.beginPath();pctx.moveTo(particles[i].x,particles[i].y);pctx.lineTo(particles[j].x,particles[j].y);
          pctx.strokeStyle='rgba(0,180,255,'+(0.12*(1-d/90))+')';pctx.lineWidth=0.5;pctx.stroke();
        }
      }
    }
  }

  function loop(){drawMatrix();drawParticles();requestAnimationFrame(loop);}
  resizeM();resizeP();initParticles();loop();
  window.addEventListener('resize',function(){resizeM();resizeP();initParticles();});
})();
</script>
"""

# ── MODELS ────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    m = SentenceTransformer('all-MiniLM-L6-v2')
    c = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return m, c

model, anthropic_client = load_models()

def extract_chunks(text, chunk_size=500, overlap=100):
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i:i+chunk_size]))
        i += chunk_size - overlap
    return chunks

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:14px 12px 10px;border-bottom:1px solid rgba(0,200,255,0.1);margin-bottom:14px;">
      <div style="font-family:'Orbitron',monospace;font-size:0.85rem;font-weight:700;color:#00dcff;letter-spacing:.06em;">PDF BOT</div>
      <div style="font-family:'Share Tech Mono',monospace;font-size:0.6rem;color:rgba(0,200,255,0.3);letter-spacing:.12em;margin-top:2px;">INTELLIGENCE SYSTEM</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<span class="s-label">// INPUT</span>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type="pdf", label_visibility="collapsed")

    if uploaded_file:
        st.markdown(f"""
        <div style="font-family:'Share Tech Mono',monospace;font-size:0.68rem;color:rgba(0,200,255,0.5);padding:6px 4px;border-left:2px solid rgba(0,200,255,0.3);margin-bottom:8px;word-break:break-all;">
          › {uploaded_file.name}
        </div>""", unsafe_allow_html=True)

        if st.button("⚡  PROCESS & INDEX"):
            with st.spinner("Indexing…"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                doc = fitz.open(tmp_path)
                full_text = ""
                prog = st.progress(0)
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
                                model="claude-haiku-4-5-20251001", max_tokens=300,
                                messages=[{"role":"user","content":[
                                    {"type":"image","source":{"type":"base64","media_type":mt,"data":b64}},
                                    {"type":"text","text":"Describe this image. If it's a diagram or chart, explain what it shows."}
                                ]}]
                            )
                            full_text += f"\n[Image p{pnum+1}]: {resp.content[0].text}\n"
                        except Exception:
                            continue
                    prog.progress((pnum+1)/total)

                doc.close()
                os.unlink(tmp_path)
                chunks = extract_chunks(full_text)
                embs = np.array(model.encode(chunks)).astype('float32')
                index = faiss.IndexFlatL2(embs.shape[1])
                index.add(embs)
                st.session_state.index  = index
                st.session_state.chunks = chunks
                st.session_state.messages = []
                st.session_state.chunk_count = len(chunks)

            st.success(f"✓ {len(chunks)} chunks indexed")

    st.markdown('<hr class="s-divider">', unsafe_allow_html=True)

    # metrics
    chunk_count = st.session_state.get("chunk_count", 0)
    st.markdown(f"""
    <span class="s-label">// SYSTEM METRICS</span>
    <div class="metric-grid">
      <div class="metric-card"><div class="metric-val">{chunk_count if chunk_count else '—'}</div><div class="metric-lbl">Chunks</div></div>
      <div class="metric-card"><div class="metric-val">384</div><div class="metric-lbl">Embed dim</div></div>
      <div class="metric-card"><div class="metric-val">L2</div><div class="metric-lbl">FAISS</div></div>
      <div class="metric-card"><div class="metric-val">TOP 3</div><div class="metric-lbl">Retrieval</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="s-divider">', unsafe_allow_html=True)

    st.markdown("""
    <span class="s-label">// QUERY TIPS</span>
    <div class="sys-tips">
      › ask specific questions<br>
      › reference page numbers<br>
      › request summaries<br>
      › ask about diagrams<br>
      › cross-reference sections
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="position:absolute;bottom:14px;left:12px;right:12px;font-family:'Share Tech Mono',monospace;
      font-size:8px;color:rgba(0,200,255,0.15);letter-spacing:.08em;text-align:center;border-top:1px solid rgba(0,200,255,0.06);padding-top:10px;">
      POWERED BY CLAUDE · FAISS · SENTENCE-TRANSFORMERS
    </div>
    """, unsafe_allow_html=True)

# ── MAIN ──────────────────────────────────────────────────────────────────────
st.components.v1.html(HERO_HTML, height=262, scrolling=False)

if "index" not in st.session_state:
    st.markdown("""
    <div class="empty-state">
      <span class="glyph">⬡</span>
      <h2>UPLOAD A DOCUMENT TO BEGIN</h2>
      <p>// drop a pdf in the sidebar · embeddings indexed locally · ask anything</p>
    </div>
    """, unsafe_allow_html=True)
else:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    for msg in st.session_state.messages:
        display = msg["content"]
        if msg["role"] == "user" and "Question: " in display:
            display = display.split("Question: ")[-1]
        with st.chat_message(msg["role"]):
            st.write(display)

    question = st.chat_input("// query the document…")

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

        with st.spinner("// processing query…"):
            response = anthropic_client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1024,
                system="You are an intelligent document assistant. Answer questions based on the provided context. If the exact term isn't found, look for related concepts. Only say you cannot find it if there is truly nothing related in the context.",
                messages=st.session_state.messages
            )

        reply = response.content[0].text
        st.session_state.messages.append({"role": "assistant", "content": reply})

        with st.chat_message("assistant"):
            st.write(reply)