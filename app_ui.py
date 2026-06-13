import streamlit as st
import anthropic
from sentence_transformers import SentenceTransformer
import fitz
import tempfile
import os
import base64
import numpy as np
import faiss
import time
import re
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="PDF BOT", page_icon="⬡", layout="wide")

# ── GLOBAL CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&family=Inter:wght@300;400;500&display=swap');
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
#MainMenu,footer,header,[data-testid="stToolbar"],.stDeployButton{display:none!important;visibility:hidden!important;}
html,body,.stApp,[data-testid="stAppViewContainer"]{background:#020208!important;color:#c8e0f0!important;font-family:'Inter',sans-serif!important;}

[data-testid="stSidebar"]{background:rgba(0,4,18,0.97)!important;border-right:1px solid rgba(0,200,255,0.12)!important;}
[data-testid="stSidebar"]>div{padding-top:0!important;}

[data-testid="stFileUploader"]{background:rgba(0,200,255,0.03)!important;border:1px solid rgba(0,200,255,0.2)!important;border-radius:6px!important;}
[data-testid="stFileUploader"]:hover{border-color:rgba(0,200,255,0.6)!important;}
[data-testid="stFileUploader"] label{color:rgba(0,200,255,0.6)!important;font-family:'Share Tech Mono',monospace!important;font-size:0.72rem!important;}
[data-testid="stFileUploader"] section{border:none!important;background:transparent!important;}
[data-testid="stFileUploader"] button{border-color:rgba(0,200,255,0.4)!important;color:#00dcff!important;background:transparent!important;}

.stButton>button{background:transparent!important;border:1px solid rgba(0,200,255,0.4)!important;color:#00dcff!important;border-radius:4px!important;font-family:'Share Tech Mono',monospace!important;font-size:0.7rem!important;letter-spacing:0.12em!important;width:100%!important;padding:10px!important;text-transform:uppercase!important;transition:all 0.2s!important;}
.stButton>button:hover{background:rgba(0,200,255,0.1)!important;border-color:#00dcff!important;box-shadow:0 0 20px rgba(0,200,255,0.25)!important;}

.stProgress>div>div{background:linear-gradient(90deg,#00dcff,#00ff88)!important;}
.stProgress>div{background:rgba(0,200,255,0.08)!important;border-radius:0!important;}
.stSuccess{background:rgba(0,255,136,0.06)!important;border:1px solid rgba(0,255,136,0.2)!important;border-radius:4px!important;color:#00ff88!important;font-family:'Share Tech Mono',monospace!important;font-size:0.75rem!important;}
.stSpinner>div{border-top-color:#00dcff!important;}

.block-container{padding:0!important;max-width:100%!important;}

[data-testid="stChatMessage"]{background:transparent!important;border:none!important;padding:4px 28px!important;margin-bottom:6px!important;}
[data-testid="stChatMessage"]>div:last-child{background:rgba(255,255,255,0.02)!important;border:1px solid rgba(255,255,255,0.07)!important;border-radius:6px 6px 6px 2px!important;padding:14px 18px!important;font-size:0.86rem!important;line-height:1.7!important;color:#b8cce0!important;font-family:'Inter',sans-serif!important;}
[data-testid="stChatMessage"]:has([aria-label="user avatar"])>div:last-child{background:rgba(0,200,255,0.07)!important;border-color:rgba(0,200,255,0.18)!important;border-radius:6px 6px 2px 6px!important;color:#a0d8f0!important;}
[data-testid="stChatMessage"] [data-testid*="Avatar"],[data-testid="stChatMessage"] img{background:rgba(0,200,255,0.1)!important;border:1px solid rgba(0,200,255,0.3)!important;border-radius:50%!important;}

[data-testid="stChatInput"]{background:rgba(0,2,15,0.95)!important;border-top:1px solid rgba(0,200,255,0.15)!important;padding:16px 28px!important;}
[data-testid="stChatInput"] textarea{background:rgba(0,200,255,0.04)!important;border:1px solid rgba(0,200,255,0.25)!important;border-radius:6px!important;color:#a0d8f0!important;font-family:'Share Tech Mono',monospace!important;font-size:0.85rem!important;transition:all 0.3s!important;}
[data-testid="stChatInput"] textarea:focus{border-color:rgba(0,200,255,0.8)!important;box-shadow:0 0 0 1px rgba(0,200,255,0.2),0 0 24px rgba(0,200,255,0.12)!important;background:rgba(0,200,255,0.07)!important;}
[data-testid="stChatInput"] textarea::placeholder{color:rgba(0,200,255,0.2)!important;}
[data-testid="stChatInput"] button{background:rgba(0,200,255,0.15)!important;border:1px solid rgba(0,200,255,0.4)!important;color:#00dcff!important;border-radius:6px!important;transition:all 0.2s!important;}
[data-testid="stChatInput"] button:hover{background:rgba(0,200,255,0.3)!important;box-shadow:0 0 16px rgba(0,200,255,0.3)!important;}

::-webkit-scrollbar{width:3px;height:3px;}
::-webkit-scrollbar-track{background:transparent;}
::-webkit-scrollbar-thumb{background:rgba(0,200,255,0.25);border-radius:2px;}
::-webkit-scrollbar-thumb:hover{background:rgba(0,200,255,0.5);}

.metric-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;}
.metric-card{background:rgba(0,200,255,0.03);border:1px solid rgba(0,200,255,0.1);border-radius:4px;padding:10px 8px;text-align:center;transition:border-color 0.3s;}
.metric-card:hover{border-color:rgba(0,200,255,0.35);}
.metric-val{font-family:'Orbitron',monospace;font-size:0.85rem;font-weight:700;color:#00dcff;}
.metric-lbl{font-family:'Share Tech Mono',monospace;font-size:0.58rem;color:rgba(0,200,255,0.35);margin-top:3px;letter-spacing:.06em;}
.sys-tips{font-family:'Share Tech Mono',monospace;font-size:0.65rem;color:rgba(0,200,255,0.3);line-height:2.1;letter-spacing:.04em;}
.s-divider{border:none;border-top:1px solid rgba(0,200,255,0.07);margin:10px 0;}
.s-label{font-family:'Share Tech Mono',monospace;font-size:0.62rem;letter-spacing:.18em;color:rgba(0,200,255,0.3);text-transform:uppercase;margin-bottom:8px;display:block;}

.empty-state{text-align:center;padding:80px 32px;}
.empty-glyph{font-size:3rem;color:rgba(0,200,255,0.12);margin-bottom:20px;display:block;font-family:'Orbitron',monospace;animation:pulse-glyph 3s ease-in-out infinite;}
@keyframes pulse-glyph{0%,100%{opacity:.12;}50%{opacity:.3;}}
.empty-state h2{font-family:'Orbitron',monospace;font-size:0.95rem;font-weight:400;color:rgba(0,200,255,0.2);letter-spacing:.12em;}
.empty-state p{font-family:'Share Tech Mono',monospace;font-size:0.63rem;color:rgba(0,200,255,0.12);margin-top:10px;letter-spacing:.08em;}

.img-frame{border:1px solid rgba(0,200,255,0.2);border-radius:6px;overflow:hidden;margin:10px 0;background:#000;}
.img-frame-label{font-family:'Share Tech Mono',monospace;font-size:0.62rem;color:rgba(0,200,255,0.5);padding:6px 12px;border-bottom:1px solid rgba(0,200,255,0.1);letter-spacing:.1em;display:flex;justify-content:space-between;}

.contact-row{display:flex;gap:8px;margin-top:8px;}
.contact-link{flex:1;display:flex;align-items:center;gap:6px;padding:8px 10px;border:1px solid rgba(0,200,255,0.12);border-radius:4px;text-decoration:none;font-family:'Share Tech Mono',monospace;font-size:0.6rem;color:rgba(0,200,255,0.45);letter-spacing:.06em;transition:all 0.2s;background:rgba(0,200,255,0.02);}
.contact-link:hover{border-color:rgba(0,200,255,0.4);color:#00dcff;background:rgba(0,200,255,0.07);}
.contact-icon{font-size:0.7rem;}
</style>
""", unsafe_allow_html=True)

# ── HERO ──────────────────────────────────────────────────────────────────────
HERO_HTML = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&display=swap');
*{box-sizing:border-box;margin:0;padding:0;}
#hero{position:relative;width:100%;height:260px;background:#020208;overflow:hidden;border-bottom:1px solid rgba(0,200,255,0.12);}
#mc{position:absolute;inset:0;width:100%;height:100%;}
#pc{position:absolute;inset:0;width:100%;height:100%;pointer-events:none;}
.gbg{position:absolute;inset:0;pointer-events:none;background-image:linear-gradient(rgba(0,200,255,0.035) 1px,transparent 1px),linear-gradient(90deg,rgba(0,200,255,0.035) 1px,transparent 1px);background-size:44px 44px;}
.sl{position:absolute;inset:0;pointer-events:none;background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,0,0,0.055) 2px,rgba(0,0,0,0.055) 4px);}
.vig{position:absolute;inset:0;pointer-events:none;background:radial-gradient(ellipse at 50% 50%,transparent 20%,rgba(2,2,8,0.88) 100%);}
.ctl{position:absolute;top:0;left:0;width:40px;height:40px;pointer-events:none;border-top:1px solid rgba(0,200,255,0.45);border-left:1px solid rgba(0,200,255,0.45);}
.ctr{position:absolute;top:0;right:0;width:40px;height:40px;pointer-events:none;border-top:1px solid rgba(0,200,255,0.45);border-right:1px solid rgba(0,200,255,0.45);}
.cbl{position:absolute;bottom:0;left:0;width:40px;height:40px;pointer-events:none;border-bottom:1px solid rgba(0,200,255,0.45);border-left:1px solid rgba(0,200,255,0.45);}
.cbr{position:absolute;bottom:0;right:0;width:40px;height:40px;pointer-events:none;border-bottom:1px solid rgba(0,200,255,0.45);border-right:1px solid rgba(0,200,255,0.45);}
.status{position:absolute;top:16px;right:20px;display:flex;gap:10px;align-items:center;z-index:10;}
.dot{width:7px;height:7px;border-radius:50%;background:#00ff88;box-shadow:0 0 8px #00ff88;animation:bp 1.8s infinite;}
@keyframes bp{0%,100%{opacity:1;}50%{opacity:.15;}}
.stxt{font-family:'Share Tech Mono',monospace;font-size:9px;color:rgba(0,200,255,0.5);letter-spacing:.12em;}
.hc{position:absolute;bottom:24px;left:28px;z-index:10;}
.htag{font-family:'Share Tech Mono',monospace;font-size:9px;letter-spacing:.2em;color:rgba(0,220,255,0.65);border:1px solid rgba(0,220,255,0.25);padding:3px 10px;border-radius:2px;display:inline-block;margin-bottom:12px;position:relative;}
.htag::before{content:'';position:absolute;left:-1px;top:-1px;width:5px;height:5px;border-top:1px solid #00dcff;border-left:1px solid #00dcff;}
.htag::after{content:'';position:absolute;right:-1px;bottom:-1px;width:5px;height:5px;border-bottom:1px solid #00dcff;border-right:1px solid #00dcff;}
.hh1{font-family:'Orbitron',monospace;font-size:2.5rem;font-weight:900;line-height:1.05;color:#fff;text-shadow:0 0 50px rgba(0,200,255,0.3);}
.hh1 span{color:#00dcff;animation:glow 2.5s ease-in-out infinite alternate;}
@keyframes glow{from{text-shadow:0 0 8px rgba(0,200,255,0.3);}to{text-shadow:0 0 28px rgba(0,200,255,0.9),0 0 55px rgba(0,200,255,0.3);}}
.hsub{font-family:'Share Tech Mono',monospace;font-size:.65rem;color:rgba(0,200,255,0.32);letter-spacing:.12em;margin-top:8px;text-transform:uppercase;}
</style>
<div id="hero">
  <canvas id="mc"></canvas><canvas id="pc"></canvas>
  <div class="gbg"></div><div class="sl"></div><div class="vig"></div>
  <div class="ctl"></div><div class="ctr"></div><div class="cbl"></div><div class="cbr"></div>
  <div class="status"><div class="dot"></div><span class="stxt">SYSTEM ONLINE</span><span class="stxt" id="clk">--:--:--</span></div>
  <div class="hc">
    <div class="htag">✦ PDF INTELLIGENCE · RAG v2.0</div>
    <div class="hh1">QUERY YOUR<br><span>DOCUMENT</span></div>
    <div class="hsub">// claude ai · faiss vector search · sentence transformers</div>
  </div>
</div>
<script>
(function(){
  function tick(){var n=new Date(),e=document.getElementById('clk');if(e)e.textContent=[n.getHours(),n.getMinutes(),n.getSeconds()].map(function(x){return String(x).padStart(2,'0');}).join(':');}
  setInterval(tick,1000);tick();
  var mc=document.getElementById('mc'),mctx=mc.getContext('2d'),hero=document.getElementById('hero');
  var W,H,cols=[],chars='01アイウエオカキクサシスタチツテトナニネノハヒフヘホ',FS=13;
  function resM(){W=mc.width=hero.offsetWidth;H=mc.height=hero.offsetHeight;cols=[];var n=Math.floor(W/FS);for(var i=0;i<n;i++)cols.push({y:Math.random()*H,sp:Math.random()*1.4+0.3,br:Math.random()>.82,tr:Math.floor(Math.random()*8)+4});}
  function drM(){mctx.fillStyle='rgba(2,2,8,0.16)';mctx.fillRect(0,0,W,H);mctx.font=FS+'px monospace';cols.forEach(function(c,i){var ch=chars[Math.floor(Math.random()*chars.length)];mctx.fillStyle=c.br?'rgba(200,240,255,0.95)':'rgba(0,180,255,0.14)';mctx.fillText(ch,i*FS,c.y);c.y+=c.sp;if(c.y>H+FS){c.y=-FS*c.tr;c.br=Math.random()>.82;}});}
  var pc=document.getElementById('pc'),pctx=pc.getContext('2d'),PW,PH,pts=[];
  function resP(){PW=pc.width=hero.offsetWidth;PH=pc.height=hero.offsetHeight;}
  function initP(){pts=[];for(var i=0;i<55;i++)pts.push({x:Math.random()*PW,y:Math.random()*PH,vx:(Math.random()-.5)*.5,vy:(Math.random()-.5)*.5,r:Math.random()*1.8+.4,a:Math.random()*.5+.15});}
  function drP(){pctx.clearRect(0,0,PW,PH);pts.forEach(function(p){p.x+=p.vx;p.y+=p.vy;if(p.x<0)p.x=PW;if(p.x>PW)p.x=0;if(p.y<0)p.y=PH;if(p.y>PH)p.y=0;pctx.beginPath();pctx.arc(p.x,p.y,p.r,0,Math.PI*2);pctx.fillStyle='rgba(0,220,255,'+p.a+')';pctx.fill();});for(var i=0;i<pts.length;i++)for(var j=i+1;j<pts.length;j++){var d=Math.hypot(pts[i].x-pts[j].x,pts[i].y-pts[j].y);if(d<90){pctx.beginPath();pctx.moveTo(pts[i].x,pts[i].y);pctx.lineTo(pts[j].x,pts[j].y);pctx.strokeStyle='rgba(0,180,255,'+(0.12*(1-d/90))+')';pctx.lineWidth=0.5;pctx.stroke();}}}
  function loop(){drM();drP();requestAnimationFrame(loop);}
  resM();resP();initP();loop();
  window.addEventListener('resize',function(){resM();resP();initP();});
})();
</script>
"""

# ── ANIMATED CHAT INPUT BANNER ────────────────────────────────────────────────
CHAT_BANNER_HTML = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@700&display=swap');
#cb{position:relative;width:100%;overflow:hidden;background:rgba(0,2,15,0.6);border-top:1px solid rgba(0,200,255,0.1);border-bottom:1px solid rgba(0,200,255,0.08);padding:0;}
#cb-canvas{position:absolute;inset:0;width:100%;height:100%;pointer-events:none;}
.cb-inner{position:relative;z-index:2;display:flex;align-items:center;gap:16px;padding:10px 28px;}
.cb-prompt{font-family:'Orbitron',monospace;font-size:0.7rem;font-weight:700;color:rgba(0,200,255,0.5);letter-spacing:.15em;white-space:nowrap;}
.cb-tags{display:flex;gap:8px;flex-wrap:wrap;}
.cb-tag{font-family:'Share Tech Mono',monospace;font-size:0.62rem;color:rgba(0,200,255,0.35);border:1px solid rgba(0,200,255,0.12);padding:3px 10px;border-radius:2px;letter-spacing:.06em;animation:tag-pulse 3s ease-in-out infinite;}
.cb-tag:nth-child(2){animation-delay:.4s;}
.cb-tag:nth-child(3){animation-delay:.8s;}
.cb-tag:nth-child(4){animation-delay:1.2s;}
.cb-tag:nth-child(5){animation-delay:1.6s;}
@keyframes tag-pulse{0%,100%{opacity:.35;border-color:rgba(0,200,255,0.12);}50%{opacity:.75;border-color:rgba(0,200,255,0.35);color:rgba(0,220,255,0.65);}}
</style>
<div id="cb">
  <canvas id="cb-canvas" height="40"></canvas>
  <div class="cb-inner">
    <div class="cb-prompt">TRY →</div>
    <div class="cb-tags">
      <div class="cb-tag">summarize this</div>
      <div class="cb-tag">show 1st image</div>
      <div class="cb-tag">what is chapter 3 about?</div>
      <div class="cb-tag">show all figures</div>
      <div class="cb-tag">key findings?</div>
    </div>
  </div>
</div>
<script>
(function(){
  var cv=document.getElementById('cb-canvas'),ctx=cv.getContext('2d');
  var W,H;
  function res(){W=cv.width=cv.parentElement.offsetWidth;H=cv.height=40;}
  var t=0;
  function dr(){
    ctx.clearRect(0,0,W,H);
    for(var i=0;i<6;i++){
      var x=((t*0.4+i*130)%W+W)%W;
      ctx.fillStyle='rgba(0,200,255,'+(0.03+i%3*0.01)+')';
      ctx.fillRect(x,0,60,H);
    }
    t++;requestAnimationFrame(dr);
  }
  res();dr();
  window.addEventListener('resize',res);
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

def is_image_request(q):
    q = q.lower()
    return any(t in q for t in ["show","display","see","view","image","figure","picture","photo","diagram","chart","graph","illustration","img"])

def find_matching_images(question, image_store):
    if not image_store:
        return []
    q = question.lower()
    ordinals = {"1st":0,"first":0,"2nd":1,"second":1,"3rd":2,"third":2,"4th":3,"fourth":3,"5th":4,"fifth":4,"6th":5,"sixth":5}
    for word, idx in ordinals.items():
        if word in q and idx < len(image_store):
            return [image_store[idx]]
    nums = re.findall(r'\b(\d+)\b', q)
    for n in nums:
        idx = int(n) - 1
        if 0 <= idx < len(image_store):
            return [image_store[idx]]
    if "last" in q:
        return [image_store[-1]]
    if "all" in q:
        return image_store
    if any(t in q for t in ["image","figure","picture","diagram","chart","graph"]):
        return [image_store[0]]
    return []

def fmt_time(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{int(seconds//60)}m {seconds%60:.0f}s"

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:14px 14px 10px;border-bottom:1px solid rgba(0,200,255,0.1);margin-bottom:14px;">
      <div style="font-family:'Orbitron',monospace;font-size:0.85rem;font-weight:700;color:#00dcff;letter-spacing:.06em;">PDF BOT</div>
      <div style="font-family:'Share Tech Mono',monospace;font-size:0.58rem;color:rgba(0,200,255,0.25);letter-spacing:.12em;margin-top:2px;">INTELLIGENCE SYSTEM</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<span class="s-label">// INPUT</span>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type="pdf", label_visibility="collapsed")

    if uploaded_file:
        st.markdown(f"""
        <div style="font-family:'Share Tech Mono',monospace;font-size:0.66rem;color:rgba(0,200,255,0.45);padding:6px 8px;border-left:2px solid rgba(0,200,255,0.25);margin-bottom:8px;word-break:break-all;">
          › {uploaded_file.name}
        </div>""", unsafe_allow_html=True)

        if st.button("⚡  PROCESS & INDEX"):
            t_start = time.time()
            with st.spinner("Indexing…"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                doc = fitz.open(tmp_path)
                full_text = ""
                prog = st.progress(0)
                total = len(doc)
                image_store = []
                img_global_idx = 0

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
                                    {"type":"text","text":"Describe this image concisely. If it's a diagram/chart, explain what it shows."}
                                ]}]
                            )
                            desc = resp.content[0].text
                            image_store.append({"b64":b64,"media_type":mt,"page":pnum+1,"index":img_global_idx,"description":desc})
                            img_global_idx += 1
                            full_text += f"\n[Image {img_global_idx} on page {pnum+1}]: {desc}\n"
                        except Exception:
                            continue
                    prog.progress((pnum+1)/total)

                doc.close()
                os.unlink(tmp_path)
                t_process = time.time() - t_start

                t_embed = time.time()
                chunks = extract_chunks(full_text)
                embs = np.array(model.encode(chunks)).astype('float32')
                index = faiss.IndexFlatL2(embs.shape[1])
                index.add(embs)
                t_embed_done = time.time() - t_embed

                st.session_state.update({
                    "index": index,
                    "chunks": chunks,
                    "messages": [],
                    "chat_display": [],
                    "chunk_count": len(chunks),
                    "image_store": image_store,
                    "image_count": len(image_store),
                    "page_count": total,
                    "process_time": t_process,
                    "embed_time": t_embed_done,
                })

            st.success(f"✓ {len(chunks)} chunks · {len(image_store)} images · {fmt_time(t_process)}")

    st.markdown('<hr class="s-divider">', unsafe_allow_html=True)

    chunk_count  = st.session_state.get("chunk_count", 0)
    img_count    = st.session_state.get("image_count", 0)
    page_count   = st.session_state.get("page_count", 0)
    proc_time    = st.session_state.get("process_time", None)
    embed_time   = st.session_state.get("embed_time", None)

    proc_disp  = fmt_time(proc_time)  if proc_time  else "—"
    embed_disp = fmt_time(embed_time) if embed_time else "—"

    st.markdown(f"""
    <span class="s-label">// LAST RUN STATS</span>
    <div class="metric-grid">
      <div class="metric-card"><div class="metric-val">{chunk_count if chunk_count else '—'}</div><div class="metric-lbl">Chunks</div></div>
      <div class="metric-card"><div class="metric-val">{img_count if img_count else '—'}</div><div class="metric-lbl">Images</div></div>
      <div class="metric-card"><div class="metric-val">{proc_disp}</div><div class="metric-lbl">Process time</div></div>
      <div class="metric-card"><div class="metric-val">{embed_disp}</div><div class="metric-lbl">Index time</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="s-divider">', unsafe_allow_html=True)

    st.markdown("""
    <span class="s-label">// QUERY TIPS</span>
    <div class="sys-tips">
      › ask specific questions<br>
      › "show me the 1st image"<br>
      › "show all figures"<br>
      › reference page numbers<br>
      › request summaries
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="s-divider">', unsafe_allow_html=True)

    st.markdown("""
    <span class="s-label">// BUILT BY</span>
    <div style="font-family:'Share Tech Mono',monospace;font-size:0.65rem;color:rgba(0,200,255,0.45);margin-bottom:8px;letter-spacing:.04em;">
      Sai Jyothi Gayathri Adabala
    </div>
    <div class="contact-row">
      <a class="contact-link" href="mailto:asjyothig@gmail.com">
        <span class="contact-icon">✉</span> email
      </a>
      <a class="contact-link" href="https://www.linkedin.com/in/sai-jyothi-gayathri-adabala-a41a9818b/" target="_blank">
        <span class="contact-icon">in</span> linkedin
      </a>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:16px;font-family:'Share Tech Mono',monospace;font-size:7.5px;
      color:rgba(0,200,255,0.12);letter-spacing:.08em;text-align:center;
      border-top:1px solid rgba(0,200,255,0.05);padding-top:10px;">
      CLAUDE · FAISS · SENTENCE-TRANSFORMERS
    </div>
    """, unsafe_allow_html=True)

# ── MAIN ──────────────────────────────────────────────────────────────────────
st.components.v1.html(HERO_HTML, height=262, scrolling=False)

if "index" not in st.session_state:
    st.markdown("""
    <div class="empty-state">
      <span class="empty-glyph">⬡</span>
      <h2>UPLOAD A DOCUMENT TO BEGIN</h2>
      <p>// drop a pdf in the sidebar · ask anything · images supported</p>
    </div>
    """, unsafe_allow_html=True)
else:
    if "chat_display" not in st.session_state:
        st.session_state.chat_display = []
    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.components.v1.html(CHAT_BANNER_HTML, height=44, scrolling=False)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    for entry in st.session_state.chat_display:
        with st.chat_message(entry["role"]):
            st.write(entry["content"])
            if entry.get("images"):
                for img_data in entry["images"]:
                    st.markdown(
                        f'<div class="img-frame"><div class="img-frame-label">'
                        f'<span>// IMAGE {img_data["index"]+1}</span>'
                        f'<span>PAGE {img_data["page"]}</span></div></div>',
                        unsafe_allow_html=True
                    )
                    img_bytes = base64.b64decode(img_data["b64"])
                    st.image(img_bytes, use_column_width=False)

    question = st.chat_input("// query the document…")

    if question:
        with st.chat_message("user"):
            st.write(question)
        st.session_state.chat_display.append({"role":"user","content":question,"images":[]})

        images_to_show = []
        reply = ""

        if is_image_request(question):
            images_to_show = find_matching_images(question, st.session_state.get("image_store", []))

        q_emb = np.array([model.encode(question)]).astype('float32')
        _, idxs = st.session_state.index.search(q_emb, 3)
        context = "\n\n".join(st.session_state.chunks[i] for i in idxs[0])

        if images_to_show:
            st.session_state.messages.append({
                "role":"user",
                "content":f"Context:\n{context}\n\nQuestion: {question}\n\nNote: You ARE showing the user the actual image(s). Briefly describe what they will see in 1-2 sentences."
            })
            system_prompt = "You are a document assistant. The user asked to see an image and you ARE displaying it. Briefly describe what the image shows in 1-2 sentences. Never say you cannot show images."
            max_tok = 250
        else:
            st.session_state.messages.append({
                "role":"user",
                "content":f"Context:\n{context}\n\nQuestion: {question}"
            })
            system_prompt = "You are an intelligent document assistant. Answer questions based on the provided context. If the exact term isn't found, look for related concepts. Only say you cannot find it if there is truly nothing related in context."
            max_tok = 1024

        with st.spinner("// processing…"):
            response = anthropic_client.messages.create(
                model="claude-haiku-4-5-20251001", max_tokens=max_tok,
                system=system_prompt,
                messages=st.session_state.messages
            )
        reply = response.content[0].text
        st.session_state.messages.append({"role":"assistant","content":reply})
        st.session_state.chat_display.append({"role":"assistant","content":reply,"images":images_to_show})

        with st.chat_message("assistant"):
            st.write(reply)
            if images_to_show:
                for img_data in images_to_show:
                    st.markdown(
                        f'<div class="img-frame"><div class="img-frame-label">'
                        f'<span>// IMAGE {img_data["index"]+1}</span>'
                        f'<span>PAGE {img_data["page"]}</span></div></div>',
                        unsafe_allow_html=True
                    )
                    img_bytes = base64.b64decode(img_data["b64"])
                    st.image(img_bytes, use_column_width=False)