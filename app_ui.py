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
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="PDF BOT", page_icon="⬡", layout="wide")

# ── INJECT FOCUS-RING KILLER via st.components (runs in actual DOM) ─────────
# Streamlit injects a red box-shadow via JS after page load; we fight it the
# same way — with a MutationObserver that strips it whenever it appears.
FOCUS_KILL = """
<style>
  /* nuclear option on every focusable element */
  *:focus, *:focus-visible, *:focus-within {
    outline: none !important;
    box-shadow: none !important;
  }
  textarea:focus, input:focus {
    outline: none !important;
    box-shadow: 0 0 0 2px rgba(0,200,255,0.25),
                0 0 24px rgba(0,200,255,0.35) !important;
    border-color: rgba(0,200,255,0.9) !important;
  }
</style>
<script>
(function() {
  // Strip red box-shadow from chat input whenever Streamlit re-applies it
  const BLUE_GLOW = '0 0 0 2px rgba(0,200,255,0.25), 0 0 24px rgba(0,200,255,0.35)';
  const RED_PAT   = /rgb\(255,\s*\d+,\s*\d+\)|red/i;
  function fixFocus() {
    document.querySelectorAll('textarea, input, [data-testid="stChatInput"] *').forEach(el => {
      const s = el.style.boxShadow || '';
      if (RED_PAT.test(s) || s.includes('255, 75')) {
        el.style.setProperty('box-shadow', BLUE_GLOW, 'important');
        el.style.setProperty('outline', 'none', 'important');
        el.style.setProperty('border-color', 'rgba(0,200,255,0.9)', 'important');
      }
    });
  }
  const obs = new MutationObserver(fixFocus);
  obs.observe(document.body, { subtree: true, attributes: true, attributeFilter: ['style', 'class'] });
  document.addEventListener('focusin',  fixFocus);
  document.addEventListener('focusout', fixFocus);
})();
</script>
"""
st.components.v1.html(FOCUS_KILL, height=0, scrolling=False)

# ── GLOBAL CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&family=Inter:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
#MainMenu, footer, header, [data-testid="stToolbar"], .stDeployButton {
  display: none !important; visibility: hidden !important;
}
html, body, .stApp, [data-testid="stAppViewContainer"] {
  background: #020208 !important;
  color: #c8e0f0 !important;
  font-family: 'Inter', sans-serif !important;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
  background: rgba(0,4,18,0.98) !important;
  border-right: 1px solid rgba(0,200,255,0.15) !important;
}
[data-testid="stSidebar"] > div { padding-top: 0 !important; }

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"] {
  background: rgba(0,200,255,0.03) !important;
  border: 1px solid rgba(0,200,255,0.2) !important;
  border-radius: 8px !important;
}
[data-testid="stFileUploader"]:hover {
  border-color: rgba(0,200,255,0.6) !important;
  box-shadow: 0 0 16px rgba(0,200,255,0.12) !important;
}
[data-testid="stFileUploader"] label {
  color: rgba(0,200,255,0.6) !important;
  font-family: 'Share Tech Mono', monospace !important;
  font-size: 0.72rem !important;
}
[data-testid="stFileUploader"] section { border: none !important; background: transparent !important; }
[data-testid="stFileUploader"] button  { border-color: rgba(0,200,255,0.4) !important; color: #00dcff !important; background: transparent !important; }

/* ── BUTTONS ── */
.stButton > button {
  background: transparent !important;
  border: 1px solid rgba(0,200,255,0.4) !important;
  color: #00dcff !important;
  border-radius: 6px !important;
  font-family: 'Share Tech Mono', monospace !important;
  font-size: 0.7rem !important;
  letter-spacing: 0.12em !important;
  width: 100% !important;
  padding: 10px !important;
  text-transform: uppercase !important;
  transition: all 0.25s !important;
}
.stButton > button:hover {
  background: rgba(0,200,255,0.12) !important;
  border-color: #00dcff !important;
  box-shadow: 0 0 22px rgba(0,200,255,0.3) !important;
}

/* ── PROGRESS / SPINNER ── */
.stProgress > div > div { background: linear-gradient(90deg,#00dcff,#00ff88) !important; }
.stProgress > div       { background: rgba(0,200,255,0.08) !important; border-radius: 0 !important; }
.stSuccess { background: rgba(0,255,136,0.06) !important; border: 1px solid rgba(0,255,136,0.2) !important; border-radius: 6px !important; color: #00ff88 !important; font-family: 'Share Tech Mono', monospace !important; font-size: 0.75rem !important; }
.stSpinner > div { border-top-color: #00dcff !important; }

.block-container { padding: 0 !important; max-width: 100% !important; }

/* ════════════════════════════════════════════
   WHATSAPP-STYLE CHAT
   ════════════════════════════════════════════ */

/* Wrapper row */
[data-testid="stChatMessage"] {
  background: transparent !important;
  border: none !important;
  padding: 3px 20px !important;
  margin-bottom: 2px !important;
}

/* ── BOT bubble — LEFT ── */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
  flex-direction: row !important;
  justify-content: flex-start !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) > div:last-child {
  background: rgba(0,18,38,0.92) !important;
  border: 1px solid rgba(0,200,255,0.18) !important;
  border-radius: 2px 18px 18px 18px !important;
  padding: 13px 17px !important;
  font-size: 0.88rem !important;
  line-height: 1.75 !important;
  color: #cce8ff !important;
  font-family: 'Inter', sans-serif !important;
  max-width: 65% !important;
  box-shadow: 0 3px 16px rgba(0,200,255,0.07) !important;
}

/* ── USER bubble — RIGHT ── */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
  flex-direction: row-reverse !important;
  justify-content: flex-start !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) > div:last-child {
  background: linear-gradient(135deg, rgba(0,100,200,0.4), rgba(60,0,180,0.45)) !important;
  border: 1px solid rgba(80,160,255,0.35) !important;
  border-radius: 18px 2px 18px 18px !important;
  padding: 13px 17px !important;
  font-size: 0.88rem !important;
  line-height: 1.75 !important;
  color: #e8f6ff !important;
  font-family: 'Inter', sans-serif !important;
  max-width: 65% !important;
  box-shadow: 0 3px 16px rgba(80,120,255,0.15) !important;
}

/* ── BOT avatar (cyan) ── */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid*="Avatar"],
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [class*="avatar"] {
  background: linear-gradient(135deg,#002a40,#005580) !important;
  border: 2px solid rgba(0,200,255,0.55) !important;
  box-shadow: 0 0 12px rgba(0,200,255,0.3) !important;
  border-radius: 50% !important;
}

/* ── USER avatar (purple) ── */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid*="Avatar"],
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [class*="avatar"] {
  background: linear-gradient(135deg,#1a0050,#4400cc) !important;
  border: 2px solid rgba(160,80,255,0.6) !important;
  box-shadow: 0 0 12px rgba(140,60,255,0.3) !important;
  border-radius: 50% !important;
}

/* ════════════════════════════════════════════
   CHAT INPUT — blue glow only, zero red
   ════════════════════════════════════════════ */
[data-testid="stChatInput"] {
  background: rgba(0,2,15,0.96) !important;
  border-top: 1px solid rgba(0,200,255,0.14) !important;
  padding: 12px 20px !important;
}
[data-testid="stChatInput"] textarea {
  background: rgba(0,200,255,0.04) !important;
  border: 1.5px solid rgba(0,200,255,0.28) !important;
  border-radius: 26px !important;
  color: #a8dcf8 !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 0.9rem !important;
  padding: 11px 20px !important;
  transition: border-color 0.25s, box-shadow 0.25s !important;
  outline: none !important;
  box-shadow: none !important;
}
[data-testid="stChatInput"] textarea:focus,
[data-testid="stChatInput"] textarea:focus-visible {
  outline: none !important;
  border-color: rgba(0,210,255,0.85) !important;
  box-shadow: 0 0 0 2px rgba(0,200,255,0.2),
              0 0 30px rgba(0,200,255,0.25) !important;
  background: rgba(0,200,255,0.06) !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: rgba(0,200,255,0.22) !important; }
[data-testid="stChatInput"] button {
  background: rgba(0,200,255,0.16) !important;
  border: 1.5px solid rgba(0,200,255,0.45) !important;
  color: #00dcff !important;
  border-radius: 50% !important;
  transition: all 0.2s !important;
}
[data-testid="stChatInput"] button:hover {
  background: rgba(0,200,255,0.32) !important;
  box-shadow: 0 0 20px rgba(0,200,255,0.45) !important;
}
/* Kill every other focus ring on the page */
*:focus        { outline: none !important; }
*:focus-visible { outline: 2px solid rgba(0,200,255,0.4) !important; outline-offset: 2px !important; }

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 3px; height: 3px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(0,200,255,0.25); border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: rgba(0,200,255,0.5); }

/* ── SIDEBAR UTILS ── */
.metric-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 6px; }
.metric-card { background: rgba(0,200,255,0.03); border: 1px solid rgba(0,200,255,0.1); border-radius: 6px; padding: 10px 8px; text-align: center; transition: all 0.3s; }
.metric-card:hover { border-color: rgba(0,200,255,0.35); background: rgba(0,200,255,0.06); }
.metric-val { font-family: 'Orbitron', monospace; font-size: 0.85rem; font-weight: 700; color: #00dcff; }
.metric-lbl { font-family: 'Share Tech Mono', monospace; font-size: 0.55rem; color: rgba(0,200,255,0.35); margin-top: 3px; letter-spacing: .06em; }
.sys-tips   { font-family: 'Share Tech Mono', monospace; font-size: 0.62rem; color: rgba(0,200,255,0.3); line-height: 2.1; letter-spacing: .04em; }
.s-divider  { border: none; border-top: 1px solid rgba(0,200,255,0.07); margin: 10px 0; }
.s-label    { font-family: 'Share Tech Mono', monospace; font-size: 0.58rem; letter-spacing: .18em; color: rgba(0,200,255,0.28); text-transform: uppercase; margin-bottom: 6px; display: block; }

/* ── CONTACT ── */
.contact-row  { display: flex; gap: 8px; margin-top: 6px; }
.contact-link {
  flex: 1; display: flex; align-items: center; justify-content: center; gap: 5px;
  padding: 7px 8px; border: 1px solid rgba(0,200,255,0.18); border-radius: 8px;
  text-decoration: none; font-family: 'Share Tech Mono', monospace; font-size: 0.58rem;
  color: rgba(0,200,255,0.55); letter-spacing: .04em;
  transition: all 0.2s; background: rgba(0,200,255,0.02);
}
.contact-link:hover {
  border-color: rgba(0,200,255,0.55); color: #00dcff;
  background: rgba(0,200,255,0.08); box-shadow: 0 0 14px rgba(0,200,255,0.12);
}

/* ── UPLOAD PANEL TOGGLE BUTTON ── */
.upload-toggle {
  width: 100%; padding: 10px 14px;
  background: rgba(0,200,255,0.06);
  border: 1.5px solid rgba(0,200,255,0.35);
  border-radius: 10px; cursor: pointer;
  font-family: 'Share Tech Mono', monospace; font-size: 0.68rem;
  color: rgba(0,220,255,0.75); letter-spacing: .12em;
  text-align: center; text-transform: uppercase;
  transition: all 0.25s; margin-bottom: 8px;
  display: flex; align-items: center; justify-content: center; gap: 8px;
}
.upload-toggle:hover {
  background: rgba(0,200,255,0.14);
  border-color: #00dcff;
  box-shadow: 0 0 20px rgba(0,200,255,0.22);
  color: #00dcff;
}
.upload-toggle.closed {
  border-color: rgba(0,200,255,0.6) !important;
  box-shadow: 0 0 16px rgba(0,200,255,0.18) !important;
  animation: pulse-btn 2s ease-in-out infinite;
}
@keyframes pulse-btn {
  0%,100% { box-shadow: 0 0 10px rgba(0,200,255,0.18); }
  50%      { box-shadow: 0 0 26px rgba(0,200,255,0.45); }
}

/* ── EMPTY STATE ── */
.empty-state { text-align: center; padding: 80px 32px; }
.empty-glyph { font-size: 3rem; color: rgba(0,200,255,0.12); margin-bottom: 20px; display: block; font-family: 'Orbitron', monospace; animation: pg 3s ease-in-out infinite; }
@keyframes pg { 0%,100%{opacity:.12;} 50%{opacity:.3;} }
.empty-state h2 { font-family: 'Orbitron', monospace; font-size: 0.95rem; font-weight: 400; color: rgba(0,200,255,0.2); letter-spacing: .12em; }
.empty-state p  { font-family: 'Share Tech Mono', monospace; font-size: 0.63rem; color: rgba(0,200,255,0.12); margin-top: 10px; letter-spacing: .08em; }

/* ── IMAGE FRAME ── */
.img-frame       { border: 1px solid rgba(0,200,255,0.2); border-radius: 8px; overflow: hidden; margin: 10px 0; background: #000; }
.img-frame-label { font-family: 'Share Tech Mono', monospace; font-size: 0.6rem; color: rgba(0,200,255,0.5); padding: 6px 12px; border-bottom: 1px solid rgba(0,200,255,0.1); letter-spacing: .1em; display: flex; justify-content: space-between; }

/* ── TIMESTAMPS ── */
.ts       { font-family: 'Share Tech Mono', monospace; font-size: 0.57rem; color: rgba(0,200,255,0.2); margin-top: 5px; display: block; }
.ts-right { text-align: right; }
.ts-left  { text-align: left; }
</style>
""", unsafe_allow_html=True)

# ── HERO ──────────────────────────────────────────────────────────────────────
HERO_HTML = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&display=swap');
*{box-sizing:border-box;margin:0;padding:0;}
#hero{position:relative;width:100%;height:230px;background:#020208;overflow:hidden;border-bottom:1px solid rgba(0,200,255,0.12);}
#mc{position:absolute;inset:0;width:100%;height:100%;}
#pc{position:absolute;inset:0;width:100%;height:100%;pointer-events:none;}
.gbg{position:absolute;inset:0;pointer-events:none;
  background-image:linear-gradient(rgba(0,200,255,0.03) 1px,transparent 1px),
                   linear-gradient(90deg,rgba(0,200,255,0.03) 1px,transparent 1px);
  background-size:44px 44px;}
.sl{position:absolute;inset:0;pointer-events:none;
  background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,0,0,0.05) 2px,rgba(0,0,0,0.05) 4px);}
.vig{position:absolute;inset:0;pointer-events:none;
  background:radial-gradient(ellipse at 50% 50%,transparent 20%,rgba(2,2,8,0.88) 100%);}
.ctl{position:absolute;top:0;left:0;width:36px;height:36px;pointer-events:none;
  border-top:1px solid rgba(0,200,255,0.45);border-left:1px solid rgba(0,200,255,0.45);}
.ctr{position:absolute;top:0;right:0;width:36px;height:36px;pointer-events:none;
  border-top:1px solid rgba(0,200,255,0.45);border-right:1px solid rgba(0,200,255,0.45);}
.cbl{position:absolute;bottom:0;left:0;width:36px;height:36px;pointer-events:none;
  border-bottom:1px solid rgba(0,200,255,0.45);border-left:1px solid rgba(0,200,255,0.45);}
.cbr{position:absolute;bottom:0;right:0;width:36px;height:36px;pointer-events:none;
  border-bottom:1px solid rgba(0,200,255,0.45);border-right:1px solid rgba(0,200,255,0.45);}
.status{position:absolute;top:13px;right:16px;display:flex;gap:9px;align-items:center;z-index:10;}
.dot{width:7px;height:7px;border-radius:50%;background:#00ff88;box-shadow:0 0 8px #00ff88;animation:bp 1.8s infinite;}
@keyframes bp{0%,100%{opacity:1;}50%{opacity:.15;}}
.stxt{font-family:'Share Tech Mono',monospace;font-size:9px;color:rgba(0,200,255,0.5);letter-spacing:.12em;}
.lclock{font-family:'Orbitron',monospace;font-size:10px;color:rgba(0,200,255,0.65);letter-spacing:.1em;}
.hc{position:absolute;bottom:18px;left:26px;z-index:10;}
.htag{font-family:'Share Tech Mono',monospace;font-size:9px;letter-spacing:.2em;
  color:rgba(0,220,255,0.65);border:1px solid rgba(0,220,255,0.25);
  padding:3px 10px;border-radius:2px;display:inline-block;margin-bottom:10px;position:relative;}
.htag::before{content:'';position:absolute;left:-1px;top:-1px;width:5px;height:5px;
  border-top:1px solid #00dcff;border-left:1px solid #00dcff;}
.htag::after{content:'';position:absolute;right:-1px;bottom:-1px;width:5px;height:5px;
  border-bottom:1px solid #00dcff;border-right:1px solid #00dcff;}
.hh1{font-family:'Orbitron',monospace;font-size:2.1rem;font-weight:900;line-height:1.05;
  color:#fff;text-shadow:0 0 50px rgba(0,200,255,0.3);}
.hh1 span{color:#00dcff;animation:glow 2.5s ease-in-out infinite alternate;}
@keyframes glow{from{text-shadow:0 0 8px rgba(0,200,255,0.3);}
  to{text-shadow:0 0 28px rgba(0,200,255,0.9),0 0 55px rgba(0,200,255,0.3);}}
.hsub{font-family:'Share Tech Mono',monospace;font-size:.58rem;
  color:rgba(0,200,255,0.3);letter-spacing:.12em;margin-top:6px;text-transform:uppercase;}
</style>
<div id="hero">
  <canvas id="mc"></canvas><canvas id="pc"></canvas>
  <div class="gbg"></div><div class="sl"></div><div class="vig"></div>
  <div class="ctl"></div><div class="ctr"></div><div class="cbl"></div><div class="cbr"></div>
  <div class="status">
    <div class="dot"></div>
    <span class="stxt">ONLINE</span>
    <span class="lclock" id="clk">--:-- --</span>
  </div>
  <div class="hc">
    <div class="htag">✦ PDF INTELLIGENCE · RAG v2.0</div>
    <div class="hh1">QUERY YOUR<br><span>DOCUMENT</span></div>
    <div class="hsub">// claude ai · faiss · sentence transformers</div>
  </div>
</div>
<script>
(function(){
  /* ── LOCAL 12-hour clock ── */
  function tick(){
    var n=new Date(),e=document.getElementById('clk');
    if(!e)return;
    var h=n.getHours(),m=n.getMinutes(),ap=h>=12?'PM':'AM';
    h=h%12||12;
    e.textContent=(h<10?'0'+h:h)+':'+(m<10?'0'+m:m)+' '+ap;
  }
  setInterval(tick,1000);tick();

  /* ── Matrix rain ── */
  var mc=document.getElementById('mc'),mctx=mc.getContext('2d'),
      hero=document.getElementById('hero'),
      W,H,cols=[],chars='01アイウエオカキクサシスタチツテトナニネノ',FS=13;
  function resM(){
    W=mc.width=hero.offsetWidth;H=mc.height=hero.offsetHeight;
    cols=[];var n=Math.floor(W/FS);
    for(var i=0;i<n;i++)cols.push({y:Math.random()*H,sp:Math.random()*1.4+0.3,br:Math.random()>.82,tr:Math.floor(Math.random()*8)+4});
  }
  function drM(){
    mctx.fillStyle='rgba(2,2,8,0.16)';mctx.fillRect(0,0,W,H);
    mctx.font=FS+'px monospace';
    cols.forEach(function(c,i){
      var ch=chars[Math.floor(Math.random()*chars.length)];
      mctx.fillStyle=c.br?'rgba(200,240,255,0.95)':'rgba(0,180,255,0.14)';
      mctx.fillText(ch,i*FS,c.y);c.y+=c.sp;
      if(c.y>H+FS){c.y=-FS*c.tr;c.br=Math.random()>.82;}
    });
  }

  /* ── Particle web ── */
  var pc=document.getElementById('pc'),pctx=pc.getContext('2d'),PW,PH,pts=[];
  function resP(){PW=pc.width=hero.offsetWidth;PH=pc.height=hero.offsetHeight;}
  function initP(){
    pts=[];
    for(var i=0;i<55;i++)
      pts.push({x:Math.random()*PW,y:Math.random()*PH,
                vx:(Math.random()-.5)*.5,vy:(Math.random()-.5)*.5,
                r:Math.random()*1.8+.4,a:Math.random()*.5+.15});
  }
  function drP(){
    pctx.clearRect(0,0,PW,PH);
    pts.forEach(function(p){
      p.x+=p.vx;p.y+=p.vy;
      if(p.x<0)p.x=PW;if(p.x>PW)p.x=0;
      if(p.y<0)p.y=PH;if(p.y>PH)p.y=0;
      pctx.beginPath();pctx.arc(p.x,p.y,p.r,0,Math.PI*2);
      pctx.fillStyle='rgba(0,220,255,'+p.a+')';pctx.fill();
    });
    for(var i=0;i<pts.length;i++)
      for(var j=i+1;j<pts.length;j++){
        var d=Math.hypot(pts[i].x-pts[j].x,pts[i].y-pts[j].y);
        if(d<90){
          pctx.beginPath();pctx.moveTo(pts[i].x,pts[i].y);pctx.lineTo(pts[j].x,pts[j].y);
          pctx.strokeStyle='rgba(0,180,255,'+(0.12*(1-d/90))+')';
          pctx.lineWidth=0.5;pctx.stroke();
        }
      }
  }
  function loop(){drM();drP();requestAnimationFrame(loop);}
  resM();resP();initP();loop();
  window.addEventListener('resize',function(){resM();resP();initP();});
})();
</script>
"""

# ── SUGGESTION BANNER ─────────────────────────────────────────────────────────
CHAT_BANNER_HTML = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@700&display=swap');
#cb{position:relative;overflow:hidden;background:rgba(0,2,15,0.72);
  border-bottom:1px solid rgba(0,200,255,0.08);}
#cbc{position:absolute;inset:0;width:100%;height:100%;pointer-events:none;}
.ci{position:relative;z-index:2;display:flex;align-items:center;gap:13px;padding:8px 20px;}
.cp{font-family:'Orbitron',monospace;font-size:0.63rem;font-weight:700;
  color:rgba(0,200,255,0.5);letter-spacing:.15em;white-space:nowrap;}
.ctags{display:flex;gap:7px;flex-wrap:wrap;}
.ct{font-family:'Share Tech Mono',monospace;font-size:0.59rem;
  color:rgba(0,200,255,0.38);border:1px solid rgba(0,200,255,0.13);
  padding:3px 10px;border-radius:20px;letter-spacing:.05em;
  animation:tp 3s ease-in-out infinite;}
.ct:nth-child(2){animation-delay:.4s;}.ct:nth-child(3){animation-delay:.8s;}
.ct:nth-child(4){animation-delay:1.2s;}.ct:nth-child(5){animation-delay:1.6s;}
@keyframes tp{
  0%,100%{opacity:.38;border-color:rgba(0,200,255,0.13);}
  50%{opacity:.85;border-color:rgba(0,200,255,0.45);color:rgba(0,220,255,0.75);}
}
</style>
<div id="cb">
  <canvas id="cbc" height="36"></canvas>
  <div class="ci">
    <div class="cp">TRY →</div>
    <div class="ctags">
      <div class="ct">summarize this</div>
      <div class="ct">show 1st image</div>
      <div class="ct">chapter 3?</div>
      <div class="ct">show all figures</div>
      <div class="ct">key findings?</div>
    </div>
  </div>
</div>
<script>
(function(){
  var cv=document.getElementById('cbc'),ctx=cv.getContext('2d'),W,H,t=0;
  function res(){W=cv.width=cv.parentElement.offsetWidth;H=cv.height=36;}
  function dr(){
    ctx.clearRect(0,0,W,H);
    for(var i=0;i<5;i++){
      var x=((t*0.35+i*140)%(W+60)+W+60)%(W+60)-60;
      ctx.fillStyle='rgba(0,200,255,'+(0.022+i%3*0.007)+')';
      ctx.fillRect(x,0,55,H);
    }
    t++;requestAnimationFrame(dr);
  }
  res();dr();window.addEventListener('resize',res);
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
    return any(t in q.lower() for t in
               ["show","display","see","view","image","figure","picture",
                "photo","diagram","chart","graph","illustration","img"])

def find_matching_images(question, image_store):
    if not image_store:
        return []
    q = question.lower()
    ordinals = {"1st":0,"first":0,"2nd":1,"second":1,"3rd":2,"third":2,
                "4th":3,"fourth":3,"5th":4,"fifth":4,"6th":5,"sixth":5}
    for word, idx in ordinals.items():
        if word in q and idx < len(image_store):
            return [image_store[idx]]
    for n in re.findall(r'\b(\d+)\b', q):
        idx = int(n) - 1
        if 0 <= idx < len(image_store):
            return [image_store[idx]]
    if "last" in q:  return [image_store[-1]]
    if "all"  in q:  return image_store
    if any(t in q for t in ["image","figure","picture","diagram","chart","graph"]):
        return [image_store[0]]
    return []

def fmt_time(seconds):
    return f"{seconds:.1f}s" if seconds < 60 else f"{int(seconds//60)}m {seconds%60:.0f}s"

def local_time():
    return datetime.now().strftime("%I:%M %p").lstrip("0")

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:

    # ── Header ──
    st.markdown("""
    <div style="padding:14px 14px 10px;border-bottom:1px solid rgba(0,200,255,0.1);margin-bottom:10px;">
      <div style="font-family:'Orbitron',monospace;font-size:0.88rem;font-weight:700;
        color:#00dcff;letter-spacing:.06em;">PDF BOT</div>
      <div style="font-family:'Share Tech Mono',monospace;font-size:0.54rem;
        color:rgba(0,200,255,0.22);letter-spacing:.14em;margin-top:2px;">INTELLIGENCE SYSTEM</div>
    </div>
    """, unsafe_allow_html=True)

    # ── BUILT BY — pinned near top ──
    st.markdown("""
    <span class="s-label">// Built by</span>
    <div style="font-family:'Share Tech Mono',monospace;font-size:0.7rem;
      color:rgba(0,200,255,0.7);margin-bottom:7px;letter-spacing:.02em;">
      Sai Jyothi Gayathri Adabala
    </div>
    <div class="contact-row">
      <a class="contact-link" href="mailto:asjyothig@gmail.com">
        <span>✉</span>&nbsp;email
      </a>
      <a class="contact-link" href="https://www.linkedin.com/in/sai-jyothi-gayathri-adabala-a41a9818b/" target="_blank">
        <span>in</span>&nbsp;linkedin
      </a>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="s-divider">', unsafe_allow_html=True)

    # ── UPLOAD PANEL with a very obvious glowing toggle ──
    panel_open = st.session_state.get("panel_open", True)

    # Always-visible toggle — pulses when closed so user knows it's there
    toggle_cls  = "upload-toggle" + ("" if panel_open else " closed")
    toggle_icon = "▲" if panel_open else "▼"
    toggle_text = "HIDE UPLOAD" if panel_open else "OPEN UPLOAD PANEL"
    st.markdown(f"""
    <div class="{toggle_cls}" id="utog" onclick=""
      style="margin-bottom:4px;">
      {toggle_icon}&nbsp;&nbsp;{toggle_text}&nbsp;&nbsp;{toggle_icon}
    </div>
    """, unsafe_allow_html=True)

    if st.button("▲ HIDE" if panel_open else "▼ OPEN UPLOAD", key="tog"):
        st.session_state.panel_open = not panel_open
        st.rerun()

    if panel_open:
        st.markdown('<span class="s-label" style="margin-top:6px;">// Drop your PDF</span>',
                    unsafe_allow_html=True)
        uploaded_file = st.file_uploader("", type="pdf", label_visibility="collapsed")

        if uploaded_file:
            st.markdown(f"""
            <div style="font-family:'Share Tech Mono',monospace;font-size:0.63rem;
              color:rgba(0,200,255,0.5);padding:6px 8px;
              border-left:2px solid rgba(0,200,255,0.3);margin-bottom:8px;
              word-break:break-all;">› {uploaded_file.name}</div>
            """, unsafe_allow_html=True)

            if st.button("⚡  PROCESS & INDEX"):
                t_start = time.time()
                with st.spinner("Indexing…"):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name
                    doc        = fitz.open(tmp_path)
                    full_text  = ""
                    prog       = st.progress(0)
                    total      = len(doc)
                    image_store      = []
                    img_global_idx   = 0

                    for pnum, page in enumerate(doc):
                        full_text += page.get_text()
                        for img in page.get_images():
                            try:
                                xref = img[0]
                                bi   = doc.extract_image(xref)
                                b64  = base64.b64encode(bi["image"]).decode()
                                mt   = "image/png" if bi["ext"] == "png" else "image/jpeg"
                                resp = anthropic_client.messages.create(
                                    model="claude-haiku-4-5-20251001", max_tokens=300,
                                    messages=[{"role":"user","content":[
                                        {"type":"image","source":{"type":"base64","media_type":mt,"data":b64}},
                                        {"type":"text","text":"Describe this image concisely. If it's a diagram/chart, explain what it shows."}
                                    ]}]
                                )
                                desc = resp.content[0].text
                                image_store.append({
                                    "b64":b64,"media_type":mt,
                                    "page":pnum+1,"index":img_global_idx,"description":desc
                                })
                                img_global_idx += 1
                                full_text += f"\n[Image {img_global_idx} on page {pnum+1}]: {desc}\n"
                            except Exception:
                                continue
                        prog.progress((pnum+1)/total)

                    doc.close()
                    os.unlink(tmp_path)
                    t_process = time.time() - t_start

                    t_embed = time.time()
                    chunks  = extract_chunks(full_text)
                    embs    = np.array(model.encode(chunks)).astype('float32')
                    index   = faiss.IndexFlatL2(embs.shape[1])
                    index.add(embs)
                    t_embed_done = time.time() - t_embed

                    st.session_state.update({
                        "index":index,"chunks":chunks,
                        "messages":[],"chat_display":[],
                        "chunk_count":len(chunks),
                        "image_store":image_store,"image_count":len(image_store),
                        "page_count":total,
                        "process_time":t_process,"embed_time":t_embed_done,
                    })
                st.success(f"✓ {len(chunks)} chunks · {len(image_store)} imgs · {fmt_time(t_process)}")
    else:
        uploaded_file = None

    st.markdown('<hr class="s-divider">', unsafe_allow_html=True)

    # ── Metrics ──
    chunk_count = st.session_state.get("chunk_count", 0)
    img_count   = st.session_state.get("image_count", 0)
    proc_time   = st.session_state.get("process_time", None)
    embed_time  = st.session_state.get("embed_time",   None)
    proc_disp   = fmt_time(proc_time)  if proc_time  else "—"
    embed_disp  = fmt_time(embed_time) if embed_time else "—"

    st.markdown(f"""
    <span class="s-label">// Last Run</span>
    <div class="metric-grid">
      <div class="metric-card">
        <div class="metric-val">{chunk_count if chunk_count else '—'}</div>
        <div class="metric-lbl">Chunks</div>
      </div>
      <div class="metric-card">
        <div class="metric-val">{img_count if img_count else '—'}</div>
        <div class="metric-lbl">Images</div>
      </div>
      <div class="metric-card">
        <div class="metric-val">{proc_disp}</div>
        <div class="metric-lbl">Process</div>
      </div>
      <div class="metric-card">
        <div class="metric-val">{embed_disp}</div>
        <div class="metric-lbl">Index</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="s-divider">', unsafe_allow_html=True)

    st.markdown("""
    <span class="s-label">// Query Tips</span>
    <div class="sys-tips">
      › ask specific questions<br>
      › "show me the 1st image"<br>
      › "show all figures"<br>
      › reference page numbers<br>
      › request summaries
    </div>
    <div style="margin-top:14px;font-family:'Share Tech Mono',monospace;font-size:7px;
      color:rgba(0,200,255,0.1);letter-spacing:.08em;text-align:center;
      border-top:1px solid rgba(0,200,255,0.05);padding-top:8px;">
      CLAUDE · FAISS · SENTENCE-TRANSFORMERS
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
st.components.v1.html(HERO_HTML, height=232, scrolling=False)

if "index" not in st.session_state:
    st.markdown("""
    <div class="empty-state">
      <span class="empty-glyph">⬡</span>
      <h2>UPLOAD A DOCUMENT TO BEGIN</h2>
      <p>// open the sidebar → drop a PDF → ask anything</p>
    </div>
    """, unsafe_allow_html=True)
else:
    if "chat_display" not in st.session_state:
        st.session_state.chat_display = []
    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.components.v1.html(CHAT_BANNER_HTML, height=38, scrolling=False)
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    # ── Render chat history ──
    for entry in st.session_state.chat_display:
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"])
            ts   = entry.get("time", "")
            side = "ts-right" if entry["role"] == "user" else "ts-left"
            tick = "✓" if entry["role"] == "user" else "⬡"
            st.markdown(f'<span class="ts {side}">{tick} {ts}</span>',
                        unsafe_allow_html=True)
            if entry.get("images"):
                for img_data in entry["images"]:
                    st.markdown(
                        f'<div class="img-frame"><div class="img-frame-label">'
                        f'<span>// IMAGE {img_data["index"]+1}</span>'
                        f'<span>PAGE {img_data["page"]}</span></div></div>',
                        unsafe_allow_html=True
                    )
                    st.image(base64.b64decode(img_data["b64"]), use_column_width=False)

    question = st.chat_input("Message PDF Bot…")

    if question:
        now = local_time()

        with st.chat_message("user"):
            st.markdown(question)
            st.markdown(f'<span class="ts ts-right">✓ {now}</span>', unsafe_allow_html=True)
        st.session_state.chat_display.append(
            {"role":"user","content":question,"images":[],"time":now}
        )

        images_to_show = []
        if is_image_request(question):
            images_to_show = find_matching_images(
                question, st.session_state.get("image_store", [])
            )

        q_emb = np.array([model.encode(question)]).astype('float32')
        _, idxs = st.session_state.index.search(q_emb, 3)
        context = "\n\n".join(st.session_state.chunks[i] for i in idxs[0])

        if images_to_show:
            st.session_state.messages.append({"role":"user","content":
                f"Context:\n{context}\n\nQuestion: {question}\n\n"
                "Note: You ARE showing the user the image(s). Describe in 1-2 sentences."
            })
            system_prompt = ("You are a document assistant. The user asked to see an image "
                             "and you ARE displaying it. Describe what the image shows in "
                             "1-2 sentences. Never say you cannot show images.")
            max_tok = 250
        else:
            st.session_state.messages.append({"role":"user","content":
                f"Context:\n{context}\n\nQuestion: {question}"
            })
            system_prompt = ("You are an intelligent document assistant. Answer clearly and "
                             "concisely. Use bullet points or numbered lists when listing "
                             "multiple items. Bold key terms using **term**. If the exact "
                             "term isn't in context, look for related concepts.")
            max_tok = 1024

        with st.spinner(""):
            response = anthropic_client.messages.create(
                model="claude-haiku-4-5-20251001", max_tokens=max_tok,
                system=system_prompt,
                messages=st.session_state.messages
            )

        reply      = response.content[0].text
        reply_time = local_time()
        st.session_state.messages.append({"role":"assistant","content":reply})
        st.session_state.chat_display.append(
            {"role":"assistant","content":reply,"images":images_to_show,"time":reply_time}
        )

        with st.chat_message("assistant"):
            st.markdown(reply)
            st.markdown(f'<span class="ts ts-left">⬡ {reply_time}</span>',
                        unsafe_allow_html=True)
            if images_to_show:
                for img_data in images_to_show:
                    st.markdown(
                        f'<div class="img-frame"><div class="img-frame-label">'
                        f'<span>// IMAGE {img_data["index"]+1}</span>'
                        f'<span>PAGE {img_data["page"]}</span></div></div>',
                        unsafe_allow_html=True
                    )
                    st.image(base64.b64decode(img_data["b64"]), use_column_width=False)