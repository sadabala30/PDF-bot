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

st.set_page_config(page_title="PDF BOT", page_icon="⬡", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&family=Inter:wght@300;400;500;600&display=swap');
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
#MainMenu,footer,header,[data-testid="stToolbar"],.stDeployButton,
[data-testid="stNotificationActionButton"],[data-testid="toastContainer"],
.stToast,[class*="toast"],[class*="Toast"],
[data-baseweb="notification"],[data-baseweb="toast"]{display:none!important;visibility:hidden!important;pointer-events:none!important;}

html,body,.stApp,[data-testid="stAppViewContainer"]{background:#020208!important;color:#c8e0f0!important;font-family:'Inter',sans-serif!important;}
[data-testid="stSidebar"]{background:rgba(0,4,18,0.98)!important;border-right:1px solid rgba(0,200,255,0.12)!important;min-width:260px!important;}
[data-testid="stSidebar"]>div{padding-top:0!important;}

/* ── SIDEBAR TOGGLE BUTTON — always visible, fixed to left edge ── */
[data-testid="stSidebarCollapseButton"],
[data-testid="collapsedControl"]{
  display:flex!important;visibility:visible!important;opacity:1!important;
  position:fixed!important;left:0!important;top:50%!important;
  transform:translateY(-50%)!important;z-index:99999!important;
  width:28px!important;height:64px!important;
  background:rgba(0,10,30,0.95)!important;
  border:1.5px solid rgba(0,200,255,0.6)!important;
  border-left:none!important;
  border-radius:0 12px 12px 0!important;
  box-shadow:0 0 20px rgba(0,200,255,0.5),4px 0 16px rgba(0,200,255,0.2)!important;
  align-items:center!important;justify-content:center!important;
  cursor:pointer!important;
  animation:sbpulse 2.2s ease-in-out infinite!important;
  transition:background 0.2s!important;
}
[data-testid="stSidebarCollapseButton"]:hover,
[data-testid="collapsedControl"]:hover{
  background:rgba(0,200,255,0.25)!important;
  box-shadow:0 0 36px rgba(0,200,255,0.9),4px 0 24px rgba(0,200,255,0.5)!important;
}
[data-testid="stSidebarCollapseButton"] svg,
[data-testid="collapsedControl"] svg{
  color:#00dcff!important;
  filter:drop-shadow(0 0 5px #00dcff)!important;
  width:16px!important;height:16px!important;
}
@keyframes sbpulse{
  0%,100%{box-shadow:0 0 14px rgba(0,200,255,0.35),4px 0 10px rgba(0,200,255,0.15);}
  50%{box-shadow:0 0 32px rgba(0,200,255,0.8),4px 0 22px rgba(0,200,255,0.45);}
}

[data-testid="stFileUploader"]{background:rgba(0,10,30,0.6)!important;border:2px dashed rgba(0,200,255,0.45)!important;border-radius:12px!important;transition:all 0.3s!important;box-shadow:0 0 18px rgba(0,200,255,0.12),inset 0 0 18px rgba(0,200,255,0.04)!important;}
[data-testid="stFileUploader"]:hover{border-color:rgba(0,200,255,0.85)!important;box-shadow:0 0 32px rgba(0,200,255,0.28),inset 0 0 24px rgba(0,200,255,0.08)!important;}
[data-testid="stFileUploader"] label{color:rgba(0,200,255,0.7)!important;font-family:'Share Tech Mono',monospace!important;font-size:0.72rem!important;}
[data-testid="stFileUploader"] section{border:none!important;background:transparent!important;}
[data-testid="stFileUploader"] button{border:1px solid rgba(0,200,255,0.4)!important;color:#00dcff!important;background:rgba(0,200,255,0.06)!important;border-radius:6px!important;font-family:'Share Tech Mono',monospace!important;font-size:0.65rem!important;}
[data-testid="stFileUploader"] svg{color:rgba(0,200,255,0.4)!important;}

.stButton>button{background:transparent!important;border:1px solid rgba(0,200,255,0.4)!important;color:#00dcff!important;border-radius:6px!important;font-family:'Share Tech Mono',monospace!important;font-size:0.7rem!important;letter-spacing:0.1em!important;width:100%!important;padding:9px!important;text-transform:uppercase!important;transition:all 0.25s!important;animation:pulsebtn 2.2s ease-in-out infinite!important;}
.stButton>button:hover{background:rgba(0,200,255,0.12)!important;border-color:#00dcff!important;box-shadow:0 0 22px rgba(0,200,255,0.3)!important;}
@keyframes pulsebtn{0%,100%{box-shadow:0 0 8px rgba(0,200,255,0.12)}50%{box-shadow:0 0 20px rgba(0,200,255,0.35)}}

.stProgress>div>div{background:linear-gradient(90deg,#00dcff,#00ff88)!important;}
.stProgress>div{background:rgba(0,200,255,0.08)!important;border-radius:0!important;}
.stSuccess{background:rgba(0,255,136,0.06)!important;border:1px solid rgba(0,255,136,0.2)!important;border-radius:6px!important;color:#00ff88!important;font-family:'Share Tech Mono',monospace!important;font-size:0.75rem!important;}
.block-container{padding:0!important;max-width:100%!important;}

[data-testid="stChatMessage"]{background:transparent!important;border:none!important;padding:3px 20px!important;margin-bottom:4px!important;}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"])>div:last-child{background:rgba(0,18,38,0.92)!important;border:1px solid rgba(0,200,255,0.18)!important;border-radius:2px 18px 18px 18px!important;padding:13px 17px!important;font-size:0.88rem!important;line-height:1.75!important;color:#cce8ff!important;max-width:65%!important;}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"])>div:last-child{background:linear-gradient(135deg,rgba(0,100,200,0.4),rgba(60,0,180,0.45))!important;border:1px solid rgba(80,160,255,0.35)!important;border-radius:18px 2px 18px 18px!important;padding:13px 17px!important;font-size:0.88rem!important;line-height:1.75!important;color:#e8f6ff!important;max-width:65%!important;}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]){flex-direction:row-reverse!important;}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid*="Avatar"]{background:linear-gradient(135deg,#002a40,#005580)!important;border:2px solid rgba(0,200,255,0.55)!important;box-shadow:0 0 12px rgba(0,200,255,0.3)!important;border-radius:50%!important;}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid*="Avatar"]{background:linear-gradient(135deg,#1a0050,#4400cc)!important;border:2px solid rgba(160,80,255,0.6)!important;box-shadow:0 0 12px rgba(140,60,255,0.3)!important;border-radius:50%!important;}

/* ══════════════════════════════════════════════
   CHAT INPUT — KILL RED BORDER COMPLETELY
   The red comes from [data-baseweb="base-input"]
   which Streamlit wraps around the textarea.
   We nuke every possible source.
══════════════════════════════════════════════ */
[data-testid="stChatInput"]{
  background:rgba(0,2,15,0.96)!important;
  border-top:1px solid rgba(0,200,255,0.14)!important;
  border:none!important;
  padding:12px 20px!important;
  outline:none!important;
  box-shadow:none!important;
}
/* The outer div Streamlit injects */
[data-testid="stChatInput"]>div{
  border:none!important;
  outline:none!important;
  box-shadow:none!important;
  background:transparent!important;
}
/* BaseWeb wrapper — this is what goes red */
[data-testid="stChatInput"] [data-baseweb="base-input"],
[data-testid="stChatInput"] [data-baseweb="textarea"],
[data-testid="stChatInput"] [class*="InputContainer"],
[data-testid="stChatInput"] [class*="BaseInput"]{
  border:none!important;
  outline:none!important;
  box-shadow:none!important;
  background:transparent!important;
}
/* The actual textarea pill */
[data-testid="stChatInput"] textarea{
  background:rgba(0,200,255,0.04)!important;
  border:1.5px solid rgba(0,200,255,0.3)!important;
  border-radius:26px!important;
  color:#a8dcf8!important;
  font-family:'Inter',sans-serif!important;
  font-size:0.9rem!important;
  padding:11px 20px!important;
  outline:none!important;
  box-shadow:none!important;
  caret-color:#00dcff!important;
}
[data-testid="stChatInput"] textarea:focus,
[data-testid="stChatInput"] textarea:focus-visible,
[data-testid="stChatInput"] textarea:focus-within{
  border:1.5px solid rgba(0,200,255,0.8)!important;
  outline:none!important;
  box-shadow:0 0 0 2px rgba(0,200,255,0.15),0 0 20px rgba(0,200,255,0.2)!important;
  background:rgba(0,200,255,0.06)!important;
}
[data-testid="stChatInput"] textarea::placeholder{color:rgba(0,200,255,0.22)!important;}
[data-testid="stChatInput"] button{background:rgba(0,200,255,0.16)!important;border:1.5px solid rgba(0,200,255,0.45)!important;color:#00dcff!important;border-radius:50%!important;}
[data-testid="stChatInput"] button:hover{background:rgba(0,200,255,0.3)!important;box-shadow:0 0 16px rgba(0,200,255,0.4)!important;}

/* Nuclear option: kill ALL red/orange/default outlines sitewide */
*:focus,*:focus-visible,*:focus-within{
  outline:none!important;
}
/* BaseWeb uses internal state classes for the red ring — kill them all */
[data-baseweb="base-input"]:focus-within,
[data-baseweb="base-input"]:focus,
[data-baseweb="textarea"]:focus-within{
  border-color:transparent!important;
  box-shadow:none!important;
  outline:none!important;
}

::-webkit-scrollbar{width:3px;height:3px;}::-webkit-scrollbar-thumb{background:rgba(0,200,255,0.25);border-radius:2px;}

.metric-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:6px;}
.metric-card{background:rgba(0,200,255,0.03);border:1px solid rgba(0,200,255,0.1);border-radius:6px;padding:10px 8px;text-align:center;transition:all 0.3s;}
.metric-card:hover{border-color:rgba(0,200,255,0.35);background:rgba(0,200,255,0.06);}
.metric-val{font-family:'Orbitron',monospace;font-size:0.85rem;font-weight:700;color:#00dcff;}
.metric-lbl{font-family:'Share Tech Mono',monospace;font-size:0.55rem;color:rgba(0,200,255,0.35);margin-top:3px;letter-spacing:.06em;}
.sys-tips{font-family:'Share Tech Mono',monospace;font-size:0.62rem;color:rgba(0,200,255,0.3);line-height:2.1;letter-spacing:.04em;}
.s-divider{border:none;border-top:1px solid rgba(0,200,255,0.07);margin:10px 0;}
.s-label{font-family:'Share Tech Mono',monospace;font-size:0.58rem;letter-spacing:.18em;color:rgba(0,200,255,0.28);text-transform:uppercase;margin-bottom:6px;display:block;}
.ts{font-family:'Share Tech Mono',monospace;font-size:0.57rem;color:rgba(0,200,255,0.2);margin-top:5px;display:block;}
.ts-right{text-align:right;}.ts-left{text-align:left;}
.img-frame{border:1px solid rgba(0,200,255,0.2);border-radius:8px;overflow:hidden;margin:10px 0;}
.img-frame-label{font-family:'Share Tech Mono',monospace;font-size:0.6rem;color:rgba(0,200,255,0.5);padding:6px 12px;border-bottom:1px solid rgba(0,200,255,0.1);letter-spacing:.1em;display:flex;justify-content:space-between;}
.img-slide{animation:slideIn 0.5s cubic-bezier(0.34,1.56,0.64,1) forwards;opacity:0;transform:translateY(20px);}
@keyframes slideIn{to{opacity:1;transform:translateY(0)}}
.claude-thinking{font-family:'Share Tech Mono',monospace;font-size:0.62rem;color:rgba(0,200,255,0.5);padding:10px 14px;background:rgba(0,18,38,0.6);border:1px solid rgba(0,200,255,0.12);border-radius:8px;margin:4px 20px;line-height:2;}
.think-line{animation:thinkfade 1.2s ease-in-out infinite;}
.think-line:nth-child(2){animation-delay:.2s}.think-line:nth-child(3){animation-delay:.4s}
@keyframes thinkfade{0%,100%{opacity:.3}50%{opacity:1}}
.think-bar{margin-top:8px;height:2px;background:rgba(0,200,255,0.08);border-radius:2px;overflow:hidden;}
.think-bar-fill{height:2px;background:linear-gradient(90deg,#00dcff,#00ff88);border-radius:2px;animation:thinkprog 2s ease-in-out infinite;}
@keyframes thinkprog{0%{width:0%}70%{width:90%}100%{width:100%}}
.rc-line{animation:rcfade 0.4s ease forwards;opacity:0;font-family:'Share Tech Mono',monospace;font-size:0.55rem;color:rgba(0,200,255,0.4);line-height:1.9;}
.rc-line:nth-child(2){animation-delay:.15s}.rc-line:nth-child(3){animation-delay:.3s}
@keyframes rcfade{to{opacity:1}}
</style>
""", unsafe_allow_html=True)

HERO_HTML = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&display=swap');
*{box-sizing:border-box;margin:0;padding:0}body{background:#020208}
#hero{position:relative;width:100%;height:200px;background:#020208;overflow:hidden;border-bottom:1px solid rgba(0,200,255,0.12)}
#hero-canvas,#particle-canvas{position:absolute;inset:0;width:100%;height:100%}
.hero-grid{position:absolute;inset:0;background-image:linear-gradient(rgba(0,200,255,0.03) 1px,transparent 1px),linear-gradient(90deg,rgba(0,200,255,0.03) 1px,transparent 1px);background-size:44px 44px}
.hero-vig{position:absolute;inset:0;background:radial-gradient(ellipse at 50% 50%,transparent 20%,rgba(2,2,8,.85) 100%)}
.hero-corners{position:absolute;inset:0;pointer-events:none}
.hero-corners span{position:absolute;width:30px;height:30px}
.tl{top:0;left:0;border-top:1px solid rgba(0,200,255,.45);border-left:1px solid rgba(0,200,255,.45)}
.tr{top:0;right:0;border-top:1px solid rgba(0,200,255,.45);border-right:1px solid rgba(0,200,255,.45)}
.bl{bottom:0;left:0;border-bottom:1px solid rgba(0,200,255,.45);border-left:1px solid rgba(0,200,255,.45)}
.br{bottom:0;right:0;border-bottom:1px solid rgba(0,200,255,.45);border-right:1px solid rgba(0,200,255,.45)}
.hero-status{position:absolute;top:12px;right:14px;display:flex;gap:8px;align-items:center;z-index:5}
.hero-dot{width:7px;height:7px;border-radius:50%;background:#00ff88;box-shadow:0 0 8px #00ff88;animation:blink 1.8s infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.15}}
.hero-stxt{font-family:'Share Tech Mono',monospace;font-size:.5rem;color:rgba(0,200,255,.5);letter-spacing:.1em}
.hero-clock{font-family:'Orbitron',monospace;font-size:.55rem;color:rgba(0,200,255,.65);letter-spacing:.1em}
.hero-content{position:absolute;bottom:16px;left:22px;z-index:5}
.hero-tag{font-family:'Share Tech Mono',monospace;font-size:.52rem;letter-spacing:.18em;color:rgba(0,220,255,.6);border:1px solid rgba(0,220,255,.22);padding:2px 8px;border-radius:2px;display:inline-block;margin-bottom:8px;position:relative}
.hero-tag::before{content:'';position:absolute;left:-1px;top:-1px;width:4px;height:4px;border-top:1px solid #00dcff;border-left:1px solid #00dcff}
.hero-tag::after{content:'';position:absolute;right:-1px;bottom:-1px;width:4px;height:4px;border-bottom:1px solid #00dcff;border-right:1px solid #00dcff}
.hero-h1{font-family:'Orbitron',monospace;font-size:1.75rem;font-weight:900;line-height:1.05;color:#fff}
.hero-h1 span{color:#00dcff;animation:glowtxt 2.5s ease-in-out infinite alternate}
@keyframes glowtxt{from{text-shadow:0 0 6px rgba(0,200,255,.3)}to{text-shadow:0 0 24px rgba(0,200,255,.9),0 0 48px rgba(0,200,255,.28)}}
.hero-sub{font-family:'Share Tech Mono',monospace;font-size:.52rem;color:rgba(0,200,255,.28);letter-spacing:.1em;margin-top:5px}
</style>
<div id="hero">
  <canvas id="hero-canvas"></canvas><canvas id="particle-canvas"></canvas>
  <div class="hero-grid"></div><div class="hero-vig"></div>
  <div class="hero-corners"><span class="tl"></span><span class="tr"></span><span class="bl"></span><span class="br"></span></div>
  <div class="hero-status"><div class="hero-dot"></div><span class="hero-stxt">ONLINE</span><span class="hero-clock" id="clk">--:-- --</span></div>
  <div class="hero-content">
    <div class="hero-tag">✦ PDF INTELLIGENCE · RAG v2.0</div>
    <div class="hero-h1">QUERY YOUR<br><span>DOCUMENT</span></div>
    <div class="hero-sub">// claude ai · faiss · sentence transformers</div>
  </div>
</div>
<script>
(function(){
  function tick(){var n=new Date(),e=document.getElementById('clk');if(!e)return;var h=n.getHours(),m=n.getMinutes(),ap=h>=12?'PM':'AM';h=h%12||12;e.textContent=(h<10?'0'+h:h)+':'+(m<10?'0'+m:m)+' '+ap;}
  setInterval(tick,1000);tick();
  var hc=document.getElementById('hero-canvas'),hctx=hc.getContext('2d'),pc=document.getElementById('particle-canvas'),pctx=pc.getContext('2d'),hero=document.getElementById('hero'),W,H,cols=[],pts=[],chars='01アイウエオカキクサシスタチツテトナニネノ',FS=13;
  function resz(){W=hc.width=pc.width=hero.offsetWidth;H=hc.height=pc.height=hero.offsetHeight;cols=[];var n=Math.floor(W/FS);for(var i=0;i<n;i++)cols.push({y:Math.random()*H,sp:Math.random()*1.4+0.3,br:Math.random()>.82,tr:Math.floor(Math.random()*8)+4});pts=[];for(var i=0;i<55;i++)pts.push({x:Math.random()*W,y:Math.random()*H,vx:(Math.random()-.5)*.5,vy:(Math.random()-.5)*.5,r:Math.random()*1.8+.4,a:Math.random()*.5+.15});}
  function loop(){hctx.fillStyle='rgba(2,2,8,0.16)';hctx.fillRect(0,0,W,H);hctx.font=FS+'px monospace';cols.forEach(function(c,i){var ch=chars[Math.floor(Math.random()*chars.length)];hctx.fillStyle=c.br?'rgba(200,240,255,0.95)':'rgba(0,180,255,0.14)';hctx.fillText(ch,i*FS,c.y);c.y+=c.sp;if(c.y>H+FS){c.y=-FS*c.tr;c.br=Math.random()>.82;}});pctx.clearRect(0,0,W,H);pts.forEach(function(p){p.x+=p.vx;p.y+=p.vy;if(p.x<0)p.x=W;if(p.x>W)p.x=0;if(p.y<0)p.y=H;if(p.y>H)p.y=0;pctx.beginPath();pctx.arc(p.x,p.y,p.r,0,Math.PI*2);pctx.fillStyle='rgba(0,220,255,'+p.a+')';pctx.fill();});for(var i=0;i<pts.length;i++)for(var j=i+1;j<pts.length;j++){var d=Math.hypot(pts[i].x-pts[j].x,pts[i].y-pts[j].y);if(d<90){pctx.beginPath();pctx.moveTo(pts[i].x,pts[i].y);pctx.lineTo(pts[j].x,pts[j].y);pctx.strokeStyle='rgba(0,180,255,'+(0.12*(1-d/90))+')';pctx.lineWidth=0.5;pctx.stroke();}}requestAnimationFrame(loop);}
  resz();loop();window.addEventListener('resize',resz);
})();
</script>
"""

RAG_PIPELINE_HTML = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700&family=Share+Tech+Mono&display=swap');
*{box-sizing:border-box;margin:0;padding:0}body{background:#020208}
#pipeline{background:rgba(0,3,14,0.92);border-bottom:1px solid rgba(0,200,255,0.1);padding:10px 16px 6px;}
.pl-label{font-family:'Share Tech Mono',monospace;font-size:0.48rem;letter-spacing:.2em;color:rgba(0,200,255,0.22);text-transform:uppercase;margin-bottom:8px;}
.pl-nodes{display:flex;align-items:center;overflow-x:auto;padding-bottom:2px;}
.pnode{display:flex;flex-direction:column;align-items:center;gap:3px;min-width:62px;flex-shrink:0;}
.pnode .picon{width:28px;height:28px;border-radius:6px;border:1px solid rgba(0,200,255,.18);display:flex;align-items:center;justify-content:center;font-size:10px;background:rgba(0,200,255,.04);transition:all .3s;position:relative;overflow:hidden;}
.pnode .plabel{font-family:'Share Tech Mono',monospace;font-size:.38rem;color:rgba(0,200,255,.25);letter-spacing:.04em;text-align:center;text-transform:uppercase;line-height:1.3;}
.pnode.active .picon{border-color:rgba(0,200,255,.9);background:rgba(0,200,255,.14);box-shadow:0 0 14px rgba(0,200,255,.4);}
.pnode.active .plabel{color:rgba(0,200,255,.85);}
.pnode.done .picon{border-color:rgba(0,255,136,.5);background:rgba(0,255,136,.06);box-shadow:0 0 8px rgba(0,255,136,.2);}
.pnode.done .plabel{color:rgba(0,255,136,.5);}
.parrow{color:rgba(0,200,255,.15);font-size:.6rem;padding:0 1px;margin-bottom:10px;flex-shrink:0;transition:all .4s;}
.parrow.lit{color:#00dcff;text-shadow:0 0 8px rgba(0,200,255,.7);}
.scan{position:absolute;left:0;right:0;height:30%;background:linear-gradient(transparent,rgba(0,200,255,0.3),transparent);pointer-events:none;}
.pnode.active .picon .scan{animation:scanbox 1s linear infinite;}
@keyframes scanbox{0%{top:-100%}100%{top:200%}}
</style>
<div id="pipeline">
  <div class="pl-label">// rag pipeline · retrieve → augment → generate</div>
  <div class="pl-nodes">
    <div class="pnode" id="pn0"><div class="picon">📄<div class="scan"></div></div><div class="plabel">PDF<br>upload</div></div>
    <div class="parrow" id="pa0">──▶</div>
    <div class="pnode" id="pn1"><div class="picon">T<div class="scan"></div></div><div class="plabel">text<br>extract</div></div>
    <div class="parrow" id="pa1">──▶</div>
    <div class="pnode" id="pn2"><div class="picon">🖼<div class="scan"></div></div><div class="plabel">image<br>ocr</div></div>
    <div class="parrow" id="pa2">──▶</div>
    <div class="pnode" id="pn3"><div class="picon">⬡<div class="scan"></div></div><div class="plabel">chunk<br>ing</div></div>
    <div class="parrow" id="pa3">──▶</div>
    <div class="pnode" id="pn4"><div class="picon">●<div class="scan"></div></div><div class="plabel">embed<br>dings</div></div>
    <div class="parrow" id="pa4">──▶</div>
    <div class="pnode" id="pn5"><div class="picon">◈<div class="scan"></div></div><div class="plabel">faiss<br>index</div></div>
    <div class="parrow" id="pa5">──▶</div>
    <div class="pnode" id="pn6"><div class="picon">◎<div class="scan"></div></div><div class="plabel">vector<br>search</div></div>
    <div class="parrow" id="pa6">──▶</div>
    <div class="pnode" id="pn7"><div class="picon">AI<div class="scan"></div></div><div class="plabel">claude<br>ai</div></div>
    <div class="parrow" id="pa7">──▶</div>
    <div class="pnode" id="pn8"><div class="picon">💬<div class="scan"></div></div><div class="plabel">answer</div></div>
  </div>
</div>
<script>
(function(){
  var total=9,step=0;
  function setNode(i,cls){var n=document.getElementById('pn'+i);if(n)n.className='pnode '+cls;}
  function setArrow(i,lit){var a=document.getElementById('pa'+i);if(a)a.className='parrow'+(lit?' lit':'');}
  function runStep(){if(step>0){setNode(step-1,'done');if(step>1)setArrow(step-2,'lit');}if(step>=total){setTimeout(function(){for(var i=0;i<total;i++){setNode(i,'');setArrow(i,'');}step=0;setTimeout(runStep,600);},2000);return;}setNode(step,'active');if(step>0)setArrow(step-1,'lit');step++;setTimeout(runStep,520);}
  setTimeout(runStep,400);
})();
</script>
"""

EMPTY_STATE_HTML = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Share+Tech+Mono&display=swap');
*{box-sizing:border-box;margin:0;padding:0}body{background:#020208}
#es{padding:50px 32px;text-align:center;background:#020208;}
.es-glyph{font-family:'Orbitron',monospace;font-size:2.4rem;color:rgba(0,200,255,.1);margin-bottom:16px;animation:empg 3s ease-in-out infinite;}
@keyframes empg{0%,100%{opacity:.1}50%{opacity:.28}}
.es-title{font-family:'Orbitron',monospace;font-size:0.88rem;font-weight:400;color:rgba(0,200,255,.2);letter-spacing:.14em;margin-bottom:6px;}
.es-sub{font-family:'Share Tech Mono',monospace;font-size:0.58rem;color:rgba(0,200,255,.12);letter-spacing:.08em;margin-bottom:28px;}
#anim-canvas{display:block;margin:0 auto;}
.es-flow{display:flex;align-items:center;justify-content:center;gap:8px;margin-top:20px;font-family:'Share Tech Mono',monospace;font-size:0.54rem;color:rgba(0,200,255,.15);letter-spacing:.1em;}
.efa{animation:flowarrow 2s ease-in-out infinite;}.efa:nth-child(2){animation-delay:.3s}.efa:nth-child(4){animation-delay:.6s}.efa:nth-child(6){animation-delay:.9s}
@keyframes flowarrow{0%,100%{opacity:.12}50%{opacity:.7}}
</style>
<div id="es">
  <div class="es-glyph">⬡</div>
  <div class="es-title">UPLOAD A DOCUMENT TO BEGIN</div>
  <div class="es-sub">// open sidebar → drop pdf → ask anything</div>
  <canvas id="anim-canvas" width="420" height="120"></canvas>
  <div class="es-flow"><span>PDF</span><span class="efa">→</span><span>TEXT+IMAGES</span><span class="efa">→</span><span>EMBEDDINGS</span><span class="efa">→</span><span>AI</span><span class="efa">→</span><span>ANSWERS</span></div>
</div>
<script>
(function(){
  var cv=document.getElementById('anim-canvas'),ctx=cv.getContext('2d'),W=420,H=120,t=0;
  var dots=[];for(var i=0;i<28;i++)dots.push({x:Math.random()*W,y:40+Math.random()*40,vx:Math.random()*1.5+0.4,a:Math.random()*.5+.1,r:Math.random()*2+.5,col:Math.random()>.5?'0,200,255':'0,255,136'});
  function draw(){ctx.clearRect(0,0,W,H);t++;function box(x,y,w,h,label,col){ctx.strokeStyle='rgba('+col+',0.35)';ctx.lineWidth=0.8;ctx.strokeRect(x,y,w,h);ctx.fillStyle='rgba('+col+',0.04)';ctx.fillRect(x,y,w,h);ctx.fillStyle='rgba('+col+',0.6)';ctx.font='bold 8px monospace';ctx.textAlign='center';ctx.fillText(label,x+w/2,y+h/2+3);}
  box(8,40,52,38,'PDF','200,60,60');ctx.strokeStyle='rgba(0,200,255,0.2)';ctx.lineWidth=0.8;ctx.beginPath();ctx.moveTo(60,59);ctx.lineTo(95,45);ctx.stroke();ctx.beginPath();ctx.moveTo(60,59);ctx.lineTo(95,75);ctx.stroke();box(95,22,60,40,'TEXT','0,200,255');box(95,66,60,40,'IMAGES','100,80,255');ctx.beginPath();ctx.moveTo(155,59);ctx.lineTo(190,59);ctx.stroke();ctx.fillStyle='rgba(0,200,255,0.4)';ctx.font='8px monospace';ctx.textAlign='left';ctx.fillText('▶',185,62);ctx.strokeStyle='rgba(0,200,255,0.15)';ctx.lineWidth=0.5;ctx.strokeRect(195,30,70,58);ctx.fillStyle='rgba(0,200,255,0.04)';ctx.fillRect(195,30,70,58);ctx.fillStyle='rgba(0,200,255,0.4)';ctx.font='7px monospace';ctx.textAlign='center';ctx.fillText('VECTORS',230,42);for(var r=0;r<3;r++)for(var c=0;c<6;c++){var px=202+c*10,py=50+r*12,g=Math.sin(t*.05+r*2+c*.5)*.5+.5;ctx.beginPath();ctx.arc(px,py,2,0,Math.PI*2);ctx.fillStyle='rgba(0,220,255,'+(0.3+g*.5)+')';ctx.fill();}ctx.beginPath();ctx.moveTo(265,59);ctx.lineTo(300,59);ctx.stroke();ctx.fillStyle='rgba(0,200,255,0.4)';ctx.textAlign='left';ctx.fillText('▶',295,62);ctx.strokeStyle='rgba(0,255,136,0.4)';ctx.lineWidth=1;ctx.strokeRect(305,35,50,48);ctx.fillStyle='rgba(0,255,136,0.04)';ctx.fillRect(305,35,50,48);ctx.fillStyle='rgba(0,255,136,0.7)';ctx.font='bold 10px monospace';ctx.textAlign='center';ctx.fillText('AI',330,63);var pulse=Math.sin(t*.06)*.5+.5;ctx.beginPath();ctx.arc(330,59,28+4*pulse,0,Math.PI*2);ctx.strokeStyle='rgba(0,255,136,'+(0.06+pulse*.08)+')';ctx.lineWidth=0.5;ctx.stroke();dots.forEach(function(d){ctx.beginPath();ctx.arc(d.x,d.y,d.r,0,Math.PI*2);ctx.fillStyle='rgba('+d.col+','+d.a+')';ctx.fill();d.x+=d.vx;if(d.x>W+5){d.x=-5;d.y=40+Math.random()*40;}});requestAnimationFrame(draw);}
  draw();
})();
</script>
"""

CHAT_BANNER_HTML = """
<style>
body{margin:0;background:rgba(0,2,14,0.85);}
.sug-bar{display:flex;gap:7px;align-items:center;padding:7px 20px;border-bottom:1px solid rgba(0,200,255,0.07);overflow-x:auto;flex-wrap:nowrap;}
.sug-prefix{font-family:'Share Tech Mono',monospace;font-size:0.5rem;color:rgba(0,200,255,0.3);letter-spacing:.14em;white-space:nowrap;}
.sug-tag{font-family:'Share Tech Mono',monospace;font-size:0.55rem;color:rgba(0,200,255,0.5);border:1px solid rgba(0,200,255,0.2);padding:4px 12px;border-radius:20px;white-space:nowrap;cursor:pointer;transition:all 0.2s;background:rgba(0,200,255,0.03);}
.sug-tag:hover{color:#00dcff;border-color:rgba(0,200,255,0.7);background:rgba(0,200,255,0.1);box-shadow:0 0 10px rgba(0,200,255,0.2);}
.sug-tag:active{transform:scale(0.96);}
</style>
<div class="sug-bar">
  <div class="sug-prefix">TRY →</div>
  <div class="sug-tag" onclick="inject(this)">summarize this</div>
  <div class="sug-tag" onclick="inject(this)">show 1st image</div>
  <div class="sug-tag" onclick="inject(this)">what is chapter 3 about?</div>
  <div class="sug-tag" onclick="inject(this)">show all figures</div>
  <div class="sug-tag" onclick="inject(this)">key findings?</div>
  <div class="sug-tag" onclick="inject(this)">explain the methodology</div>
</div>
<script>
function inject(el){
  var text=el.textContent.trim();
  var pdoc=window.parent.document;
  var ta=pdoc.querySelector('[data-testid="stChatInput"] textarea');
  if(!ta){ta=pdoc.querySelector('textarea');}
  if(!ta){return;}
  var nativeSetter=Object.getOwnPropertyDescriptor(window.parent.HTMLTextAreaElement.prototype,'value').set;
  nativeSetter.call(ta,text);
  ta.dispatchEvent(new window.parent.Event('input',{bubbles:true}));
  ta.dispatchEvent(new window.parent.Event('change',{bubbles:true}));
  ta.focus();
  ta.setSelectionRange(text.length,text.length);
}
</script>
"""

THINKING_HTML = """
<div class="claude-thinking">
  <div class="think-line">> retrieving relevant chunks…</div>
  <div class="think-line">> ranking by cosine similarity…</div>
  <div class="think-line">> generating answer…</div>
  <div class="think-bar"><div class="think-bar-fill"></div></div>
</div>
"""

def retrieval_html(chunk_indices):
    lines = "".join(f'<div class="rc-line">› Chunk {i} matched ✓</div>' for i in chunk_indices)
    return f'<div style="padding:6px 20px;">{lines}</div>'

def fmt_time(s): return f"{s:.1f}s" if s < 60 else f"{int(s//60)}m {s%60:.0f}s"
def local_time(): return datetime.now().strftime("%I:%M %p").lstrip("0")

def is_image_request(q):
    return any(t in q.lower() for t in ["show","display","see","view","image","figure","picture","photo","diagram","chart","graph","illustration"])

def find_matching_images(question, image_store):
    if not image_store: return []
    q = question.lower()
    ordinals = {"1st":0,"first":0,"2nd":1,"second":1,"3rd":2,"third":2,"4th":3,"fourth":3,"5th":4,"fifth":4}
    for word,idx in ordinals.items():
        if word in q and idx < len(image_store): return [image_store[idx]]
    for n in re.findall(r'\b(\d+)\b', q):
        idx = int(n)-1
        if 0 <= idx < len(image_store): return [image_store[idx]]
    if "last" in q: return [image_store[-1]]
    if "all" in q: return image_store[:5]
    return [image_store[0]]

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
    <div style="padding:14px 14px 10px;border-bottom:1px solid rgba(0,200,255,0.1);margin-bottom:8px;">
      <div style="font-family:'Orbitron',monospace;font-size:0.88rem;font-weight:700;color:#00dcff;letter-spacing:.06em;">PDF BOT</div>
      <div style="font-family:'Share Tech Mono',monospace;font-size:0.5rem;color:rgba(0,200,255,0.22);letter-spacing:.14em;margin-top:2px;">INTELLIGENCE SYSTEM</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<span class="s-label">// Document</span>', unsafe_allow_html=True)
    if "show_uploader" not in st.session_state:
        st.session_state.show_uploader = True

    if st.button("📂  UPLOAD / CHANGE PDF", key="toggle_upload"):
        st.session_state.show_uploader = not st.session_state.show_uploader
        st.rerun()

    if st.session_state.show_uploader:
        uploaded_file = st.file_uploader("", type="pdf", label_visibility="collapsed")
        if uploaded_file:
            st.markdown(f"""
            <div style="font-family:'Share Tech Mono',monospace;font-size:0.6rem;color:rgba(0,200,255,0.5);
              padding:5px 8px;border-left:2px solid rgba(0,200,255,0.3);margin-bottom:8px;word-break:break-all;">
              › {uploaded_file.name}</div>""", unsafe_allow_html=True)

            if st.button("⚡  PROCESS & INDEX"):
                t_start = time.time()
                proc_ph = st.empty()
                STEPS = ["extracting text","analyzing images","chunking","embedding","building index","ready ✓"]

                def show_proc(step, pct, chunks_n=0, images_n=0):
                    rows = ""
                    for i, lbl in enumerate(STEPS):
                        col = "rgba(0,255,136,0.85)" if i < step else ("rgba(0,200,255,0.95)" if i == step else "rgba(0,200,255,0.18)")
                        ind = "✓" if i < step else ("●" if i == step else "○")
                        rows += f'<div style="font-family:Share Tech Mono,monospace;font-size:0.62rem;color:{col};padding:3px 0;letter-spacing:.05em;">{ind} &gt; {lbl}</div>'
                    stats = ""
                    if chunks_n > 0:
                        stats = (f'<div style="display:flex;gap:24px;margin-top:10px;padding-top:10px;border-top:1px solid rgba(0,200,255,0.1);">'
                                 f'<div style="text-align:center;"><div style="font-family:Orbitron,monospace;font-size:1.1rem;font-weight:700;color:#00dcff;">{chunks_n}</div>'
                                 f'<div style="font-family:Share Tech Mono,monospace;font-size:0.48rem;color:rgba(0,200,255,0.35);letter-spacing:.1em;">chunks</div></div>'
                                 f'<div style="text-align:center;"><div style="font-family:Orbitron,monospace;font-size:1.1rem;font-weight:700;color:#00ff88;">{images_n}</div>'
                                 f'<div style="font-family:Share Tech Mono,monospace;font-size:0.48rem;color:rgba(0,200,255,0.35);letter-spacing:.1em;">images</div></div></div>')
                    proc_ph.markdown(
                        f'<div style="background:rgba(0,6,22,0.97);border:1px solid rgba(0,200,255,0.22);border-radius:14px;padding:16px 18px;">'
                        f'<div style="font-family:Orbitron,monospace;font-size:0.68rem;color:#00dcff;letter-spacing:.08em;margin-bottom:10px;">⬡ PROCESSING</div>'
                        f'<div style="background:rgba(0,200,255,0.06);border-radius:2px;height:2px;margin-bottom:10px;overflow:hidden;">'
                        f'<div style="height:2px;background:linear-gradient(90deg,#00dcff,#00ff88);width:{pct}%;transition:width 0.5s;"></div></div>'
                        f'{rows}{stats}</div>', unsafe_allow_html=True)

                show_proc(0, 5)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read()); tmp_path = tmp.name

                doc = fitz.open(tmp_path)
                full_text, total, image_store, img_idx = "", len(doc), [], 0

                show_proc(1, 15)
                for pnum, page in enumerate(doc):
                    full_text += page.get_text()
                    for img in page.get_images():
                        try:
                            xref = img[0]; bi = doc.extract_image(xref)
                            b64 = base64.b64encode(bi["image"]).decode()
                            mt = "image/png" if bi["ext"] == "png" else "image/jpeg"
                            resp = anthropic_client.messages.create(
                                model="claude-haiku-4-5-20251001", max_tokens=300,
                                messages=[{"role":"user","content":[
                                    {"type":"image","source":{"type":"base64","media_type":mt,"data":b64}},
                                    {"type":"text","text":"Describe this image concisely. If diagram/chart, explain what it shows."}
                                ]}]
                            )
                            desc = resp.content[0].text
                            image_store.append({"b64":b64,"media_type":mt,"page":pnum+1,"index":img_idx,"description":desc})
                            img_idx += 1
                            full_text += f"\n[Image {img_idx} on page {pnum+1}]: {desc}\n"
                        except: continue
                    show_proc(1, 15+int((pnum+1)/total*30), images_n=img_idx)

                doc.close(); os.unlink(tmp_path)
                show_proc(2, 50, images_n=img_idx)
                chunks = extract_chunks(full_text)
                show_proc(3, 68, chunks_n=len(chunks), images_n=img_idx)
                embs = np.array(model.encode(chunks)).astype('float32')
                show_proc(4, 88, chunks_n=len(chunks), images_n=img_idx)
                index = faiss.IndexFlatL2(embs.shape[1]); index.add(embs)
                show_proc(5, 100, chunks_n=len(chunks), images_n=img_idx)
                time.sleep(0.4); proc_ph.empty()
                t_done = time.time() - t_start

                st.session_state.update({
                    "index":index,"chunks":chunks,"messages":[],"chat_display":[],
                    "chunk_count":len(chunks),"image_store":image_store,"image_count":len(image_store),
                    "page_count":total,"process_time":t_done,"show_uploader":False,
                })
                st.rerun()
    else:
        uploaded_file = None
        if "chunk_count" in st.session_state:
            st.markdown("""
            <div style="font-family:'Share Tech Mono',monospace;font-size:0.6rem;color:rgba(0,255,136,0.6);
              padding:5px 8px;border-left:2px solid rgba(0,255,136,0.3);margin-bottom:4px;">
              ✓ Document indexed &amp; ready</div>""", unsafe_allow_html=True)

    st.markdown('<hr class="s-divider">', unsafe_allow_html=True)

    cc = st.session_state.get("chunk_count",0)
    ic = st.session_state.get("image_count",0)
    pc = st.session_state.get("page_count",0)
    pt = st.session_state.get("process_time",None)
    st.markdown('<span class="s-label">// Last Run</span>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="metric-grid">
      <div class="metric-card"><div class="metric-val">{pc or '—'}</div><div class="metric-lbl">Pages</div></div>
      <div class="metric-card"><div class="metric-val">{cc or '—'}</div><div class="metric-lbl">Chunks</div></div>
      <div class="metric-card"><div class="metric-val">{ic or '—'}</div><div class="metric-lbl">Images</div></div>
      <div class="metric-card"><div class="metric-val">{cc or '—'}</div><div class="metric-lbl">Vectors</div></div>
      <div class="metric-card" style="grid-column:span 2"><div class="metric-val">{fmt_time(pt) if pt else '—'}</div><div class="metric-lbl">Process Time</div></div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<hr class="s-divider">', unsafe_allow_html=True)

    st.markdown("""
    <span class="s-label">// Tips</span>
    <div class="sys-tips">
      › ask specific questions<br>
      › "show me the 1st image"<br>
      › "show all figures"<br>
      › reference page numbers<br>
      › request summaries
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="s-divider">', unsafe_allow_html=True)

    st.markdown('<span class="s-label">// Built by</span>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'Share Tech Mono',monospace;font-size:0.66rem;
      color:rgba(0,200,255,0.6);margin-bottom:10px;">
      Sai Jyothi Gayathri Adabala
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    if col1.button("✉ email", key="btn_email"):
        st.session_state["contact_show"] = "email" if st.session_state.get("contact_show") != "email" else None
    if col2.button("in linkedin", key="btn_li"):
        st.session_state["contact_show"] = "linkedin" if st.session_state.get("contact_show") != "linkedin" else None

    cs = st.session_state.get("contact_show")
    if cs == "email":
        st.markdown("""
        <div style="margin-top:6px;background:rgba(0,10,28,0.97);border:1px solid rgba(0,200,255,0.22);
          border-radius:8px;padding:10px 14px;">
          <div style="font-family:'Orbitron',monospace;font-size:0.58rem;color:#00dcff;letter-spacing:.08em;margin-bottom:6px;">✦ Sai Jyothi Gayathri Adabala</div>
          <div style="font-family:'Share Tech Mono',monospace;font-size:0.62rem;color:rgba(0,200,255,0.75);letter-spacing:.04em;">
            ✉ &nbsp;asjyothig@gmail.com</div>
        </div>""", unsafe_allow_html=True)
    elif cs == "linkedin":
        st.markdown("""
        <div style="margin-top:6px;background:rgba(0,10,28,0.97);border:1px solid rgba(0,200,255,0.22);
          border-radius:8px;padding:10px 14px;">
          <div style="font-family:'Orbitron',monospace;font-size:0.58rem;color:#00dcff;letter-spacing:.08em;margin-bottom:6px;">✦ Sai Jyothi Gayathri Adabala</div>
          <a href="https://www.linkedin.com/in/sai-jyothi-gayathri-adabala-a41a9818b/" target="_blank"
             style="font-family:'Share Tech Mono',monospace;font-size:0.6rem;color:#00dcff;text-decoration:none;
               display:block;border:1px solid rgba(0,200,255,0.35);border-radius:6px;padding:7px 10px;
               text-align:center;background:rgba(0,200,255,0.06);">
            → Open LinkedIn Profile ↗
          </a>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:12px;font-family:'Share Tech Mono',monospace;font-size:7px;
      color:rgba(0,200,255,0.1);letter-spacing:.08em;text-align:center;
      border-top:1px solid rgba(0,200,255,0.05);padding-top:8px;">
      CLAUDE · FAISS · SENTENCE-TRANSFORMERS
    </div>""", unsafe_allow_html=True)

# ── MAIN ──────────────────────────────────────────────────────────────────────
# Floating sidebar toggle button — injected via JS into the parent page.
# Works on desktop AND mobile. Persists even when sidebar is fully collapsed.
st.components.v1.html("""
<script>
(function(){
  var STYLE_ID = 'sb-float-style';
  var BTN_ID   = 'sb-float-btn';

  function getDoc(){ return window.parent ? window.parent.document : document; }

  function ensureStyle(doc){
    if(doc.getElementById(STYLE_ID)) return;
    var s = doc.createElement('style');
    s.id = STYLE_ID;
    s.textContent = [
      '@keyframes sbglow{0%,100%{box-shadow:0 0 14px rgba(0,200,255,.4),3px 0 10px rgba(0,200,255,.15)}',
      '50%{box-shadow:0 0 30px rgba(0,200,255,.85),3px 0 20px rgba(0,200,255,.4)}}',
      '#'+BTN_ID+'{',
        'position:fixed!important;',
        'left:0!important;top:50%!important;',
        'transform:translateY(-50%)!important;',
        'z-index:2147483647!important;',
        'width:26px!important;height:60px!important;',
        'background:rgba(0,8,24,0.95)!important;',
        'border:1.5px solid rgba(0,200,255,0.65)!important;',
        'border-left:none!important;',
        'border-radius:0 10px 10px 0!important;',
        'color:#00dcff!important;',
        'font-size:14px!important;',
        'cursor:pointer!important;',
        'display:flex!important;align-items:center!important;justify-content:center!important;',
        'animation:sbglow 2.2s ease-in-out infinite!important;',
        'transition:background 0.2s!important;',
        'outline:none!important;',
        'padding:0!important;',
      '}',
      '#'+BTN_ID+':hover{background:rgba(0,200,255,0.28)!important;}'
    ].join('');
    doc.head.appendChild(s);
  }

  function ensureButton(doc){
    if(doc.getElementById(BTN_ID)) return;
    var btn = doc.createElement('button');
    btn.id = BTN_ID;
    btn.title = 'Toggle Sidebar';
    btn.innerHTML = '&#8942;'; // ⋮ vertical dots
    btn.setAttribute('aria-label','Toggle sidebar');

    btn.addEventListener('click', function(){
      // Strategy 1: click Streamlit's native collapse/expand button
      var targets = [
        '[data-testid="stSidebarCollapseButton"] button',
        '[data-testid="stSidebarCollapseButton"]',
        '[data-testid="collapsedControl"] button',
        '[data-testid="collapsedControl"]',
      ];
      for(var i=0;i<targets.length;i++){
        var el = doc.querySelector(targets[i]);
        if(el){ el.click(); return; }
      }
      // Strategy 2: toggle sidebar display directly
      var sb = doc.querySelector('[data-testid="stSidebar"]');
      if(sb){
        sb.style.display = (sb.style.display === 'none') ? '' : 'none';
      }
    });

    doc.body.appendChild(btn);
  }

  function init(){
    var doc = getDoc();
    ensureStyle(doc);
    ensureButton(doc);
  }

  // Run immediately + after Streamlit rehydrates
  init();
  setTimeout(init, 600);
  setTimeout(init, 1800);

  // Watch for Streamlit re-renders that might remove the button
  var doc = getDoc();
  var observer = new MutationObserver(function(mutations){
    if(!doc.getElementById(BTN_ID)) ensureButton(doc);
  });
  setTimeout(function(){
    observer.observe(doc.body, {childList: true, subtree: false});
  }, 800);
})();
</script>
""", height=0, scrolling=False)

st.components.v1.html(HERO_HTML, height=202, scrolling=False)

if "index" not in st.session_state:
    st.components.v1.html(EMPTY_STATE_HTML, height=270, scrolling=False)
else:
    if "chat_display" not in st.session_state: st.session_state.chat_display = []
    if "messages" not in st.session_state: st.session_state.messages = []

    st.components.v1.html(RAG_PIPELINE_HTML, height=82, scrolling=False)
    st.components.v1.html(CHAT_BANNER_HTML, height=44, scrolling=False)

    for entry in st.session_state.chat_display:
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"])
            side = "ts-right" if entry["role"]=="user" else "ts-left"
            tick = "✓" if entry["role"]=="user" else "⬡"
            st.markdown(f'<span class="ts {side}">{tick} {entry.get("time","")}</span>', unsafe_allow_html=True)
            for img_data in entry.get("images",[]):
                st.markdown(f'<div class="img-frame img-slide"><div class="img-frame-label"><span>// IMAGE {img_data["index"]+1}</span><span>PAGE {img_data["page"]}</span></div></div>', unsafe_allow_html=True)
                st.image(base64.b64decode(img_data["b64"]))

    question = st.chat_input("Message PDF Bot…")

    if question:
        now = local_time()
        with st.chat_message("user"):
            st.markdown(question)
            st.markdown(f'<span class="ts ts-right">✓ {now}</span>', unsafe_allow_html=True)
        st.session_state.chat_display.append({"role":"user","content":question,"images":[],"time":now})

        images_to_show = find_matching_images(question, st.session_state.get("image_store",[])) if is_image_request(question) else []

        think_ph = st.empty()
        think_ph.markdown(THINKING_HTML, unsafe_allow_html=True)

        q_emb = np.array([model.encode(question)]).astype('float32')
        _, idxs = st.session_state.index.search(q_emb, 3)
        context = "\n\n".join(st.session_state.chunks[i] for i in idxs[0])

        think_ph.markdown(retrieval_html([int(i)+1 for i in idxs[0]]), unsafe_allow_html=True)
        time.sleep(0.3)
        think_ph.empty()

        if images_to_show:
            st.session_state.messages.append({"role":"user","content":f"Context:\n{context}\n\nQuestion: {question}\n\nNote: You are showing the user the image(s). Describe in 1-2 sentences."})
            system_prompt = "You are a document assistant. Briefly describe what the image shows in 1-2 sentences."
            max_tok = 250
        else:
            st.session_state.messages.append({"role":"user","content":f"Context:\n{context}\n\nQuestion: {question}"})
            system_prompt = "You are an intelligent document assistant. Answer clearly. Use bullet points for lists. Bold key terms using **term**. If not in context, look for related concepts."
            max_tok = 1024

        response = anthropic_client.messages.create(
            model="claude-haiku-4-5-20251001", max_tokens=max_tok,
            system=system_prompt, messages=st.session_state.messages
        )
        reply = response.content[0].text
        reply_time = local_time()
        st.session_state.messages.append({"role":"assistant","content":reply})
        st.session_state.chat_display.append({"role":"assistant","content":reply,"images":images_to_show,"time":reply_time})

        with st.chat_message("assistant"):
            st.markdown(reply)
            st.markdown(f'<span class="ts ts-left">⬡ {reply_time}</span>', unsafe_allow_html=True)
            for img_data in images_to_show:
                st.markdown(f'<div class="img-frame img-slide"><div class="img-frame-label"><span>// IMAGE {img_data["index"]+1}</span><span>PAGE {img_data["page"]}</span></div></div>', unsafe_allow_html=True)
                st.image(base64.b64decode(img_data["b64"]))