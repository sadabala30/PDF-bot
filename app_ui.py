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

# ── GLOBAL CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&family=Inter:wght@300;400;500;600&display=swap');
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
#MainMenu,footer,header,[data-testid="stToolbar"],.stDeployButton{display:none!important;visibility:hidden!important;}
html,body,.stApp,[data-testid="stAppViewContainer"]{background:#020208!important;color:#c8e0f0!important;font-family:'Inter',sans-serif!important;}

[data-testid="stSidebar"]{background:rgba(0,4,18,0.98)!important;border-right:1px solid rgba(0,200,255,0.15)!important;}
[data-testid="stSidebar"]>div{padding-top:0!important;}

[data-testid="stFileUploader"]{background:rgba(0,200,255,0.03)!important;border:1px solid rgba(0,200,255,0.2)!important;border-radius:8px!important;}
[data-testid="stFileUploader"]:hover{border-color:rgba(0,200,255,0.6)!important;box-shadow:0 0 16px rgba(0,200,255,0.12)!important;}
[data-testid="stFileUploader"] label{color:rgba(0,200,255,0.6)!important;font-family:'Share Tech Mono',monospace!important;font-size:0.72rem!important;}
[data-testid="stFileUploader"] section{border:none!important;background:transparent!important;}
[data-testid="stFileUploader"] button{border-color:rgba(0,200,255,0.4)!important;color:#00dcff!important;background:transparent!important;}

.stButton>button{background:transparent!important;border:1px solid rgba(0,200,255,0.4)!important;color:#00dcff!important;border-radius:6px!important;font-family:'Share Tech Mono',monospace!important;font-size:0.7rem!important;letter-spacing:0.1em!important;width:100%!important;padding:9px!important;text-transform:uppercase!important;transition:all 0.25s!important;}
.stButton>button:hover{background:rgba(0,200,255,0.12)!important;border-color:#00dcff!important;box-shadow:0 0 22px rgba(0,200,255,0.3)!important;}

.stProgress>div>div{background:linear-gradient(90deg,#00dcff,#00ff88)!important;}
.stProgress>div{background:rgba(0,200,255,0.08)!important;border-radius:0!important;}
.stSuccess{background:rgba(0,255,136,0.06)!important;border:1px solid rgba(0,255,136,0.2)!important;border-radius:6px!important;color:#00ff88!important;font-family:'Share Tech Mono',monospace!important;font-size:0.75rem!important;}
.stSpinner>div{border-top-color:#00dcff!important;}
.block-container{padding:0!important;max-width:100%!important;}

[data-testid="stChatMessage"]{background:transparent!important;border:none!important;padding:3px 20px!important;margin-bottom:4px!important;}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"])>div:last-child{background:rgba(0,18,38,0.92)!important;border:1px solid rgba(0,200,255,0.18)!important;border-radius:2px 18px 18px 18px!important;padding:13px 17px!important;font-size:0.88rem!important;line-height:1.75!important;color:#cce8ff!important;max-width:68%!important;box-shadow:0 3px 16px rgba(0,200,255,0.07)!important;}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"])>div:last-child{background:linear-gradient(135deg,rgba(0,100,200,0.4),rgba(60,0,180,0.45))!important;border:1px solid rgba(80,160,255,0.35)!important;border-radius:18px 2px 18px 18px!important;padding:13px 17px!important;font-size:0.88rem!important;line-height:1.75!important;color:#e8f6ff!important;max-width:68%!important;}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]){flex-direction:row-reverse!important;}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid*="Avatar"]{background:linear-gradient(135deg,#002a40,#005580)!important;border:2px solid rgba(0,200,255,0.55)!important;box-shadow:0 0 12px rgba(0,200,255,0.3)!important;border-radius:50%!important;}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid*="Avatar"]{background:linear-gradient(135deg,#1a0050,#4400cc)!important;border:2px solid rgba(160,80,255,0.6)!important;box-shadow:0 0 12px rgba(140,60,255,0.3)!important;border-radius:50%!important;}

[data-testid="stChatInput"]{background:rgba(0,2,15,0.96)!important;border-top:1px solid rgba(0,200,255,0.14)!important;padding:12px 20px!important;}
[data-testid="stChatInput"] textarea{background:rgba(0,200,255,0.04)!important;border:1.5px solid rgba(0,200,255,0.28)!important;border-radius:26px!important;color:#a8dcf8!important;font-family:'Inter',sans-serif!important;font-size:0.9rem!important;padding:11px 20px!important;}
[data-testid="stChatInput"] textarea:focus{border-color:rgba(0,210,255,0.85)!important;box-shadow:0 0 0 2px rgba(0,200,255,0.2),0 0 30px rgba(0,200,255,0.25)!important;}
[data-testid="stChatInput"] textarea::placeholder{color:rgba(0,200,255,0.22)!important;}
[data-testid="stChatInput"] button{background:rgba(0,200,255,0.16)!important;border:1.5px solid rgba(0,200,255,0.45)!important;color:#00dcff!important;border-radius:50%!important;}
[data-testid="stChatInput"] button:hover{background:rgba(0,200,255,0.32)!important;box-shadow:0 0 20px rgba(0,200,255,0.45)!important;}

::-webkit-scrollbar{width:3px;height:3px;}
::-webkit-scrollbar-track{background:transparent;}
::-webkit-scrollbar-thumb{background:rgba(0,200,255,0.25);border-radius:2px;}

.metric-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:6px;}
.metric-card{background:rgba(0,200,255,0.03);border:1px solid rgba(0,200,255,0.1);border-radius:6px;padding:10px 8px;text-align:center;transition:all 0.3s;}
.metric-card:hover{border-color:rgba(0,200,255,0.35);background:rgba(0,200,255,0.06);}
.metric-val{font-family:'Orbitron',monospace;font-size:0.85rem;font-weight:700;color:#00dcff;}
.metric-lbl{font-family:'Share Tech Mono',monospace;font-size:0.55rem;color:rgba(0,200,255,0.35);margin-top:3px;letter-spacing:.06em;}
.sys-tips{font-family:'Share Tech Mono',monospace;font-size:0.62rem;color:rgba(0,200,255,0.3);line-height:2.1;letter-spacing:.04em;}
.s-divider{border:none;border-top:1px solid rgba(0,200,255,0.07);margin:10px 0;}
.s-label{font-family:'Share Tech Mono',monospace;font-size:0.58rem;letter-spacing:.18em;color:rgba(0,200,255,0.28);text-transform:uppercase;margin-bottom:6px;display:block;}

.empty-state{text-align:center;padding:60px 32px;}
.empty-glyph{font-size:2.8rem;color:rgba(0,200,255,0.12);margin-bottom:20px;display:block;font-family:'Orbitron',monospace;animation:pg 3s ease-in-out infinite;}
@keyframes pg{0%,100%{opacity:.12}50%{opacity:.3}}
.empty-state h2{font-family:'Orbitron',monospace;font-size:0.95rem;font-weight:400;color:rgba(0,200,255,0.2);letter-spacing:.12em;}
.empty-state p{font-family:'Share Tech Mono',monospace;font-size:0.63rem;color:rgba(0,200,255,0.12);margin-top:10px;letter-spacing:.08em;}

.img-frame{border:1px solid rgba(0,200,255,0.2);border-radius:8px;overflow:hidden;margin:10px 0;background:#000;}
.img-frame-label{font-family:'Share Tech Mono',monospace;font-size:0.6rem;color:rgba(0,200,255,0.5);padding:6px 12px;border-bottom:1px solid rgba(0,200,255,0.1);letter-spacing:.1em;display:flex;justify-content:space-between;}
.ts{font-family:'Share Tech Mono',monospace;font-size:0.57rem;color:rgba(0,200,255,0.2);margin-top:5px;display:block;}
.ts-right{text-align:right;}.ts-left{text-align:left;}

.sug-bar{display:flex;gap:7px;align-items:center;padding:7px 20px;border-bottom:1px solid rgba(0,200,255,0.07);background:rgba(0,2,14,0.8);overflow-x:auto;}
.sug-prefix{font-family:'Share Tech Mono',monospace;font-size:0.5rem;color:rgba(0,200,255,0.3);letter-spacing:.14em;white-space:nowrap;}
.sug-tag{font-family:'Share Tech Mono',monospace;font-size:0.55rem;color:rgba(0,200,255,0.38);border:1px solid rgba(0,200,255,0.14);padding:3px 10px;border-radius:20px;white-space:nowrap;cursor:pointer;transition:all 0.2s;animation:taganim 3s ease-in-out infinite;}
.sug-tag:hover{color:#00dcff;border-color:rgba(0,200,255,0.6);background:rgba(0,200,255,0.08);}
.sug-tag:nth-child(3){animation-delay:.4s}.sug-tag:nth-child(4){animation-delay:.8s}.sug-tag:nth-child(5){animation-delay:1.2s}
@keyframes taganim{0%,100%{opacity:.38}50%{opacity:.9;border-color:rgba(0,200,255,.42)}}

.claude-thinking{font-family:'Share Tech Mono',monospace;font-size:0.62rem;color:rgba(0,200,255,0.5);padding:10px 14px;background:rgba(0,18,38,0.6);border:1px solid rgba(0,200,255,0.12);border-radius:8px;margin:4px 20px;line-height:2;}
.think-line{animation:thinkfade 1.2s ease-in-out infinite;}
.think-line:nth-child(2){animation-delay:.2s}.think-line:nth-child(3){animation-delay:.4s}
@keyframes thinkfade{0%,100%{opacity:.3}50%{opacity:1}}
.think-bar{margin-top:8px;height:2px;background:rgba(0,200,255,0.08);border-radius:2px;overflow:hidden;}
.think-bar-fill{height:2px;background:linear-gradient(90deg,#00dcff,#00ff88);border-radius:2px;animation:thinkprog 2s ease-in-out infinite;}
@keyframes thinkprog{0%{width:0%}70%{width:90%}100%{width:100%}}
.retrieved-chunks{font-family:'Share Tech Mono',monospace;font-size:0.55rem;color:rgba(0,200,255,0.4);padding:6px 20px;letter-spacing:.06em;line-height:1.9;}
.rc-line{animation:rcfade 0.4s ease forwards;opacity:0;}
.rc-line:nth-child(2){animation-delay:.15s}.rc-line:nth-child(3){animation-delay:.3s}
@keyframes rcfade{to{opacity:1}}

/* image retrieval slide-in */
.img-slide{animation:slideIn 0.5s cubic-bezier(0.34,1.56,0.64,1) forwards;opacity:0;transform:translateY(20px);}
@keyframes slideIn{to{opacity:1;transform:translateY(0)}}
</style>
""", unsafe_allow_html=True)

# ── HERO ──────────────────────────────────────────────────────────────────────
HERO_HTML = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&display=swap');
*{box-sizing:border-box;margin:0;padding:0}
body{background:#020208}
#hero{position:relative;width:100%;height:210px;background:#020208;overflow:hidden;border-bottom:1px solid rgba(0,200,255,0.12)}
#mc,#pc{position:absolute;inset:0;width:100%;height:100%}
.gbg{position:absolute;inset:0;background-image:linear-gradient(rgba(0,200,255,0.03) 1px,transparent 1px),linear-gradient(90deg,rgba(0,200,255,0.03) 1px,transparent 1px);background-size:44px 44px}
.vig{position:absolute;inset:0;background:radial-gradient(ellipse at 50% 50%,transparent 20%,rgba(2,2,8,0.88) 100%)}
.ctl{position:absolute;top:0;left:0;width:34px;height:34px;border-top:1px solid rgba(0,200,255,.45);border-left:1px solid rgba(0,200,255,.45)}
.ctr{position:absolute;top:0;right:0;width:34px;height:34px;border-top:1px solid rgba(0,200,255,.45);border-right:1px solid rgba(0,200,255,.45)}
.cbl{position:absolute;bottom:0;left:0;width:34px;height:34px;border-bottom:1px solid rgba(0,200,255,.45);border-left:1px solid rgba(0,200,255,.45)}
.cbr{position:absolute;bottom:0;right:0;width:34px;height:34px;border-bottom:1px solid rgba(0,200,255,.45);border-right:1px solid rgba(0,200,255,.45)}
.status{position:absolute;top:12px;right:14px;display:flex;gap:8px;align-items:center;z-index:10}
.dot{width:7px;height:7px;border-radius:50%;background:#00ff88;box-shadow:0 0 8px #00ff88;animation:bp 1.8s infinite}
@keyframes bp{0%,100%{opacity:1}50%{opacity:.15}}
.stxt{font-family:'Share Tech Mono',monospace;font-size:9px;color:rgba(0,200,255,.5);letter-spacing:.12em}
.clk{font-family:'Orbitron',monospace;font-size:10px;color:rgba(0,200,255,.65);letter-spacing:.1em}
.hc{position:absolute;bottom:18px;left:22px;z-index:10}
.htag{font-family:'Share Tech Mono',monospace;font-size:9px;letter-spacing:.2em;color:rgba(0,220,255,.65);border:1px solid rgba(0,220,255,.25);padding:2px 9px;border-radius:2px;display:inline-block;margin-bottom:8px;position:relative}
.htag::before{content:'';position:absolute;left:-1px;top:-1px;width:5px;height:5px;border-top:1px solid #00dcff;border-left:1px solid #00dcff}
.htag::after{content:'';position:absolute;right:-1px;bottom:-1px;width:5px;height:5px;border-bottom:1px solid #00dcff;border-right:1px solid #00dcff}
.hh1{font-family:'Orbitron',monospace;font-size:2.1rem;font-weight:900;line-height:1.05;color:#fff}
.hh1 span{color:#00dcff;animation:glow 2.5s ease-in-out infinite alternate}
@keyframes glow{from{text-shadow:0 0 8px rgba(0,200,255,.3)}to{text-shadow:0 0 28px rgba(0,200,255,.9),0 0 55px rgba(0,200,255,.3)}}
.hsub{font-family:'Share Tech Mono',monospace;font-size:.52rem;color:rgba(0,200,255,.28);letter-spacing:.12em;margin-top:5px}
</style>
<div id="hero">
  <canvas id="mc"></canvas><canvas id="pc"></canvas>
  <div class="gbg"></div><div class="vig"></div>
  <div class="ctl"></div><div class="ctr"></div><div class="cbl"></div><div class="cbr"></div>
  <div class="status"><div class="dot"></div><span class="stxt">ONLINE</span><span class="clk" id="clk">--:-- --</span></div>
  <div class="hc">
    <div class="htag">✦ PDF INTELLIGENCE · RAG v2.0</div>
    <div class="hh1">QUERY YOUR<br><span>DOCUMENT</span></div>
    <div class="hsub">// claude ai · faiss · sentence transformers</div>
  </div>
</div>
<script>
(function(){
  function tick(){var n=new Date(),e=document.getElementById('clk');if(!e)return;var h=n.getHours(),m=n.getMinutes(),ap=h>=12?'PM':'AM';h=h%12||12;e.textContent=(h<10?'0'+h:h)+':'+(m<10?'0'+m:m)+' '+ap;}
  setInterval(tick,1000);tick();
  var mc=document.getElementById('mc'),mctx=mc.getContext('2d'),pc=document.getElementById('pc'),pctx=pc.getContext('2d'),hero=document.getElementById('hero'),W,H,cols=[],pts=[],chars='01アイウエオカキクサシスタチツテトナニネノ',FS=13;
  function resz(){W=mc.width=pc.width=hero.offsetWidth;H=mc.height=pc.height=hero.offsetHeight;cols=[];var n=Math.floor(W/FS);for(var i=0;i<n;i++)cols.push({y:Math.random()*H,sp:Math.random()*1.4+0.3,br:Math.random()>.82,tr:Math.floor(Math.random()*8)+4});pts=[];for(var i=0;i<55;i++)pts.push({x:Math.random()*W,y:Math.random()*H,vx:(Math.random()-.5)*.5,vy:(Math.random()-.5)*.5,r:Math.random()*1.8+.4,a:Math.random()*.5+.15});}
  function loop(){
    mctx.fillStyle='rgba(2,2,8,0.16)';mctx.fillRect(0,0,W,H);mctx.font=FS+'px monospace';
    cols.forEach(function(c,i){var ch=chars[Math.floor(Math.random()*chars.length)];mctx.fillStyle=c.br?'rgba(200,240,255,0.95)':'rgba(0,180,255,0.14)';mctx.fillText(ch,i*FS,c.y);c.y+=c.sp;if(c.y>H+FS){c.y=-FS*c.tr;c.br=Math.random()>.82;}});
    pctx.clearRect(0,0,W,H);pts.forEach(function(p){p.x+=p.vx;p.y+=p.vy;if(p.x<0)p.x=W;if(p.x>W)p.x=0;if(p.y<0)p.y=H;if(p.y>H)p.y=0;pctx.beginPath();pctx.arc(p.x,p.y,p.r,0,Math.PI*2);pctx.fillStyle='rgba(0,220,255,'+p.a+')';pctx.fill();});
    for(var i=0;i<pts.length;i++)for(var j=i+1;j<pts.length;j++){var d=Math.hypot(pts[i].x-pts[j].x,pts[i].y-pts[j].y);if(d<90){pctx.beginPath();pctx.moveTo(pts[i].x,pts[i].y);pctx.lineTo(pts[j].x,pts[j].y);pctx.strokeStyle='rgba(0,180,255,'+(0.12*(1-d/90))+')';pctx.lineWidth=0.5;pctx.stroke();}}
    requestAnimationFrame(loop);
  }
  resz();loop();window.addEventListener('resize',resz);
})();
</script>
"""

# ── ANIMATED RAG PIPELINE ─────────────────────────────────────────────────────
RAG_PIPELINE_HTML = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700&family=Share+Tech+Mono&display=swap');
*{box-sizing:border-box;margin:0;padding:0}
body{background:#020208}
#pipeline{background:rgba(0,3,14,0.92);border-bottom:1px solid rgba(0,200,255,0.1);padding:14px 20px 10px;position:relative;overflow:hidden;}
#flow-canvas{position:absolute;inset:0;width:100%;height:100%;pointer-events:none;}
.pl-label{font-family:'Share Tech Mono',monospace;font-size:0.48rem;letter-spacing:.2em;color:rgba(0,200,255,0.22);text-transform:uppercase;margin-bottom:10px;}
.pl-nodes{display:flex;align-items:center;gap:0;overflow-x:auto;padding-bottom:4px;}
.pl-nodes::-webkit-scrollbar{height:2px;}.pl-nodes::-webkit-scrollbar-thumb{background:rgba(0,200,255,0.2);}
.pnode{display:flex;flex-direction:column;align-items:center;gap:4px;min-width:62px;flex-shrink:0;position:relative;z-index:2;}
.pnode-box{width:36px;height:36px;border-radius:8px;border:1px solid rgba(0,200,255,0.12);display:flex;align-items:center;justify-content:center;font-size:11px;background:rgba(0,200,255,0.02);font-family:'Share Tech Mono',monospace;color:rgba(0,200,255,0.22);transition:all 0.4s;position:relative;overflow:hidden;}
.pnode-box::after{content:'';position:absolute;inset:0;background:rgba(0,200,255,0);transition:background 0.4s;}
.pnode-lbl{font-family:'Share Tech Mono',monospace;font-size:0.3rem;color:rgba(0,200,255,0.18);letter-spacing:.04em;text-align:center;text-transform:uppercase;line-height:1.3;max-width:58px;}
.parrow{color:rgba(0,200,255,0.1);font-size:0.6rem;padding:0 1px;margin-bottom:14px;flex-shrink:0;transition:all 0.5s;font-family:'Share Tech Mono',monospace;}

/* active */
.pnode.active .pnode-box{border-color:rgba(0,200,255,0.9);background:rgba(0,200,255,0.14);color:#00dcff;box-shadow:0 0 0 2px rgba(0,200,255,0.15),0 0 20px rgba(0,200,255,0.5);}
.pnode.active .pnode-lbl{color:rgba(0,200,255,0.85);}
.pnode.active .pnode-box::after{background:rgba(0,200,255,0.06);}
/* done */
.pnode.done .pnode-box{border-color:rgba(0,255,136,0.45);background:rgba(0,255,136,0.05);color:#00ff88;box-shadow:0 0 10px rgba(0,255,136,0.2);}
.pnode.done .pnode-lbl{color:rgba(0,255,136,0.5);}
/* arrow lit */
.parrow.lit{color:#00dcff;text-shadow:0 0 8px rgba(0,200,255,0.8);animation:arrowpulse 0.6s ease-in-out;}
@keyframes arrowpulse{0%{opacity:.2}50%{opacity:1}100%{opacity:1}}
/* scanning bar inside active box */
@keyframes scanbox{0%{top:-100%}100%{top:200%}}
.pnode.active .pnode-box .scan{position:absolute;left:0;right:0;height:30%;background:linear-gradient(transparent,rgba(0,200,255,0.3),transparent);animation:scanbox 1s linear infinite;pointer-events:none;}
</style>

<div id="pipeline">
  <canvas id="flow-canvas"></canvas>
  <div class="pl-label">// rag pipeline · retrieve → understand → answer</div>
  <div class="pl-nodes" id="pl-nodes">
    <div class="pnode" id="pn0"><div class="pnode-box">📄<div class="scan"></div></div><div class="pnode-lbl">PDF<br>upload</div></div>
    <div class="parrow" id="pa0">──▶</div>
    <div class="pnode" id="pn1"><div class="pnode-box">T<div class="scan"></div></div><div class="pnode-lbl">text +<br>ocr</div></div>
    <div class="parrow" id="pa1">──▶</div>
    <div class="pnode" id="pn2"><div class="pnode-box">🖼<div class="scan"></div></div><div class="pnode-lbl">image<br>extract</div></div>
    <div class="parrow" id="pa2">──▶</div>
    <div class="pnode" id="pn3"><div class="pnode-box">⬡<div class="scan"></div></div><div class="pnode-lbl">chunk<br>ing</div></div>
    <div class="parrow" id="pa3">──▶</div>
    <div class="pnode" id="pn4"><div class="pnode-box">●<div class="scan"></div></div><div class="pnode-lbl">embed<br>dings</div></div>
    <div class="parrow" id="pa4">──▶</div>
    <div class="pnode" id="pn5"><div class="pnode-box">◈<div class="scan"></div></div><div class="pnode-lbl">faiss<br>index</div></div>
    <div class="parrow" id="pa5">──▶</div>
    <div class="pnode" id="pn6"><div class="pnode-box">◎<div class="scan"></div></div><div class="pnode-lbl">vector<br>search</div></div>
    <div class="parrow" id="pa6">──▶</div>
    <div class="pnode" id="pn7"><div class="pnode-box">AI<div class="scan"></div></div><div class="pnode-lbl">claude<br>ai</div></div>
    <div class="parrow" id="pa7">──▶</div>
    <div class="pnode" id="pn8"><div class="pnode-box">💬<div class="scan"></div></div><div class="pnode-lbl">answer</div></div>
  </div>
</div>

<script>
(function(){
  var total=9, step=0, looping=true;

  function setNode(i,cls){
    var n=document.getElementById('pn'+i);
    if(n){n.className='pnode '+cls;}
    if(cls==='active'){var b=n.querySelector('.pnode-box');if(b){b.innerHTML=b.innerHTML;}}
  }
  function setArrow(i,lit){
    var a=document.getElementById('pa'+i);
    if(a){a.className='parrow'+(lit?' lit':'');}
  }

  function runStep(){
    if(step>0){setNode(step-1,'done');if(step>1)setArrow(step-2,'lit');}
    if(step>=total){
      // pause then restart
      setTimeout(function(){
        for(var i=0;i<total;i++){setNode(i,'');setArrow(i,'');}
        step=0;setTimeout(runStep,600);
      },2200);
      return;
    }
    setNode(step,'active');
    if(step>0)setArrow(step-1,'lit');
    step++;
    setTimeout(runStep, 520);
  }
  setTimeout(runStep,400);

  // flowing energy particles on canvas
  var cv=document.getElementById('flow-canvas'),ctx=cv.getContext('2d');
  var W,H,dots=[];
  function res(){W=cv.width=cv.parentElement.offsetWidth;H=cv.height=cv.parentElement.offsetHeight;}
  for(var i=0;i<18;i++)dots.push({x:Math.random()*800,y:Math.random()*90,vx:Math.random()*1.2+0.5,a:Math.random()*.3+.05,r:Math.random()*1.5+.5});
  function drDots(){
    ctx.clearRect(0,0,W,H);
    dots.forEach(function(d){
      ctx.beginPath();ctx.arc(d.x,d.y,d.r,0,Math.PI*2);
      ctx.fillStyle='rgba(0,200,255,'+d.a+')';ctx.fill();
      d.x+=d.vx;
      if(d.x>W){d.x=0;d.y=20+Math.random()*60;}
    });
    requestAnimationFrame(drDots);
  }
  res();drDots();
  window.addEventListener('resize',res);
})();
</script>
"""

# ── ANIMATED EMPTY STATE ──────────────────────────────────────────────────────
EMPTY_STATE_HTML = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Share+Tech+Mono&display=swap');
*{box-sizing:border-box;margin:0;padding:0}
body{background:#020208}
#es{padding:40px 32px;text-align:center;background:#020208;}
.es-title{font-family:'Orbitron',monospace;font-size:0.9rem;font-weight:400;color:rgba(0,200,255,0.2);letter-spacing:.14em;margin-bottom:6px;}
.es-sub{font-family:'Share Tech Mono',monospace;font-size:0.6rem;color:rgba(0,200,255,0.12);letter-spacing:.08em;margin-bottom:32px;}
#anim-canvas{display:block;margin:0 auto;}
.es-flow{display:flex;align-items:center;justify-content:center;gap:8px;margin-top:24px;font-family:'Share Tech Mono',monospace;font-size:0.55rem;color:rgba(0,200,255,0.15);letter-spacing:.1em;}
.efa{animation:flowarrow 2s ease-in-out infinite;}
.efa:nth-child(2){animation-delay:.3s}.efa:nth-child(4){animation-delay:.6s}.efa:nth-child(6){animation-delay:.9s}
@keyframes flowarrow{0%,100%{opacity:.12}50%{opacity:.7}}
</style>
<div id="es">
  <div class="es-title">UPLOAD A DOCUMENT TO BEGIN</div>
  <div class="es-sub">// open sidebar → drop pdf → ask anything</div>
  <canvas id="anim-canvas" width="440" height="160"></canvas>
  <div class="es-flow">
    <span>PDF</span><span class="efa">→</span>
    <span>TEXT+IMAGES</span><span class="efa">→</span>
    <span>EMBEDDINGS</span><span class="efa">→</span>
    <span>AI</span><span class="efa">→</span>
    <span>ANSWERS</span>
  </div>
</div>
<script>
(function(){
  var cv=document.getElementById('anim-canvas'),ctx=cv.getContext('2d');
  var W=440,H=160;
  var t=0;

  // particles streaming PDF→AI
  var dots=[];
  for(var i=0;i<30;i++)dots.push({x:Math.random()*W,y:60+Math.random()*40,vx:Math.random()*1.5+0.5,a:Math.random()*.5+.1,r:Math.random()*2+.5,col:Math.random()>.5?'0,200,255':'0,255,136'});

  // floating text chunks
  var chunks=[];
  var words=['A','B','C','D','the','AI','doc','text','data','●●●●'];
  for(var i=0;i<8;i++)chunks.push({x:20+i*40,y:30+Math.random()*20,vy:-0.3-Math.random()*0.2,a:0,life:Math.random()*200,word:words[i%words.length]});

  // floating image icons
  var imgs=[];
  var iicons=['🖼','📊','📈'];
  for(var i=0;i<3;i++)imgs.push({x:240+i*50,y:40,vy:-0.25-Math.random()*0.15,a:0,life:Math.random()*200+i*80,icon:iicons[i]});

  function draw(){
    ctx.clearRect(0,0,W,H);
    t++;

    // PDF box left
    ctx.strokeStyle='rgba(0,200,255,0.35)';ctx.lineWidth=1;
    ctx.strokeRect(10,55,55,50);
    ctx.fillStyle='rgba(0,200,255,0.06)';ctx.fillRect(10,55,55,50);
    ctx.fillStyle='rgba(200,60,60,0.8)';ctx.font='11px monospace';ctx.textAlign='center';ctx.fillText('PDF',37,86);

    // scanning line inside PDF box
    var sy=55+((t*0.8)%52);
    ctx.fillStyle='rgba(0,200,255,0.12)';ctx.fillRect(11,sy,53,2);

    // split lines
    ctx.strokeStyle='rgba(0,200,255,0.2)';ctx.lineWidth=0.8;
    ctx.beginPath();ctx.moveTo(65,80);ctx.lineTo(110,50);ctx.stroke();
    ctx.beginPath();ctx.moveTo(65,80);ctx.lineTo(110,110);ctx.stroke();

    // TEXT box
    ctx.strokeStyle='rgba(0,200,255,0.3)';ctx.lineWidth=0.8;
    ctx.strokeRect(110,25,70,45);
    ctx.fillStyle='rgba(0,200,255,0.04)';ctx.fillRect(110,25,70,45);
    ctx.fillStyle='rgba(0,200,255,0.5)';ctx.font='bold 9px monospace';ctx.textAlign='center';ctx.fillText('TEXT',145,43);
    // floating chars from text box
    chunks.forEach(function(c){
      c.life++;c.y+=c.vy;
      c.a=Math.min(0.7,c.life/30)*Math.max(0,(1-(c.life-120)/80));
      if(c.life>200){c.life=0;c.y=30+Math.random()*20;c.a=0;}
      ctx.fillStyle='rgba(0,200,255,'+c.a+')';ctx.font='8px monospace';ctx.textAlign='center';
      ctx.fillText(c.word,c.x,c.y);
    });

    // IMAGES box
    ctx.strokeStyle='rgba(100,80,255,0.35)';ctx.lineWidth=0.8;
    ctx.strokeRect(110,85,70,45);
    ctx.fillStyle='rgba(100,80,255,0.04)';ctx.fillRect(110,85,70,45);
    ctx.fillStyle='rgba(140,100,255,0.6)';ctx.font='bold 9px monospace';ctx.textAlign='center';ctx.fillText('IMAGES',145,103);
    // floating icons
    imgs.forEach(function(c){
      c.life++;c.y+=c.vy;
      c.a=Math.min(1,c.life/40)*Math.max(0,(1-(c.life-140)/80));
      if(c.life>220){c.life=0;c.y=105;c.a=0;}
      ctx.globalAlpha=c.a;ctx.font='13px sans-serif';ctx.textAlign='center';
      ctx.fillText(c.icon,c.x,c.y);ctx.globalAlpha=1;
    });

    // right arrow to embeddings
    ctx.strokeStyle='rgba(0,200,255,0.25)';ctx.lineWidth=0.8;
    ctx.beginPath();ctx.moveTo(180,80);ctx.lineTo(220,80);ctx.stroke();
    ctx.fillStyle='rgba(0,200,255,0.4)';ctx.font='8px monospace';ctx.textAlign='left';ctx.fillText('▶',215,83);

    // EMBEDDINGS zone — vector dots
    var ex=230,ey=55,ew=80,eh=50;
    ctx.strokeStyle='rgba(0,200,255,0.15)';ctx.lineWidth=0.5;ctx.strokeRect(ex,ey,ew,eh);
    ctx.fillStyle='rgba(0,200,255,0.04)';ctx.fillRect(ex,ey,ew,eh);
    ctx.fillStyle='rgba(0,200,255,0.35)';ctx.font='7px monospace';ctx.textAlign='center';ctx.fillText('VECTORS',ex+ew/2,ey+10);
    // vector dots grid
    var cols2=8,rows2=3;
    for(var r=0;r<rows2;r++)for(var c=0;c<cols2;c++){
      var px=ex+8+c*9,py=ey+18+r*10;
      var glow=Math.sin(t*0.05+r*2+c*0.5)*0.5+0.5;
      ctx.beginPath();ctx.arc(px,py,2,0,Math.PI*2);
      ctx.fillStyle='rgba(0,220,255,'+(0.3+glow*0.5)+')';
      if(glow>.8)ctx.shadowBlur=6,ctx.shadowColor='#00dcff';
      ctx.fill();ctx.shadowBlur=0;
    }

    // arrow to FAISS
    ctx.strokeStyle='rgba(0,200,255,0.25)';ctx.lineWidth=0.8;
    ctx.beginPath();ctx.moveTo(310,80);ctx.lineTo(348,80);ctx.stroke();
    ctx.fillStyle='rgba(0,200,255,0.4)';ctx.textAlign='left';ctx.fillText('▶',343,83);

    // FAISS search pulse
    var fx=350,fy=55,fr=28;
    ctx.beginPath();ctx.arc(fx,fy+fr,fr,0,Math.PI*2);
    var pulse=Math.sin(t*0.06)*0.5+0.5;
    ctx.strokeStyle='rgba(0,200,255,'+(0.2+pulse*0.4)+')';ctx.lineWidth=1;ctx.stroke();
    // pulse ring
    ctx.beginPath();ctx.arc(fx,fy+fr,fr+6*pulse,0,Math.PI*2);
    ctx.strokeStyle='rgba(0,200,255,'+(0.08+pulse*0.12)+')';ctx.lineWidth=0.5;ctx.stroke();
    ctx.fillStyle='rgba(0,200,255,0.5)';ctx.font='7px monospace';ctx.textAlign='center';ctx.fillText('FAISS',fx,fy+fr+3);

    // AI box right
    ctx.strokeStyle='rgba(0,255,136,0.4)';ctx.lineWidth=1;
    ctx.strokeRect(395,55,44,50);
    ctx.fillStyle='rgba(0,255,136,0.04)';ctx.fillRect(395,55,44,50);
    ctx.fillStyle='rgba(0,255,136,0.7)';ctx.font='bold 11px monospace';ctx.textAlign='center';ctx.fillText('AI',417,85);

    // flowing dots on main stream
    dots.forEach(function(d){
      ctx.beginPath();ctx.arc(d.x,d.y,d.r,0,Math.PI*2);
      ctx.fillStyle='rgba('+d.col+','+d.a+')';ctx.fill();
      d.x+=d.vx;
      if(d.x>W+5){d.x=-5;d.y=55+Math.random()*50;}
    });

    requestAnimationFrame(draw);
  }
  draw();
})();
</script>
"""

THINKING_HTML = """
<div class="claude-thinking">
  <div class="think-line">> retrieving relevant chunks…</div>
  <div class="think-line">> ranking by relevance…</div>
  <div class="think-line">> generating answer…</div>
  <div class="think-bar"><div class="think-bar-fill"></div></div>
</div>
"""

def retrieval_html(chunk_indices):
    lines = "".join(f'<div class="rc-line">› Chunk {i} ✓</div>' for i in chunk_indices)
    return f'<div class="retrieved-chunks">{lines}</div>'

CHAT_BANNER_HTML = """
<div class="sug-bar">
  <div class="sug-prefix">TRY →</div>
  <div class="sug-tag">summarize this</div>
  <div class="sug-tag">show 1st image</div>
  <div class="sug-tag">chapter 3?</div>
  <div class="sug-tag">show all figures</div>
  <div class="sug-tag">key findings?</div>
</div>
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
    return any(t in q.lower() for t in ["show","display","see","view","image","figure","picture","photo","diagram","chart","graph","illustration","img"])

def find_matching_images(question, image_store):
    if not image_store: return []
    q = question.lower()
    ordinals = {"1st":0,"first":0,"2nd":1,"second":1,"3rd":2,"third":2,"4th":3,"fourth":3,"5th":4,"fifth":4,"6th":5,"sixth":5}
    for word,idx in ordinals.items():
        if word in q and idx < len(image_store): return [image_store[idx]]
    for n in re.findall(r'\b(\d+)\b', q):
        idx = int(n)-1
        if 0 <= idx < len(image_store): return [image_store[idx]]
    if "last" in q: return [image_store[-1]]
    if "all" in q: return image_store
    if any(t in q for t in ["image","figure","picture","diagram","chart","graph"]): return [image_store[0]]
    return []

def fmt_time(s): return f"{s:.1f}s" if s < 60 else f"{int(s//60)}m {s%60:.0f}s"
def local_time(): return datetime.now().strftime("%I:%M %p").lstrip("0")

# animated counter helper
def count_up_html(val, label, delay_ms=0):
    return f"""
    <div class="metric-card">
      <div class="metric-val" id="cv_{label.replace(' ','_')}">{val if val else '—'}</div>
      <div class="metric-lbl">{label}</div>
    </div>
    <script>
    (function(){{
      var el=document.getElementById('cv_{label.replace(' ','_')}');
      if(!el||{int(val) if val else 0}===0)return;
      var target={int(val) if val else 0},cur=0,step=Math.max(1,Math.floor(target/40));
      setTimeout(function(){{
        var t=setInterval(function(){{
          cur=Math.min(cur+step,target);
          el.textContent=cur;
          if(cur>=target)clearInterval(t);
        }},30);
      }},{delay_ms});
    }})();
    </script>
    """

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="padding:14px 14px 10px;border-bottom:1px solid rgba(0,200,255,0.1);margin-bottom:8px;">
      <div style="font-family:'Orbitron',monospace;font-size:0.88rem;font-weight:700;color:#00dcff;letter-spacing:.06em;">PDF BOT</div>
      <div style="font-family:'Share Tech Mono',monospace;font-size:0.54rem;color:rgba(0,200,255,0.22);letter-spacing:.14em;margin-top:2px;">INTELLIGENCE SYSTEM</div>
    </div>
    """, unsafe_allow_html=True)

    # ── UPLOAD ──
    st.markdown('<span class="s-label">// Drop your PDF</span>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type="pdf", label_visibility="collapsed")

    if uploaded_file:
        st.markdown(f"""
        <div style="font-family:'Share Tech Mono',monospace;font-size:0.6rem;color:rgba(0,200,255,0.5);
          padding:5px 8px;border-left:2px solid rgba(0,200,255,0.3);margin-bottom:8px;word-break:break-all;">
          › {uploaded_file.name}
        </div>""", unsafe_allow_html=True)

        if st.button("⚡  PROCESS & INDEX"):
            t_start = time.time()
            proc_ph = st.empty()

            STEP_LABELS = ["extracting text + images","analyzing images (claude vision)","chunking document","generating embeddings","building FAISS index","complete ✓"]

            def show_proc(step, pct, pages=0, chunks=0, imgs=0, vecs=0):
                rows = ""
                for i, lbl in enumerate(STEP_LABELS):
                    cls = "ok" if i < step else ("live" if i == step else "dim")
                    ind = "✓" if i < step else ("●" if i == step else "○")
                    rows += f'<div style="font-family:\'Share Tech Mono\',monospace;font-size:0.6rem;letter-spacing:.06em;display:flex;align-items:center;gap:8px;padding:3px 0;color:{\"rgba(0,255,136,0.75)\" if cls==\"ok\" else (\"rgba(0,200,255,0.9)\" if cls==\"live\" else \"rgba(0,200,255,0.2)\")};"><span>{ind}</span>> {lbl}</div>'
                counters = ""
                if pct >= 75:
                    counters = f"""<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin-top:8px;">
                      {"".join(f'<div style="text-align:center;"><div style="font-family:Orbitron,monospace;font-size:0.78rem;color:#00dcff;">{v}</div><div style="font-family:Share Tech Mono,monospace;font-size:0.44rem;color:rgba(0,200,255,0.3);margin-top:2px;">{l}</div></div>' for v,l in [(pages,"pages"),(chunks,"chunks"),(imgs,"images"),(vecs,"vectors")])}
                    </div>"""
                titles = ["EXTRACTING","ANALYZING IMAGES","CHUNKING","EMBEDDING","BUILDING INDEX","COMPLETE ✓"]
                title = titles[min(step, len(titles)-1)]
                proc_ph.markdown(f"""
                <div style="background:rgba(0,10,28,0.95);border:1px solid rgba(0,200,255,0.2);border-radius:10px;padding:16px 18px;margin:8px 0;">
                  <div style="font-family:'Orbitron',monospace;font-size:0.72rem;color:#00dcff;letter-spacing:.1em;margin-bottom:10px;">⬡ {title} — {uploaded_file.name[:28]}{'…' if len(uploaded_file.name)>28 else ''}</div>
                  <div style="background:rgba(0,200,255,0.08);border-radius:2px;height:2px;margin-bottom:10px;overflow:hidden;"><div style="height:2px;background:linear-gradient(90deg,#00dcff,#00ff88);border-radius:2px;width:{pct}%;transition:width 0.4s;"></div></div>
                  {rows}{counters}
                </div>""", unsafe_allow_html=True)

            show_proc(0, 5)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read()); tmp_path = tmp.name

            doc = fitz.open(tmp_path)
            full_text, total, image_store, img_global_idx = "", len(doc), [], 0

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
                                {"type":"text","text":"Describe this image concisely. If it's a diagram/chart, explain what it shows."}
                            ]}]
                        )
                        desc = resp.content[0].text
                        image_store.append({"b64":b64,"media_type":mt,"page":pnum+1,"index":img_global_idx,"description":desc})
                        img_global_idx += 1
                        full_text += f"\n[Image {img_global_idx} on page {pnum+1}]: {desc}\n"
                    except Exception: continue
                show_proc(1, 15+int((pnum+1)/total*40))

            doc.close(); os.unlink(tmp_path)
            t_process = time.time() - t_start

            show_proc(2, 58)
            chunks = extract_chunks(full_text); time.sleep(0.15)

            show_proc(3, 70)
            t_embed = time.time()
            embs = np.array(model.encode(chunks)).astype('float32')

            show_proc(4, 88, pages=total, chunks=len(chunks), imgs=len(image_store), vecs=len(chunks))
            index = faiss.IndexFlatL2(embs.shape[1]); index.add(embs)
            t_embed_done = time.time() - t_embed; time.sleep(0.25)

            show_proc(5, 100, pages=total, chunks=len(chunks), imgs=len(image_store), vecs=len(chunks))
            time.sleep(0.4); proc_ph.empty()

            st.session_state.update({
                "index":index,"chunks":chunks,"messages":[],"chat_display":[],
                "chunk_count":len(chunks),"image_store":image_store,"image_count":len(image_store),
                "page_count":total,"process_time":t_process,"embed_time":t_embed_done,
            })
            st.success(f"✓ {len(chunks)} chunks · {len(image_store)} imgs · {fmt_time(t_process)}")

    st.markdown('<hr class="s-divider">', unsafe_allow_html=True)

    # ── LIVE COUNTER METRICS ──
    chunk_count = st.session_state.get("chunk_count", 0)
    img_count   = st.session_state.get("image_count", 0)
    page_count  = st.session_state.get("page_count",  0)
    proc_time   = st.session_state.get("process_time", None)
    embed_time  = st.session_state.get("embed_time",   None)
    proc_disp   = fmt_time(proc_time)  if proc_time  else "—"
    embed_disp  = fmt_time(embed_time) if embed_time else "—"

    st.markdown('<span class="s-label">// Last Run</span>', unsafe_allow_html=True)
    # animated counting metrics
    st.markdown(f"""
    <div class="metric-grid">
      <div class="metric-card">
        <div class="metric-val" id="mc_pages" style="transition:all 0.1s;">{page_count or '—'}</div>
        <div class="metric-lbl">Pages</div>
      </div>
      <div class="metric-card">
        <div class="metric-val" id="mc_chunks">{chunk_count or '—'}</div>
        <div class="metric-lbl">Chunks</div>
      </div>
      <div class="metric-card">
        <div class="metric-val" id="mc_imgs">{img_count or '—'}</div>
        <div class="metric-lbl">Images</div>
      </div>
      <div class="metric-card">
        <div class="metric-val" id="mc_vecs">{chunk_count or '—'}</div>
        <div class="metric-lbl">Vectors</div>
      </div>
      <div class="metric-card">
        <div class="metric-val">{proc_disp}</div>
        <div class="metric-lbl">Process</div>
      </div>
      <div class="metric-card">
        <div class="metric-val">{embed_disp}</div>
        <div class="metric-lbl">Index time</div>
      </div>
    </div>
    <script>
    (function(){{
      function animCount(id,target){{
        var el=document.getElementById(id);
        if(!el||!target||isNaN(target))return;
        var cur=0,step=Math.max(1,Math.floor(target/35));
        var t=setInterval(function(){{cur=Math.min(cur+step,target);el.textContent=cur;if(cur>=target)clearInterval(t);}},25);
      }}
      setTimeout(function(){{
        animCount('mc_pages',{page_count});
        animCount('mc_chunks',{chunk_count});
        animCount('mc_imgs',{img_count});
        animCount('mc_vecs',{chunk_count});
      }},200);
    }})();
    </script>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="s-divider">', unsafe_allow_html=True)

    # ── TIPS ──
    st.markdown("""
    <span class="s-label">// Query Tips</span>
    <div class="sys-tips">
      › ask specific questions<br>
      › "show me the 1st image"<br>
      › "show all figures"<br>
      › reference page numbers<br>
      › request summaries
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="s-divider">', unsafe_allow_html=True)

    # ── CONTACT HOVER CARD ──
    st.markdown("""
    <style>
    @keyframes nameGlow{from{text-shadow:0 0 5px rgba(0,200,255,0.3)}to{text-shadow:0 0 20px rgba(0,200,255,1),0 0 40px rgba(0,200,255,0.4)}}
    @keyframes cardIn{from{opacity:0;transform:translateY(-6px) scale(0.97)}to{opacity:1;transform:none}}
    @keyframes particleFloat{0%{transform:translateY(0) translateX(0);opacity:0}20%{opacity:1}100%{transform:translateY(-60px) translateX(var(--dx));opacity:0}}
    .contact-hover-wrap{position:relative;cursor:pointer;}
    .contact-hover-trigger{padding:10px 12px;background:rgba(0,200,255,0.04);border:1px solid rgba(0,200,255,0.2);border-radius:8px;font-family:'Share Tech Mono',monospace;font-size:0.62rem;color:rgba(0,200,255,0.55);letter-spacing:.06em;text-align:center;transition:all 0.25s;position:relative;overflow:hidden;}
    .contact-hover-trigger::before{content:'';position:absolute;left:-100%;top:0;width:60%;height:100%;background:linear-gradient(90deg,transparent,rgba(0,200,255,0.08),transparent);animation:shimmer 2.5s infinite;pointer-events:none;}
    @keyframes shimmer{0%{left:-100%}100%{left:200%}}
    .contact-hover-trigger:hover{border-color:rgba(0,200,255,0.7);background:rgba(0,200,255,0.1);color:#00dcff;box-shadow:0 0 20px rgba(0,200,255,0.25);}
    .contact-popup{display:none;position:absolute;left:0;right:0;bottom:calc(100% + 8px);background:rgba(0,8,28,0.98);border:1px solid rgba(0,200,255,0.35);border-radius:10px;padding:14px 14px 12px;z-index:999;animation:cardIn 0.3s cubic-bezier(0.34,1.56,0.64,1) forwards;box-shadow:0 0 40px rgba(0,200,255,0.2);}
    .contact-hover-wrap:hover .contact-popup{display:block;}
    .c-name{font-family:'Orbitron',monospace;font-size:0.65rem;color:#00dcff;letter-spacing:.08em;margin-bottom:10px;animation:nameGlow 2s ease-in-out infinite alternate;}
    .c-particle{position:absolute;width:4px;height:4px;border-radius:50%;background:#00dcff;pointer-events:none;--dx:0px;animation:particleFloat 1.2s ease-out forwards;}
    .c-link{display:flex;align-items:center;gap:8px;padding:7px 0;font-family:'Share Tech Mono',monospace;font-size:0.6rem;color:rgba(0,200,255,0.65);text-decoration:none;border-bottom:1px solid rgba(0,200,255,0.07);transition:color 0.2s;letter-spacing:.04em;}
    .c-link:last-child{border-bottom:none;padding-bottom:0;}
    .c-link:hover{color:#00dcff;}
    .c-link-icon{width:20px;height:20px;border-radius:4px;border:1px solid rgba(0,200,255,0.25);display:flex;align-items:center;justify-content:center;font-size:9px;flex-shrink:0;background:rgba(0,200,255,0.06);}
    .c-divider{height:1px;background:linear-gradient(90deg,transparent,rgba(0,200,255,0.3),transparent);margin:8px 0;}
    </style>

    <span class="s-label">// Built by</span>
    <div style="font-family:'Share Tech Mono',monospace;font-size:0.7rem;color:rgba(0,200,255,0.6);margin-bottom:8px;">
      Sai Jyothi Gayathri Adabala
    </div>

    <div class="contact-hover-wrap" id="chw">
      <div class="contact-hover-trigger" id="cht">
        ✦ hover for contact info
      </div>
      <div class="contact-popup" id="cpop">
        <div class="c-name">✦ Sai Jyothi Gayathri Adabala</div>
        <div class="c-divider"></div>
        <a class="c-link" href="mailto:asjyothig@gmail.com">
          <div class="c-link-icon">✉</div>
          <div>
            <div style="font-size:0.5rem;color:rgba(0,200,255,0.3);letter-spacing:.1em;text-transform:uppercase;">email</div>
            <div>asjyothig@gmail.com</div>
          </div>
        </a>
        <a class="c-link" href="https://www.linkedin.com/in/sai-jyothi-gayathri-adabala-a41a9818b/" target="_blank">
          <div class="c-link-icon" style="font-weight:bold;">in</div>
          <div>
            <div style="font-size:0.5rem;color:rgba(0,200,255,0.3);letter-spacing:.1em;text-transform:uppercase;">linkedin</div>
            <div>View Profile →</div>
          </div>
        </a>
      </div>
    </div>

    <script>
    (function(){
      var wrap=document.getElementById('chw');
      if(!wrap)return;
      wrap.addEventListener('mouseenter',function(){
        // spawn particles
        for(var i=0;i<8;i++){
          var p=document.createElement('div');
          p.className='c-particle';
          p.style.left=(10+Math.random()*80)+'%';
          p.style.bottom='0';
          p.style.setProperty('--dx',(Math.random()-0.5)*40+'px');
          p.style.animationDelay=(Math.random()*0.4)+'s';
          p.style.background=Math.random()>.5?'#00dcff':'#00ff88';
          wrap.appendChild(p);
          setTimeout(function(pp){pp.remove();},(1200+Math.random()*400),p);
        }
      });
    })();
    </script>

    <div style="margin-top:14px;font-family:'Share Tech Mono',monospace;font-size:7px;
      color:rgba(0,200,255,0.1);letter-spacing:.08em;text-align:center;
      border-top:1px solid rgba(0,200,255,0.05);padding-top:8px;">
      CLAUDE · FAISS · SENTENCE-TRANSFORMERS
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
st.components.v1.html(HERO_HTML, height=212, scrolling=False)

if "index" not in st.session_state:
    st.components.v1.html(EMPTY_STATE_HTML, height=290, scrolling=False)
else:
    if "chat_display" not in st.session_state: st.session_state.chat_display = []
    if "messages" not in st.session_state: st.session_state.messages = []

    st.components.v1.html(RAG_PIPELINE_HTML, height=90, scrolling=False)
    st.markdown(CHAT_BANNER_HTML, unsafe_allow_html=True)
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    for entry in st.session_state.chat_display:
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"])
            side = "ts-right" if entry["role"]=="user" else "ts-left"
            tick = "✓" if entry["role"]=="user" else "⬡"
            st.markdown(f'<span class="ts {side}">{tick} {entry.get("time","")}</span>', unsafe_allow_html=True)
            for img_data in entry.get("images",[]):
                st.markdown(
                    f'<div class="img-frame img-slide"><div class="img-frame-label">'
                    f'<span>// IMAGE {img_data["index"]+1}</span>'
                    f'<span>PAGE {img_data["page"]}</span></div></div>',
                    unsafe_allow_html=True)
                st.image(base64.b64decode(img_data["b64"]), use_column_width=False)

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
        time.sleep(0.35)
        think_ph.empty()

        if images_to_show:
            st.session_state.messages.append({"role":"user","content":f"Context:\n{context}\n\nQuestion: {question}\n\nNote: You ARE showing the user the image(s). Describe in 1-2 sentences."})
            system_prompt = "You are a document assistant displaying images to the user. Briefly describe what the image shows in 1-2 sentences. Never say you cannot show images."
            max_tok = 250
        else:
            st.session_state.messages.append({"role":"user","content":f"Context:\n{context}\n\nQuestion: {question}"})
            system_prompt = "You are an intelligent document assistant. Answer clearly and concisely. Use bullet points or numbered lists when listing multiple items. Bold key terms using **term**. If the exact term isn't in context, look for related concepts."
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
                st.markdown(
                    f'<div class="img-frame img-slide"><div class="img-frame-label">'
                    f'<span>// IMAGE {img_data["index"]+1}</span>'
                    f'<span>PAGE {img_data["page"]}</span></div></div>',
                    unsafe_allow_html=True)
                st.image(base64.b64decode(img_data["b64"]), use_column_width=False)