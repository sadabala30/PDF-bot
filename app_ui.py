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

st.set_page_config(
    page_title="PDF BOT",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

/* ── MAKE SIDEBAR TOGGLE SUPER VISIBLE ── */
[data-testid="collapsedControl"] {
  display: flex !important;
  visibility: visible !important;
  opacity: 1 !important;
  width: 32px !important;
  height: 80px !important;
  background: rgba(0,200,255,0.2) !important;
  border: 2px solid #00dcff !important;
  border-left: none !important;
  border-radius: 0 14px 14px 0 !important;
  box-shadow: 0 0 30px rgba(0,200,255,0.6) !important;
  align-items: center !important;
  justify-content: center !important;
  animation: sbpulse 1.8s ease-in-out infinite !important;
}
[data-testid="collapsedControl"] svg {
  color: #00dcff !important;
  width: 20px !important;
  height: 20px !important;
  filter: drop-shadow(0 0 8px #00dcff) !important;
}
@keyframes sbpulse {
  0%,100% { box-shadow: 0 0 16px rgba(0,200,255,0.4); background: rgba(0,200,255,0.15); }
  50%      { box-shadow: 0 0 40px rgba(0,200,255,0.9), 0 0 60px rgba(0,200,255,0.4); background: rgba(0,200,255,0.32); }
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
  background: rgba(0,4,18,0.98) !important;
  border-right: 1px solid rgba(0,200,255,0.15) !important;
  min-width: 260px !important;
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

/* ── PROGRESS ── */
.stProgress > div > div { background: linear-gradient(90deg,#00dcff,#00ff88) !important; }
.stProgress > div       { background: rgba(0,200,255,0.08) !important; border-radius: 0 !important; }
.stSuccess { background: rgba(0,255,136,0.06) !important; border: 1px solid rgba(0,255,136,0.2) !important; border-radius: 6px !important; color: #00ff88 !important; font-family: 'Share Tech Mono', monospace !important; font-size: 0.75rem !important; }
.stSpinner > div { border-top-color: #00dcff !important; }

.block-container { padding: 0 !important; max-width: 100% !important; }

/* ── CHAT MESSAGES ── */
[data-testid="stChatMessage"] {
  background: transparent !important;
  border: none !important;
  padding: 3px 20px !important;
  margin-bottom: 2px !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
  flex-direction: row !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) > div:last-child {
  background: rgba(0,18,38,0.92) !important;
  border: 1px solid rgba(0,200,255,0.18) !important;
  border-radius: 2px 18px 18px 18px !important;
  padding: 13px 17px !important;
  font-size: 0.88rem !important;
  line-height: 1.75 !important;
  color: #cce8ff !important;
  max-width: 65% !important;
  box-shadow: 0 3px 16px rgba(0,200,255,0.07) !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
  flex-direction: row-reverse !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) > div:last-child {
  background: linear-gradient(135deg, rgba(0,100,200,0.4), rgba(60,0,180,0.45)) !important;
  border: 1px solid rgba(80,160,255,0.35) !important;
  border-radius: 18px 2px 18px 18px !important;
  padding: 13px 17px !important;
  font-size: 0.88rem !important;
  line-height: 1.75 !important;
  color: #e8f6ff !important;
  max-width: 65% !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid*="Avatar"] {
  background: linear-gradient(135deg,#002a40,#005580) !important;
  border: 2px solid rgba(0,200,255,0.55) !important;
  box-shadow: 0 0 12px rgba(0,200,255,0.3) !important;
  border-radius: 50% !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid*="Avatar"] {
  background: linear-gradient(135deg,#1a0050,#4400cc) !important;
  border: 2px solid rgba(160,80,255,0.6) !important;
  box-shadow: 0 0 12px rgba(140,60,255,0.3) !important;
  border-radius: 50% !important;
}

/* ── CHAT INPUT ── */
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
  outline: none !important;
  box-shadow: none !important;
}
[data-testid="stChatInput"] textarea:focus {
  outline: none !important;
  border-color: rgba(0,210,255,0.85) !important;
  box-shadow: 0 0 0 2px rgba(0,200,255,0.2), 0 0 30px rgba(0,200,255,0.25) !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: rgba(0,200,255,0.22) !important; }
[data-testid="stChatInput"] button {
  background: rgba(0,200,255,0.16) !important;
  border: 1.5px solid rgba(0,200,255,0.45) !important;
  color: #00dcff !important;
  border-radius: 50% !important;
}
[data-testid="stChatInput"] button:hover {
  background: rgba(0,200,255,0.32) !important;
  box-shadow: 0 0 20px rgba(0,200,255,0.45) !important;
}

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 3px; height: 3px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(0,200,255,0.25); border-radius: 2px; }

/* ── SIDEBAR UTILS ── */
.metric-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 6px; }
.metric-card { background: rgba(0,200,255,0.03); border: 1px solid rgba(0,200,255,0.1); border-radius: 6px; padding: 10px 8px; text-align: center; transition: all 0.3s; }
.metric-card:hover { border-color: rgba(0,200,255,0.35); background: rgba(0,200,255,0.06); }
.metric-val  { font-family: 'Orbitron', monospace; font-size: 0.85rem; font-weight: 700; color: #00dcff; }
.metric-lbl  { font-family: 'Share Tech Mono', monospace; font-size: 0.55rem; color: rgba(0,200,255,0.35); margin-top: 3px; letter-spacing: .06em; }
.sys-tips    { font-family: 'Share Tech Mono', monospace; font-size: 0.62rem; color: rgba(0,200,255,0.3); line-height: 2.1; letter-spacing: .04em; }
.s-divider   { border: none; border-top: 1px solid rgba(0,200,255,0.07); margin: 10px 0; }
.s-label     { font-family: 'Share Tech Mono', monospace; font-size: 0.58rem; letter-spacing: .18em; color: rgba(0,200,255,0.28); text-transform: uppercase; margin-bottom: 6px; display: block; }

/* ── CONTACT CARD ── */
.contact-card {
  background: rgba(0,200,255,0.04);
  border: 1px solid rgba(0,200,255,0.2);
  border-radius: 8px; padding: 10px 12px; margin-top: 6px;
  animation: cardIn 0.35s ease;
}
@keyframes cardIn { from{opacity:0;transform:translateY(-5px)} to{opacity:1;transform:none} }
.contact-card .c-name {
  font-family: 'Orbitron', monospace; font-size: 0.6rem;
  color: #00dcff; letter-spacing: .08em; margin-bottom: 5px;
  animation: nameGlow 2.5s ease-in-out infinite alternate;
}
@keyframes nameGlow {
  from { text-shadow: 0 0 5px rgba(0,200,255,0.3); }
  to   { text-shadow: 0 0 18px rgba(0,200,255,0.9); }
}

/* ── UPLOAD TOGGLE ── */
.upload-toggle {
  padding: 10px 14px;
  background: rgba(0,200,255,0.06);
  border: 1.5px solid rgba(0,200,255,0.35);
  border-radius: 10px;
  font-family: 'Share Tech Mono', monospace; font-size: 0.68rem;
  color: rgba(0,220,255,0.75); letter-spacing: .12em;
  text-align: center; text-transform: uppercase;
  margin-bottom: 8px; animation: pulse-btn 2s ease-in-out infinite;
}
.upload-toggle.closed {
  border-color: rgba(0,200,255,0.7) !important;
  box-shadow: 0 0 22px rgba(0,200,255,0.4) !important;
}
@keyframes pulse-btn {
  0%,100% { box-shadow: 0 0 8px rgba(0,200,255,0.15); }
  50%      { box-shadow: 0 0 24px rgba(0,200,255,0.45); }
}

/* ── PIPELINE ── */
.pipeline-bar { background: rgba(0,2,14,0.85); border-bottom: 1px solid rgba(0,200,255,0.08); padding: 8px 16px 6px; }
.pipeline-label { font-family: 'Share Tech Mono', monospace; font-size: 0.48rem; letter-spacing: .18em; color: rgba(0,200,255,0.25); text-transform: uppercase; margin-bottom: 6px; }
.pipeline-nodes { display: flex; align-items: center; overflow-x: auto; }
.pipeline-nodes::-webkit-scrollbar { height: 2px; }
.pn { display: flex; flex-direction: column; align-items: center; gap: 3px; min-width: 56px; }
.pn-icon { width: 26px; height: 26px; border-radius: 6px; border: 1px solid rgba(0,200,255,0.15); display: flex; align-items: center; justify-content: center; font-size: 9px; background: rgba(0,200,255,0.03); font-family: 'Share Tech Mono', monospace; color: rgba(0,200,255,0.3); transition: all 0.3s; }
.pn-lbl { font-family: 'Share Tech Mono', monospace; font-size: 0.36rem; color: rgba(0,200,255,0.22); letter-spacing: .05em; text-align: center; }
.pn.active .pn-icon { border-color: rgba(0,200,255,0.85); background: rgba(0,200,255,0.15); box-shadow: 0 0 12px rgba(0,200,255,0.4); color: #00dcff; animation: nodeActive 0.6s ease-in-out infinite alternate; }
.pn.active .pn-lbl { color: rgba(0,200,255,0.85); }
.pn.done .pn-icon { border-color: rgba(0,255,136,0.5); background: rgba(0,255,136,0.06); color: #00ff88; }
.pn.done .pn-lbl { color: rgba(0,255,136,0.55); }
@keyframes nodeActive { from{box-shadow:0 0 6px rgba(0,200,255,0.25)} to{box-shadow:0 0 18px rgba(0,200,255,0.7)} }
.p-arrow { color: rgba(0,200,255,0.12); font-size: 0.55rem; padding: 0 1px; margin-bottom: 12px; flex-shrink: 0; transition: color 0.4s; }
.p-arrow.lit { color: rgba(0,200,255,0.7); text-shadow: 0 0 6px rgba(0,200,255,0.5); }

/* ── PROCESSING STEPS (inline, not overlay) ── */
.proc-panel {
  background: rgba(0,10,28,0.95);
  border: 1px solid rgba(0,200,255,0.2);
  border-radius: 10px; padding: 16px 18px; margin: 8px 0;
}
.proc-panel-title { font-family: 'Orbitron', monospace; font-size: 0.72rem; color: #00dcff; letter-spacing: .1em; margin-bottom: 10px; }
.proc-step-row { font-family: 'Share Tech Mono', monospace; font-size: 0.6rem; letter-spacing: .06em; display: flex; align-items: center; gap: 8px; padding: 3px 0; }
.proc-step-row.dim  { color: rgba(0,200,255,0.2); }
.proc-step-row.live { color: rgba(0,200,255,0.9); }
.proc-step-row.ok   { color: rgba(0,255,136,0.75); }
.proc-bar-wrap { background: rgba(0,200,255,0.08); border-radius: 2px; height: 2px; margin: 8px 0; overflow: hidden; }
.proc-bar-fill { height: 2px; background: linear-gradient(90deg,#00dcff,#00ff88); border-radius: 2px; transition: width 0.4s; }
.proc-counters { display: grid; grid-template-columns: repeat(4,1fr); gap: 6px; margin-top: 8px; }
.pc-val { font-family: 'Orbitron', monospace; font-size: 0.78rem; color: #00dcff; text-align: center; }
.pc-lbl { font-family: 'Share Tech Mono', monospace; font-size: 0.45rem; color: rgba(0,200,255,0.3); text-align: center; margin-top: 2px; }

/* ── EMPTY STATE ── */
.empty-state { text-align: center; padding: 80px 32px; }
.empty-glyph { font-size: 3rem; color: rgba(0,200,255,0.12); margin-bottom: 20px; display: block; font-family: 'Orbitron', monospace; animation: pg 3s ease-in-out infinite; }
@keyframes pg { 0%,100%{opacity:.12} 50%{opacity:.3} }
.empty-state h2 { font-family: 'Orbitron', monospace; font-size: 0.95rem; font-weight: 400; color: rgba(0,200,255,0.2); letter-spacing: .12em; }
.empty-state p  { font-family: 'Share Tech Mono', monospace; font-size: 0.63rem; color: rgba(0,200,255,0.12); margin-top: 10px; letter-spacing: .08em; }
.empty-flow { display: flex; align-items: center; justify-content: center; gap: 8px; margin-top: 16px; font-family: 'Share Tech Mono', monospace; font-size: 0.58rem; color: rgba(0,200,255,0.12); letter-spacing: .1em; }
.ef-arrow { animation: flowarrow 2s ease-in-out infinite; }
.ef-arrow:nth-child(2){animation-delay:.3s} .ef-arrow:nth-child(4){animation-delay:.6s} .ef-arrow:nth-child(6){animation-delay:.9s}
@keyframes flowarrow { 0%,100%{opacity:.12} 50%{opacity:.75} }

/* ── IMAGE FRAME ── */
.img-frame { border: 1px solid rgba(0,200,255,0.2); border-radius: 8px; overflow: hidden; margin: 10px 0; background: #000; }
.img-frame-label { font-family: 'Share Tech Mono', monospace; font-size: 0.6rem; color: rgba(0,200,255,0.5); padding: 6px 12px; border-bottom: 1px solid rgba(0,200,255,0.1); letter-spacing: .1em; display: flex; justify-content: space-between; }

/* ── TIMESTAMPS ── */
.ts { font-family: 'Share Tech Mono', monospace; font-size: 0.57rem; color: rgba(0,200,255,0.2); margin-top: 5px; display: block; }
.ts-right { text-align: right; }
.ts-left  { text-align: left; }

/* ── SUGGESTION TAGS ── */
.sug-bar { display: flex; gap: 7px; align-items: center; padding: 6px 20px; border-top: 1px solid rgba(0,200,255,0.07); background: rgba(0,2,14,0.72); overflow-x: auto; }
.sug-prefix { font-family: 'Share Tech Mono', monospace; font-size: 0.5rem; color: rgba(0,200,255,0.3); letter-spacing: .14em; white-space: nowrap; }
.sug-tag { font-family: 'Share Tech Mono', monospace; font-size: 0.55rem; color: rgba(0,200,255,0.35); border: 1px solid rgba(0,200,255,0.12); padding: 3px 10px; border-radius: 20px; white-space: nowrap; animation: taganim 3s ease-in-out infinite; }
.sug-tag:nth-child(3){animation-delay:.4s} .sug-tag:nth-child(4){animation-delay:.8s} .sug-tag:nth-child(5){animation-delay:1.2s}
@keyframes taganim { 0%,100%{opacity:.35} 50%{opacity:.9;border-color:rgba(0,200,255,.42)} }

/* ── THINKING ── */
.claude-thinking { font-family: 'Share Tech Mono', monospace; font-size: 0.62rem; color: rgba(0,200,255,0.5); padding: 10px 14px; background: rgba(0,18,38,0.6); border: 1px solid rgba(0,200,255,0.12); border-radius: 8px; margin: 4px 20px; line-height: 2; }
.think-line { animation: thinkfade 1.2s ease-in-out infinite; }
.think-line:nth-child(2){animation-delay:.2s} .think-line:nth-child(3){animation-delay:.4s}
@keyframes thinkfade { 0%,100%{opacity:.3} 50%{opacity:1} }
.think-bar { margin-top: 8px; height: 2px; background: rgba(0,200,255,0.08); border-radius: 2px; overflow: hidden; }
.think-bar-fill { height: 2px; width: 0%; background: linear-gradient(90deg,#00dcff,#00ff88); border-radius: 2px; animation: thinkprog 2s ease-in-out infinite; }
@keyframes thinkprog { 0%{width:0%} 70%{width:90%} 100%{width:100%} }

/* ── RETRIEVED CHUNKS ── */
.retrieved-chunks { font-family: 'Share Tech Mono', monospace; font-size: 0.55rem; color: rgba(0,200,255,0.4); padding: 6px 20px; letter-spacing: .06em; line-height: 1.9; }
.rc-line { animation: rcfade 0.4s ease forwards; opacity: 0; }
.rc-line:nth-child(2){animation-delay:.15s} .rc-line:nth-child(3){animation-delay:.3s}
@keyframes rcfade { to{opacity:1} }
</style>
""", unsafe_allow_html=True)

# ── HERO ──────────────────────────────────────────────────────────────────────
HERO_HTML = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&display=swap');
*{box-sizing:border-box;margin:0;padding:0}
#hero{position:relative;width:100%;height:200px;background:#020208;overflow:hidden;border-bottom:1px solid rgba(0,200,255,0.12)}
#mc,#pc{position:absolute;inset:0;width:100%;height:100%}
.gbg{position:absolute;inset:0;background-image:linear-gradient(rgba(0,200,255,0.03) 1px,transparent 1px),linear-gradient(90deg,rgba(0,200,255,0.03) 1px,transparent 1px);background-size:44px 44px}
.vig{position:absolute;inset:0;background:radial-gradient(ellipse at 50% 50%,transparent 20%,rgba(2,2,8,0.88) 100%)}
.ctl,.ctr,.cbl,.cbr{position:absolute;width:34px;height:34px}
.ctl{top:0;left:0;border-top:1px solid rgba(0,200,255,.45);border-left:1px solid rgba(0,200,255,.45)}
.ctr{top:0;right:0;border-top:1px solid rgba(0,200,255,.45);border-right:1px solid rgba(0,200,255,.45)}
.cbl{bottom:0;left:0;border-bottom:1px solid rgba(0,200,255,.45);border-left:1px solid rgba(0,200,255,.45)}
.cbr{bottom:0;right:0;border-bottom:1px solid rgba(0,200,255,.45);border-right:1px solid rgba(0,200,255,.45)}
.status{position:absolute;top:12px;right:14px;display:flex;gap:8px;align-items:center;z-index:10}
.dot{width:7px;height:7px;border-radius:50%;background:#00ff88;box-shadow:0 0 8px #00ff88;animation:bp 1.8s infinite}
@keyframes bp{0%,100%{opacity:1}50%{opacity:.15}}
.stxt{font-family:'Share Tech Mono',monospace;font-size:9px;color:rgba(0,200,255,.5);letter-spacing:.12em}
.clk{font-family:'Orbitron',monospace;font-size:10px;color:rgba(0,200,255,.65);letter-spacing:.1em}
.hc{position:absolute;bottom:16px;left:22px;z-index:10}
.htag{font-family:'Share Tech Mono',monospace;font-size:9px;letter-spacing:.2em;color:rgba(0,220,255,.65);border:1px solid rgba(0,220,255,.25);padding:2px 9px;border-radius:2px;display:inline-block;margin-bottom:8px;position:relative}
.htag::before{content:'';position:absolute;left:-1px;top:-1px;width:5px;height:5px;border-top:1px solid #00dcff;border-left:1px solid #00dcff}
.htag::after{content:'';position:absolute;right:-1px;bottom:-1px;width:5px;height:5px;border-bottom:1px solid #00dcff;border-right:1px solid #00dcff}
.hh1{font-family:'Orbitron',monospace;font-size:2rem;font-weight:900;line-height:1.05;color:#fff}
.hh1 span{color:#00dcff;animation:glow 2.5s ease-in-out infinite alternate}
@keyframes glow{from{text-shadow:0 0 8px rgba(0,200,255,.3)}to{text-shadow:0 0 28px rgba(0,200,255,.9),0 0 55px rgba(0,200,255,.3)}}
.hsub{font-family:'Share Tech Mono',monospace;font-size:.52rem;color:rgba(0,200,255,.28);letter-spacing:.12em;margin-top:5px}
</style>
<div id="hero">
  <canvas id="mc"></canvas><canvas id="pc"></canvas>
  <div class="gbg"></div><div class="vig"></div>
  <div class="ctl"></div><div class="ctr"></div><div class="cbl"></div><div class="cbr"></div>
  <div class="status">
    <div class="dot"></div><span class="stxt">ONLINE</span>
    <span class="clk" id="clk">--:-- --</span>
  </div>
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

# ── PIPELINE HTML ─────────────────────────────────────────────────────────────
def pipeline_html():
    steps = [("📄","pdf upload"),("T","text+ocr"),("⬡","chunking"),("⊛","embeddings"),("◈","faiss idx"),("◎","retrieval"),("AI","claude ai"),("💬","answers")]
    nodes = ""
    for i,(icon,label) in enumerate(steps):
        nodes += f'<div class="pn" id="pn{i}"><div class="pn-icon">{icon}</div><div class="pn-lbl">{label}</div></div>'
        if i < len(steps)-1:
            nodes += f'<div class="p-arrow" id="pa{i}">→</div>'
    return f"""
<div class="pipeline-bar">
  <div class="pipeline-label">// rag pipeline</div>
  <div class="pipeline-nodes">{nodes}</div>
</div>
<script>
(function(){{
  var step=0,total=8;
  function tick(){{
    if(step>=total)return;
    var n=document.getElementById('pn'+step);
    if(!n)return;
    n.className='pn active';
    if(step>0){{var a=document.getElementById('pa'+(step-1));if(a)a.className='p-arrow lit';}}
    var s=step;
    setTimeout(function(){{
      var nn=document.getElementById('pn'+s);if(nn)nn.className='pn done';
      step++;setTimeout(tick,90);
    }},550);
  }}
  setTimeout(tick,300);
}})();
</script>
"""

# ── SUGGESTION BANNER ─────────────────────────────────────────────────────────
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

# ── THINKING ANIMATION ────────────────────────────────────────────────────────
THINKING_HTML = """
<div class="claude-thinking">
  <div class="think-line">> retrieving relevant chunks…</div>
  <div class="think-line">> ranking relevance…</div>
  <div class="think-line">> generating answer…</div>
  <div class="think-bar"><div class="think-bar-fill"></div></div>
</div>
"""

def retrieval_html(chunk_indices):
    lines = "".join(f'<div class="rc-line">› Chunk {i} ✓</div>' for i in chunk_indices)
    return f'<div class="retrieved-chunks">{lines}</div>'

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
    if not image_store: return []
    q = question.lower()
    ordinals = {"1st":0,"first":0,"2nd":1,"second":1,"3rd":2,"third":2,"4th":3,"fourth":3,"5th":4,"fifth":4}
    for word,idx in ordinals.items():
        if word in q and idx < len(image_store): return [image_store[idx]]
    for n in re.findall(r'\b(\d+)\b', q):
        idx = int(n)-1
        if 0 <= idx < len(image_store): return [image_store[idx]]
    if "last" in q: return [image_store[-1]]
    if "all"  in q: return image_store
    if any(t in q for t in ["image","figure","picture","diagram","chart","graph"]): return [image_store[0]]
    return []

def fmt_time(seconds):
    return f"{seconds:.1f}s" if seconds < 60 else f"{int(seconds//60)}m {seconds%60:.0f}s"

def local_time():
    return datetime.now().strftime("%I:%M %p").lstrip("0")

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

    # ── UPLOAD PANEL ──
    panel_open = st.session_state.get("panel_open", True)
    toggle_cls = "upload-toggle" + ("" if panel_open else " closed")
    toggle_text = "▲ HIDE UPLOAD" if panel_open else "▼ OPEN UPLOAD PANEL"
    st.markdown(f'<div class="{toggle_cls}">{toggle_text}</div>', unsafe_allow_html=True)

    if st.button("▲ HIDE" if panel_open else "▼ OPEN UPLOAD", key="tog"):
        st.session_state.panel_open = not panel_open
        st.rerun()

    if panel_open:
        st.markdown('<span class="s-label" style="margin-top:4px;">// Drop your PDF</span>', unsafe_allow_html=True)
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

                STEP_LABELS = [
                    "extracting text + OCR",
                    "analyzing images with Claude",
                    "chunking document",
                    "generating embeddings",
                    "building FAISS index",
                    "ready ✓"
                ]

                def show_proc(step, pct, pages=0, chunks=0, imgs=0, vecs=0):
                    rows = ""
                    for i, lbl in enumerate(STEP_LABELS):
                        if i < step:   cls,ind = "ok",  "✓"
                        elif i == step: cls,ind = "live","●"
                        else:           cls,ind = "dim", "○"
                        rows += f'<div class="proc-step-row {cls}"><span>{ind}</span>> {lbl}</div>'
                    counters = ""
                    if pct >= 80:
                        counters = f"""<div class="proc-counters">
                          <div><div class="pc-val">{pages}</div><div class="pc-lbl">pages</div></div>
                          <div><div class="pc-val">{chunks}</div><div class="pc-lbl">chunks</div></div>
                          <div><div class="pc-val">{imgs}</div><div class="pc-lbl">images</div></div>
                          <div><div class="pc-val">{vecs}</div><div class="pc-lbl">vectors</div></div>
                        </div>"""
                    title_map = ["EXTRACTING TEXT","ANALYZING IMAGES","CHUNKING","EMBEDDING","BUILDING INDEX","COMPLETE ✓"]
                    title = title_map[min(step, len(title_map)-1)]
                    proc_ph.markdown(f"""
                    <div class="proc-panel">
                      <div class="proc-panel-title">⬡ {title} — {uploaded_file.name}</div>
                      <div class="proc-bar-wrap"><div class="proc-bar-fill" style="width:{pct}%"></div></div>
                      {rows}
                      {counters}
                    </div>""", unsafe_allow_html=True)

                show_proc(0, 5)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                doc = fitz.open(tmp_path)
                full_text, total = "", len(doc)
                image_store, img_global_idx = [], 0

                show_proc(1, 15)
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
                            image_store.append({"b64":b64,"media_type":mt,"page":pnum+1,"index":img_global_idx,"description":desc})
                            img_global_idx += 1
                            full_text += f"\n[Image {img_global_idx} on page {pnum+1}]: {desc}\n"
                        except Exception:
                            continue
                    show_proc(1, 15 + int((pnum+1)/total*40))

                doc.close()
                os.unlink(tmp_path)
                t_process = time.time() - t_start

                show_proc(2, 58)
                chunks = extract_chunks(full_text)
                time.sleep(0.2)

                show_proc(3, 70)
                t_embed = time.time()
                embs = np.array(model.encode(chunks)).astype('float32')

                show_proc(4, 88, pages=total, chunks=len(chunks), imgs=len(image_store), vecs=len(chunks))
                index = faiss.IndexFlatL2(embs.shape[1])
                index.add(embs)
                t_embed_done = time.time() - t_embed
                time.sleep(0.3)

                show_proc(5, 100, pages=total, chunks=len(chunks), imgs=len(image_store), vecs=len(chunks))
                time.sleep(0.5)
                proc_ph.empty()

                st.session_state.update({
                    "index": index, "chunks": chunks,
                    "messages": [], "chat_display": [],
                    "chunk_count": len(chunks),
                    "image_store": image_store, "image_count": len(image_store),
                    "page_count": total,
                    "process_time": t_process, "embed_time": t_embed_done,
                })
                st.success(f"✓ {len(chunks)} chunks · {len(image_store)} imgs · {fmt_time(t_process)}")
    else:
        uploaded_file = None

    st.markdown('<hr class="s-divider">', unsafe_allow_html=True)

    # ── METRICS ──
    chunk_count = st.session_state.get("chunk_count", 0)
    img_count   = st.session_state.get("image_count", 0)
    page_count  = st.session_state.get("page_count",  0)
    proc_time   = st.session_state.get("process_time", None)
    embed_time  = st.session_state.get("embed_time",   None)
    proc_disp   = fmt_time(proc_time)  if proc_time  else "—"
    embed_disp  = fmt_time(embed_time) if embed_time else "—"

    st.markdown(f"""
    <span class="s-label">// Last Run</span>
    <div class="metric-grid">
      <div class="metric-card"><div class="metric-val">{page_count or '—'}</div><div class="metric-lbl">Pages</div></div>
      <div class="metric-card"><div class="metric-val">{chunk_count or '—'}</div><div class="metric-lbl">Chunks</div></div>
      <div class="metric-card"><div class="metric-val">{img_count or '—'}</div><div class="metric-lbl">Images</div></div>
      <div class="metric-card"><div class="metric-val">{chunk_count or '—'}</div><div class="metric-lbl">Vectors</div></div>
      <div class="metric-card"><div class="metric-val">{proc_disp}</div><div class="metric-lbl">Process</div></div>
      <div class="metric-card"><div class="metric-val">{embed_disp}</div><div class="metric-lbl">Index</div></div>
    </div>
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

    # ── BUILT BY (below tips) ──
    st.markdown('<span class="s-label">// Built by</span>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'Share Tech Mono',monospace;font-size:0.68rem;
      color:rgba(0,200,255,0.65);margin-bottom:8px;letter-spacing:.02em;">
      Sai Jyothi Gayathri Adabala
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    show_email    = col1.button("✉ email",    key="btn_email")
    show_linkedin = col2.button("in linkedin", key="btn_li")

    if show_email:
        st.session_state["contact_show"] = "email" if st.session_state.get("contact_show") != "email" else None
    if show_linkedin:
        st.session_state["contact_show"] = "linkedin" if st.session_state.get("contact_show") != "linkedin" else None

    cs = st.session_state.get("contact_show")
    if cs == "email":
        st.markdown("""
        <div class="contact-card">
          <div class="c-name">✦ Sai Jyothi Gayathri Adabala</div>
          <div style="font-family:'Share Tech Mono',monospace;font-size:0.62rem;
            color:rgba(0,200,255,0.75);letter-spacing:.04em;">
            ✉ &nbsp;asjyothig@gmail.com
          </div>
          <div style="font-size:0.48rem;color:rgba(0,200,255,0.22);margin-top:5px;letter-spacing:.08em;">
            // copy and email me anytime
          </div>
        </div>
        """, unsafe_allow_html=True)
    elif cs == "linkedin":
        # Clickable button that opens LinkedIn directly
        st.markdown("""
        <a href="https://www.linkedin.com/in/sai-jyothi-gayathri-adabala-a41a9818b/"
           target="_blank"
           style="display:block;font-family:'Share Tech Mono',monospace;font-size:0.62rem;
             color:#00dcff;text-decoration:none;
             background:rgba(0,200,255,0.06);
             border:1px solid rgba(0,200,255,0.3);
             border-radius:8px;padding:10px 12px;margin-top:6px;
             animation:cardIn 0.35s ease;letter-spacing:.04em;">
          <div style="font-family:'Orbitron',monospace;font-size:0.6rem;margin-bottom:5px;
            animation:nameGlow 2.5s ease-in-out infinite alternate;">
            ✦ Sai Jyothi Gayathri Adabala
          </div>
          in &nbsp;→ Open LinkedIn Profile
        </a>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:12px;font-family:'Share Tech Mono',monospace;font-size:7px;
      color:rgba(0,200,255,0.1);letter-spacing:.08em;text-align:center;
      border-top:1px solid rgba(0,200,255,0.05);padding-top:8px;">
      CLAUDE · FAISS · SENTENCE-TRANSFORMERS
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
st.components.v1.html(HERO_HTML, height=202, scrolling=False)

if "index" not in st.session_state:
    st.markdown("""
    <div class="empty-state">
      <span class="empty-glyph">⬡</span>
      <h2>UPLOAD A DOCUMENT TO BEGIN</h2>
      <p>// open the sidebar → drop a PDF → ask anything</p>
      <div class="empty-flow">
        <span>PDF</span><span class="ef-arrow">→</span>
        <span>TEXT</span><span class="ef-arrow">→</span>
        <span>IMAGES</span><span class="ef-arrow">→</span>
        <span>AI</span><span class="ef-arrow">→</span>
        <span>ANSWERS</span>
      </div>
    </div>
    """, unsafe_allow_html=True)
else:
    if "chat_display" not in st.session_state:
        st.session_state.chat_display = []
    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.components.v1.html(pipeline_html(), height=70, scrolling=False)
    st.markdown(CHAT_BANNER_HTML, unsafe_allow_html=True)
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    for entry in st.session_state.chat_display:
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"])
            side = "ts-right" if entry["role"] == "user" else "ts-left"
            tick = "✓" if entry["role"] == "user" else "⬡"
            st.markdown(f'<span class="ts {side}">{tick} {entry.get("time","")}</span>', unsafe_allow_html=True)
            for img_data in entry.get("images", []):
                st.markdown(
                    f'<div class="img-frame"><div class="img-frame-label">'
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

        images_to_show = find_matching_images(question, st.session_state.get("image_store", [])) if is_image_request(question) else []

        think_ph = st.empty()
        think_ph.markdown(THINKING_HTML, unsafe_allow_html=True)

        q_emb = np.array([model.encode(question)]).astype('float32')
        _, idxs = st.session_state.index.search(q_emb, 3)
        context = "\n\n".join(st.session_state.chunks[i] for i in idxs[0])

        think_ph.markdown(retrieval_html([int(i)+1 for i in idxs[0]]), unsafe_allow_html=True)
        time.sleep(0.4)
        think_ph.empty()

        if images_to_show:
            st.session_state.messages.append({"role":"user","content":
                f"Context:\n{context}\n\nQuestion: {question}\n\nNote: You ARE showing the user the image(s). Describe in 1-2 sentences."})
            system_prompt = "You are a document assistant. The user asked to see an image and you ARE displaying it. Describe what the image shows in 1-2 sentences. Never say you cannot show images."
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
                    f'<div class="img-frame"><div class="img-frame-label">'
                    f'<span>// IMAGE {img_data["index"]+1}</span>'
                    f'<span>PAGE {img_data["page"]}</span></div></div>',
                    unsafe_allow_html=True)
                st.image(base64.b64decode(img_data["b64"]), use_column_width=False)