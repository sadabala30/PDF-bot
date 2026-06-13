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

# ── FOCUS-RING KILLER ─────────────────────────────────────────────────────────
FOCUS_KILL = """
<style>
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

/* ── SIDEBAR TOGGLE ARROW — super visible */
[data-testid="collapsedControl"] {
  display: flex !important;
  visibility: visible !important;
  opacity: 1 !important;
  position: fixed !important;
  left: 0 !important;
  top: 50% !important;
  transform: translateY(-50%) !important;
  z-index: 9999 !important;
  width: 28px !important;
  height: 72px !important;
  background: rgba(0,200,255,0.18) !important;
  border: 1.5px solid rgba(0,200,255,0.7) !important;
  border-left: none !important;
  border-radius: 0 12px 12px 0 !important;
  box-shadow: 0 0 24px rgba(0,200,255,0.5), 4px 0 20px rgba(0,200,255,0.3) !important;
  animation: sidebarPulse 2s ease-in-out infinite !important;
  align-items: center !important;
  justify-content: center !important;
  cursor: pointer !important;
}
[data-testid="collapsedControl"] svg {
  color: #00dcff !important;
  width: 18px !important;
  height: 18px !important;
  filter: drop-shadow(0 0 6px rgba(0,200,255,0.9)) !important;
}
@keyframes sidebarPulse {
  0%,100% { box-shadow: 0 0 16px rgba(0,200,255,0.4), 4px 0 14px rgba(0,200,255,0.2); background: rgba(0,200,255,0.14); }
  50%      { box-shadow: 0 0 32px rgba(0,200,255,0.8), 4px 0 28px rgba(0,200,255,0.5); background: rgba(0,200,255,0.28); }
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

/* ── WHATSAPP-STYLE CHAT ── */
[data-testid="stChatMessage"] {
  background: transparent !important;
  border: none !important;
  padding: 3px 20px !important;
  margin-bottom: 2px !important;
}
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
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid*="Avatar"],
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [class*="avatar"] {
  background: linear-gradient(135deg,#002a40,#005580) !important;
  border: 2px solid rgba(0,200,255,0.55) !important;
  box-shadow: 0 0 12px rgba(0,200,255,0.3) !important;
  border-radius: 50% !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid*="Avatar"],
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [class*="avatar"] {
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
  transition: border-color 0.25s, box-shadow 0.25s !important;
  outline: none !important;
  box-shadow: none !important;
}
[data-testid="stChatInput"] textarea:focus,
[data-testid="stChatInput"] textarea:focus-visible {
  outline: none !important;
  border-color: rgba(0,210,255,0.85) !important;
  box-shadow: 0 0 0 2px rgba(0,200,255,0.2), 0 0 30px rgba(0,200,255,0.25) !important;
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
.metric-val  { font-family: 'Orbitron', monospace; font-size: 0.85rem; font-weight: 700; color: #00dcff; }
.metric-lbl  { font-family: 'Share Tech Mono', monospace; font-size: 0.55rem; color: rgba(0,200,255,0.35); margin-top: 3px; letter-spacing: .06em; }
.sys-tips    { font-family: 'Share Tech Mono', monospace; font-size: 0.62rem; color: rgba(0,200,255,0.3); line-height: 2.1; letter-spacing: .04em; }
.s-divider   { border: none; border-top: 1px solid rgba(0,200,255,0.07); margin: 10px 0; }
.s-label     { font-family: 'Share Tech Mono', monospace; font-size: 0.58rem; letter-spacing: .18em; color: rgba(0,200,255,0.28); text-transform: uppercase; margin-bottom: 6px; display: block; }

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

/* ── CONTACT CARD REVEAL ── */
.contact-card {
  background: rgba(0,200,255,0.04);
  border: 1px solid rgba(0,200,255,0.15);
  border-radius: 8px;
  padding: 10px 12px;
  margin-top: 6px;
  animation: contactReveal 0.4s ease;
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.62rem;
  line-height: 2;
}
@keyframes contactReveal {
  from { opacity: 0; transform: translateY(-6px); }
  to   { opacity: 1; transform: translateY(0); }
}
.contact-card .c-name {
  font-family: 'Orbitron', monospace;
  font-size: 0.65rem;
  color: #00dcff;
  letter-spacing: .08em;
  margin-bottom: 6px;
  animation: nameGlow 2.5s ease-in-out infinite alternate;
}
@keyframes nameGlow {
  from { text-shadow: 0 0 6px rgba(0,200,255,0.3); }
  to   { text-shadow: 0 0 18px rgba(0,200,255,0.9), 0 0 36px rgba(0,200,255,0.3); }
}
.contact-card a { color: rgba(0,200,255,0.7); text-decoration: none; }
.contact-card a:hover { color: #00dcff; }

/* ── UPLOAD TOGGLE ── */
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
  animation: pulse-btn 2s ease-in-out infinite;
}
@keyframes pulse-btn {
  0%,100% { box-shadow: 0 0 10px rgba(0,200,255,0.18); }
  50%      { box-shadow: 0 0 26px rgba(0,200,255,0.45); }
}

/* ── PIPELINE BAR ── */
.pipeline-bar {
  background: rgba(0,2,14,0.8);
  border-bottom: 1px solid rgba(0,200,255,0.08);
  padding: 8px 16px 6px;
}
.pipeline-label {
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.5rem; letter-spacing: .18em;
  color: rgba(0,200,255,0.25); text-transform: uppercase;
  margin-bottom: 6px;
}
.pipeline-nodes {
  display: flex; align-items: center; gap: 0;
  overflow-x: auto;
}
.pipeline-nodes::-webkit-scrollbar { height: 2px; }
.pipeline-nodes::-webkit-scrollbar-thumb { background: rgba(0,200,255,0.2); }
.pn {
  display: flex; flex-direction: column; align-items: center;
  gap: 3px; min-width: 58px; cursor: pointer; transition: all 0.25s;
}
.pn-icon {
  width: 26px; height: 26px; border-radius: 6px;
  border: 1px solid rgba(0,200,255,0.15);
  display: flex; align-items: center; justify-content: center;
  font-size: 9px; background: rgba(0,200,255,0.03);
  transition: all 0.3s; position: relative;
  font-family: 'Share Tech Mono', monospace;
  color: rgba(0,200,255,0.35);
}
.pn-lbl {
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.38rem; color: rgba(0,200,255,0.25);
  letter-spacing: .06em; text-align: center;
}
.pn.active .pn-icon {
  border-color: rgba(0,200,255,0.85);
  background: rgba(0,200,255,0.15);
  box-shadow: 0 0 14px rgba(0,200,255,0.35);
  color: #00dcff;
  animation: nodeActive 0.6s ease-in-out infinite alternate;
}
.pn.active .pn-lbl { color: rgba(0,200,255,0.85); }
.pn.done .pn-icon {
  border-color: rgba(0,255,136,0.5);
  background: rgba(0,255,136,0.06);
  color: #00ff88;
}
.pn.done .pn-lbl { color: rgba(0,255,136,0.6); }
@keyframes nodeActive {
  from { box-shadow: 0 0 8px rgba(0,200,255,0.25); }
  to   { box-shadow: 0 0 20px rgba(0,200,255,0.6); }
}
.p-arrow { color: rgba(0,200,255,0.15); font-size: 0.6rem; padding: 0 1px; margin-bottom: 12px; flex-shrink: 0; transition: color 0.4s; }
.p-arrow.lit { color: rgba(0,200,255,0.75); text-shadow: 0 0 8px rgba(0,200,255,0.5); }

/* ── PROCESSING OVERLAY ── */
.proc-overlay {
  position: fixed; inset: 0;
  background: rgba(0,2,14,0.92);
  display: flex; align-items: center; justify-content: center;
  z-index: 9999; flex-direction: column; gap: 20px;
}
.proc-box {
  width: 340px; padding: 28px;
  background: rgba(0,10,30,0.97);
  border: 1px solid rgba(0,200,255,0.25);
  border-radius: 14px;
  display: flex; flex-direction: column; gap: 14px;
}
.proc-title {
  font-family: 'Orbitron', monospace; font-size: 0.78rem;
  color: #00dcff; letter-spacing: .1em; text-align: center;
}
.pdf-scan-wrap {
  display: flex; align-items: center; gap: 14px;
}
.pdf-icon-anim {
  position: relative; width: 44px; height: 52px;
  display: flex; align-items: center; justify-content: center;
  flex-shrink: 0;
}
.pdf-icon-anim .pdf-glyph {
  font-family: 'Orbitron', monospace; font-size: 1.5rem;
  color: #e85038; text-shadow: 0 0 20px rgba(232,80,56,0.4);
}
.pdf-icon-anim .scan-line {
  position: absolute; left: 0; right: 0; height: 1px;
  background: linear-gradient(90deg, transparent, #00dcff, transparent);
  animation: scandown 1.4s linear infinite;
}
@keyframes scandown { 0%{top:0;opacity:1} 100%{top:100%;opacity:0} }
.proc-step {
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.6rem; color: rgba(0,200,255,0.22);
  letter-spacing: .08em;
  display: flex; align-items: center; gap: 8px;
  transition: all 0.5s;
}
.proc-step.active { color: rgba(0,200,255,0.9); }
.proc-step.done   { color: rgba(0,255,136,0.75); }
.proc-step .pind  {
  width: 12px; height: 12px; border-radius: 50%;
  border: 1px solid currentColor; flex-shrink: 0;
  display: flex; align-items: center; justify-content: center;
  font-size: 7px;
}
.proc-step.active .pind { animation: procblink 0.8s infinite; }
@keyframes procblink { 0%,100%{opacity:1} 50%{opacity:0.2} }
.proc-bar-wrap {
  background: rgba(0,200,255,0.08); border-radius: 2px; height: 2px; overflow: hidden;
}
.proc-bar {
  height: 2px;
  background: linear-gradient(90deg,#00dcff,#00ff88);
  border-radius: 2px; transition: width 0.4s ease;
}
.chunk-visual {
  display: flex; flex-direction: column; gap: 3px; margin-top: 2px;
}
.chunk-block {
  height: 4px; background: rgba(0,200,255,0.28); border-radius: 2px;
  animation: chunkgrow 0.45s ease forwards;
}
@keyframes chunkgrow { from{width:0} to{width:var(--w)} }
.proc-counters {
  display: grid; grid-template-columns: repeat(4,1fr); gap: 8px; margin-top: 4px;
}
.proc-counter { text-align: center; }
.pc-val { font-family: 'Orbitron', monospace; font-size: 0.82rem; color: #00dcff; }
.pc-lbl { font-family: 'Share Tech Mono', monospace; font-size: 0.45rem; color: rgba(0,200,255,0.3); margin-top: 2px; }

/* ── EMPTY STATE ── */
.empty-state { text-align: center; padding: 80px 32px; }
.empty-glyph { font-size: 3rem; color: rgba(0,200,255,0.12); margin-bottom: 20px; display: block; font-family: 'Orbitron', monospace; animation: pg 3s ease-in-out infinite; }
@keyframes pg { 0%,100%{opacity:.12;} 50%{opacity:.3;} }
.empty-state h2 { font-family: 'Orbitron', monospace; font-size: 0.95rem; font-weight: 400; color: rgba(0,200,255,0.2); letter-spacing: .12em; }
.empty-state p  { font-family: 'Share Tech Mono', monospace; font-size: 0.63rem; color: rgba(0,200,255,0.12); margin-top: 10px; letter-spacing: .08em; }
.empty-flow {
  display: flex; align-items: center; justify-content: center;
  gap: 8px; margin-top: 16px;
  font-family: 'Share Tech Mono', monospace; font-size: 0.58rem;
  color: rgba(0,200,255,0.12); letter-spacing: .1em;
}
.ef-arrow { animation: flowarrow 2s ease-in-out infinite; }
.ef-arrow:nth-child(2) { animation-delay: .3s; }
.ef-arrow:nth-child(4) { animation-delay: .6s; }
.ef-arrow:nth-child(6) { animation-delay: .9s; }
@keyframes flowarrow { 0%,100%{opacity:.12} 50%{opacity:.75} }

/* ── IMAGE FRAME ── */
.img-frame       { border: 1px solid rgba(0,200,255,0.2); border-radius: 8px; overflow: hidden; margin: 10px 0; background: #000; }
.img-frame-label { font-family: 'Share Tech Mono', monospace; font-size: 0.6rem; color: rgba(0,200,255,0.5); padding: 6px 12px; border-bottom: 1px solid rgba(0,200,255,0.1); letter-spacing: .1em; display: flex; justify-content: space-between; }

/* ── TIMESTAMPS ── */
.ts       { font-family: 'Share Tech Mono', monospace; font-size: 0.57rem; color: rgba(0,200,255,0.2); margin-top: 5px; display: block; }
.ts-right { text-align: right; }
.ts-left  { text-align: left; }

/* ── SUGGESTION TAGS ── */
.sug-bar { display: flex; gap: 7px; align-items: center; padding: 6px 20px; border-top: 1px solid rgba(0,200,255,0.07); background: rgba(0,2,14,0.72); overflow-x: auto; }
.sug-bar::-webkit-scrollbar { height: 2px; }
.sug-prefix { font-family: 'Share Tech Mono', monospace; font-size: 0.5rem; color: rgba(0,200,255,0.3); letter-spacing: .14em; white-space: nowrap; }
.sug-tag {
  font-family: 'Share Tech Mono', monospace; font-size: 0.55rem;
  color: rgba(0,200,255,0.35); border: 1px solid rgba(0,200,255,0.12);
  padding: 3px 10px; border-radius: 20px; white-space: nowrap; cursor: pointer;
  transition: all 0.2s; animation: taganim 3s ease-in-out infinite;
}
.sug-tag:nth-child(3) { animation-delay: .4s; }
.sug-tag:nth-child(4) { animation-delay: .8s; }
.sug-tag:nth-child(5) { animation-delay: 1.2s; }
.sug-tag:nth-child(6) { animation-delay: 1.6s; }
.sug-tag:hover { border-color: rgba(0,200,255,0.55); color: rgba(0,220,255,0.8); }
@keyframes taganim { 0%,100%{opacity:.35} 50%{opacity:.9;border-color:rgba(0,200,255,.42)} }

/* ── CLAUDE THINKING ── */
.claude-thinking {
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.62rem; color: rgba(0,200,255,0.5);
  padding: 10px 14px;
  background: rgba(0,18,38,0.6);
  border: 1px solid rgba(0,200,255,0.12);
  border-radius: 8px; margin: 4px 20px;
  line-height: 2;
}
.claude-thinking .think-line { animation: thinkfade 1.2s ease-in-out infinite; }
.claude-thinking .think-line:nth-child(2) { animation-delay: .2s; }
.claude-thinking .think-line:nth-child(3) { animation-delay: .4s; }
@keyframes thinkfade { 0%,100%{opacity:.3} 50%{opacity:1} }
.think-bar {
  margin-top: 8px; height: 2px;
  background: rgba(0,200,255,0.08); border-radius: 2px; overflow: hidden;
}
.think-bar-fill {
  height: 2px; width: 0%;
  background: linear-gradient(90deg,#00dcff,#00ff88);
  border-radius: 2px;
  animation: thinkprog 2s ease-in-out infinite;
}
@keyframes thinkprog { 0%{width:0%} 70%{width:90%} 100%{width:100%} }

/* ── RETRIEVAL CHUNKS ── */
.retrieved-chunks {
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.55rem; color: rgba(0,200,255,0.4);
  padding: 6px 14px; letter-spacing: .06em;
  line-height: 1.9;
}
.rc-line { animation: rcfade 0.4s ease forwards; opacity: 0; }
.rc-line:nth-child(1) { animation-delay: 0s; }
.rc-line:nth-child(2) { animation-delay: .15s; }
.rc-line:nth-child(3) { animation-delay: .3s; }
@keyframes rcfade { to { opacity: 1; } }
</style>
""", unsafe_allow_html=True)

# ── HERO ──────────────────────────────────────────────────────────────────────
HERO_HTML = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&display=swap');
*{box-sizing:border-box;margin:0;padding:0;}
#hero{position:relative;width:100%;height:200px;background:#020208;overflow:hidden;border-bottom:1px solid rgba(0,200,255,0.12);}
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
.hh1{font-family:'Orbitron',monospace;font-size:2rem;font-weight:900;line-height:1.05;
  color:#fff;text-shadow:0 0 50px rgba(0,200,255,0.3);}
.hh1 span{color:#00dcff;animation:glow 2.5s ease-in-out infinite alternate;}
@keyframes glow{from{text-shadow:0 0 8px rgba(0,200,255,0.3);}
  to{text-shadow:0 0 28px rgba(0,200,255,0.9),0 0 55px rgba(0,200,255,0.3);}}
.hsub{font-family:'Share Tech Mono',monospace;font-size:.55rem;
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
  function tick(){
    var n=new Date(),e=document.getElementById('clk');if(!e)return;
    var h=n.getHours(),m=n.getMinutes(),ap=h>=12?'PM':'AM';
    h=h%12||12;
    e.textContent=(h<10?'0'+h:h)+':'+(m<10?'0'+m:m)+' '+ap;
  }
  setInterval(tick,1000);tick();

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

# ── PIPELINE BAR HTML ─────────────────────────────────────────────────────────
def pipeline_html(active_step=None, done_steps=None):
    """Render the animated RAG pipeline bar."""
    done_steps = done_steps or []
    steps = [
        ("📄","pdf upload"),
        ("T","text+ocr"),
        ("⬡","chunking"),
        ("⊛","embeddings"),
        ("◈","faiss idx"),
        ("◎","retrieval"),
        ("AI","claude ai"),
        ("💬","answers"),
    ]
    nodes_html = ""
    for i, (icon, label) in enumerate(steps):
        cls = "pn"
        if i in done_steps:
            cls += " done"
        elif i == active_step:
            cls += " active"
        nodes_html += f'<div class="{cls}"><div class="pn-icon">{icon}</div><div class="pn-lbl">{label}</div></div>'
        if i < len(steps) - 1:
            arrow_cls = "p-arrow lit" if i in done_steps else "p-arrow"
            nodes_html += f'<div class="{arrow_cls}">→</div>'

    return f"""
<div class="pipeline-bar">
  <div class="pipeline-label">// rag pipeline</div>
  <div class="pipeline-nodes" id="pipeline-nodes">
    {nodes_html}
  </div>
</div>
<script>
(function(){{
  var nodes = document.querySelectorAll('.pn');
  var arrows = document.querySelectorAll('.p-arrow');
  var step = 0;
  function animate() {{
    nodes.forEach(function(n, i) {{ n.className = 'pn'; }});
    arrows.forEach(function(a) {{ a.className = 'p-arrow'; }});
    function tick() {{
      if (step >= nodes.length) return;
      nodes[step].className = 'pn active';
      if (step > 0) arrows[step-1].className = 'p-arrow lit';
      var s = step;
      setTimeout(function() {{
        nodes[s].className = 'pn done';
        step++;
        setTimeout(tick, 80);
      }}, 550);
    }}
    tick();
  }}
  animate();
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

# ── PROCESSING OVERLAY ────────────────────────────────────────────────────────
def processing_overlay_html(filename, step_idx, progress_pct, pages=0, chunks=0, imgs=0, vecs=0):
    steps = [
        "extracting text + OCR",
        "analyzing images with Claude",
        "chunking document",
        "generating embeddings",
        "building FAISS index",
        "ready for queries ✓",
    ]
    steps_html = ""
    for i, s in enumerate(steps):
        if i < step_idx:
            cls = "proc-step done"
            ind = "✓"
        elif i == step_idx:
            cls = "proc-step active"
            ind = "●"
        else:
            cls = "proc-step"
            ind = "○"
        steps_html += f'<div class="{cls}"><div class="pind">{ind}</div>> {s}</div>'

    chunk_vis = ""
    if step_idx >= 2:
        widths = ["100%","78%","55%","88%","62%","92%","45%"]
        for w in widths[:5]:
            chunk_vis += f'<div class="chunk-block" style="--w:{w}"></div>'

    counter_html = ""
    if step_idx >= 4:
        counter_html = f"""
        <div class="proc-counters">
          <div class="proc-counter"><div class="pc-val">{pages}</div><div class="pc-lbl">pages</div></div>
          <div class="proc-counter"><div class="pc-val">{chunks}</div><div class="pc-lbl">chunks</div></div>
          <div class="proc-counter"><div class="pc-val">{imgs}</div><div class="pc-lbl">images</div></div>
          <div class="proc-counter"><div class="pc-val">{vecs}</div><div class="pc-lbl">vectors</div></div>
        </div>
        """

    title_map = ["EXTRACTING TEXT","ANALYZING IMAGES","CHUNKING DOCUMENT","GENERATING EMBEDDINGS","BUILDING FAISS INDEX","COMPLETE ✓"]
    title = title_map[min(step_idx, len(title_map)-1)]

    return f"""
<div class="proc-overlay" id="proc-overlay">
  <div class="proc-box">
    <div class="pdf-scan-wrap">
      <div class="pdf-icon-anim">
        <div class="pdf-glyph">⬡</div>
        <div class="scan-line"></div>
      </div>
      <div style="flex:1">
        <div class="proc-title">{title}</div>
        <div style="font-family:'Share Tech Mono',monospace;font-size:0.5rem;color:rgba(0,200,255,0.3);margin-top:3px;letter-spacing:.08em">{filename}</div>
      </div>
    </div>
    <div class="proc-bar-wrap">
      <div class="proc-bar" style="width:{progress_pct}%"></div>
    </div>
    <div>{steps_html}</div>
    {"<div class='chunk-visual'>" + chunk_vis + "</div>" if chunk_vis else ""}
    {counter_html}
  </div>
</div>
"""

# ── CLAUDE THINKING ANIMATION ─────────────────────────────────────────────────
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

    st.markdown("""
    <div style="padding:14px 14px 10px;border-bottom:1px solid rgba(0,200,255,0.1);margin-bottom:10px;">
      <div style="font-family:'Orbitron',monospace;font-size:0.88rem;font-weight:700;
        color:#00dcff;letter-spacing:.06em;">PDF BOT</div>
      <div style="font-family:'Share Tech Mono',monospace;font-size:0.54rem;
        color:rgba(0,200,255,0.22);letter-spacing:.14em;margin-top:2px;">INTELLIGENCE SYSTEM</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="s-divider">', unsafe_allow_html=True)

    # ── UPLOAD PANEL ──
    panel_open = st.session_state.get("panel_open", True)
    toggle_cls  = "upload-toggle" + ("" if panel_open else " closed")
    toggle_icon = "▲" if panel_open else "▼"
    toggle_text = "HIDE UPLOAD" if panel_open else "OPEN UPLOAD PANEL"
    st.markdown(f'<div class="{toggle_cls}">{toggle_icon}&nbsp;&nbsp;{toggle_text}&nbsp;&nbsp;{toggle_icon}</div>', unsafe_allow_html=True)

    if st.button("▲ HIDE" if panel_open else "▼ OPEN UPLOAD", key="tog"):
        st.session_state.panel_open = not panel_open
        st.rerun()

    if panel_open:
        st.markdown('<span class="s-label" style="margin-top:6px;">// Drop your PDF</span>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("", type="pdf", label_visibility="collapsed")

        if uploaded_file:
            st.markdown(f"""
            <div style="font-family:'Share Tech Mono',monospace;font-size:0.63rem;
              color:rgba(0,200,255,0.5);padding:6px 8px;
              border-left:2px solid rgba(0,200,255,0.3);margin-bottom:8px;word-break:break-all;">
              › {uploaded_file.name}
            </div>
            """, unsafe_allow_html=True)

            if st.button("⚡  PROCESS & INDEX"):
                t_start = time.time()

                # ── Step 0: show overlay ──
                proc_placeholder = st.empty()
                proc_placeholder.markdown(
                    processing_overlay_html(uploaded_file.name, 0, 0),
                    unsafe_allow_html=True
                )

                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                doc        = fitz.open(tmp_path)
                full_text  = ""
                total      = len(doc)
                image_store    = []
                img_global_idx = 0

                # ── Step 1: text + images ──
                proc_placeholder.markdown(
                    processing_overlay_html(uploaded_file.name, 1, 15),
                    unsafe_allow_html=True
                )

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

                    progress_pct = 15 + int((pnum+1)/total * 40)
                    proc_placeholder.markdown(
                        processing_overlay_html(uploaded_file.name, 1, progress_pct),
                        unsafe_allow_html=True
                    )

                doc.close()
                os.unlink(tmp_path)
                t_process = time.time() - t_start

                # ── Step 2: chunking ──
                proc_placeholder.markdown(
                    processing_overlay_html(uploaded_file.name, 2, 60),
                    unsafe_allow_html=True
                )
                chunks = extract_chunks(full_text)
                time.sleep(0.3)

                # ── Step 3: embeddings ──
                proc_placeholder.markdown(
                    processing_overlay_html(uploaded_file.name, 3, 72),
                    unsafe_allow_html=True
                )
                t_embed = time.time()
                embs    = np.array(model.encode(chunks)).astype('float32')

                # ── Step 4: FAISS ──
                proc_placeholder.markdown(
                    processing_overlay_html(uploaded_file.name, 4, 88,
                        pages=total, chunks=len(chunks), imgs=len(image_store), vecs=len(chunks)),
                    unsafe_allow_html=True
                )
                index = faiss.IndexFlatL2(embs.shape[1])
                index.add(embs)
                t_embed_done = time.time() - t_embed
                time.sleep(0.4)

                # ── Step 5: done ──
                proc_placeholder.markdown(
                    processing_overlay_html(uploaded_file.name, 5, 100,
                        pages=total, chunks=len(chunks), imgs=len(image_store), vecs=len(chunks)),
                    unsafe_allow_html=True
                )
                time.sleep(0.6)
                proc_placeholder.empty()

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

    # ── Metrics with live-counter feel ──
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
      <div class="metric-card"><div class="metric-val" id="mc-pages">{page_count if page_count else '—'}</div><div class="metric-lbl">Pages</div></div>
      <div class="metric-card"><div class="metric-val" id="mc-chunks">{chunk_count if chunk_count else '—'}</div><div class="metric-lbl">Chunks</div></div>
      <div class="metric-card"><div class="metric-val" id="mc-imgs">{img_count if img_count else '—'}</div><div class="metric-lbl">Images</div></div>
      <div class="metric-card"><div class="metric-val" id="mc-vecs">{chunk_count if chunk_count else '—'}</div><div class="metric-lbl">Vectors</div></div>
      <div class="metric-card"><div class="metric-val">{proc_disp}</div><div class="metric-lbl">Process</div></div>
      <div class="metric-card"><div class="metric-val">{embed_disp}</div><div class="metric-lbl">Index</div></div>
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
    """, unsafe_allow_html=True)

    st.markdown('<hr class="s-divider">', unsafe_allow_html=True)

    # ── BUILT BY — below tips ──
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
        # Open LinkedIn directly in new tab using an auto-click anchor
        st.markdown("""
        <script>window.open('https://www.linkedin.com/in/sai-jyothi-gayathri-adabala-a41a9818b/','_blank');</script>
        """, unsafe_allow_html=True)

    cs = st.session_state.get("contact_show")
    if cs == "email":
        st.markdown("""
        <div class="contact-card">
          <div class="c-name">✦ Sai Jyothi Gayathri Adabala</div>
          <div style="color:rgba(0,200,255,0.75);font-family:'Share Tech Mono',monospace;
            font-size:0.62rem;letter-spacing:.04em;">
            ✉ &nbsp;asjyothig@gmail.com
          </div>
          <div style="font-size:0.48rem;color:rgba(0,200,255,0.22);margin-top:5px;letter-spacing:.08em;">
            // copy and email me anytime
          </div>
        </div>
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
    # ── EMPTY STATE with animated flow ──
    st.markdown("""
    <div class="empty-state">
      <span class="empty-glyph">⬡</span>
      <h2>UPLOAD A DOCUMENT TO BEGIN</h2>
      <p>// open the sidebar → drop a PDF → ask anything</p>
      <div class="empty-flow">
        <span>PDF</span>
        <span class="ef-arrow">→</span>
        <span>TEXT</span>
        <span class="ef-arrow">→</span>
        <span>IMAGES</span>
        <span class="ef-arrow">→</span>
        <span>AI</span>
        <span class="ef-arrow">→</span>
        <span>ANSWERS</span>
      </div>
    </div>
    """, unsafe_allow_html=True)
else:
    if "chat_display" not in st.session_state:
        st.session_state.chat_display = []
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # ── Pipeline bar (animated on every page load) ──
    st.components.v1.html(pipeline_html(), height=72, scrolling=False)

    # ── Suggestion banner ──
    st.markdown(CHAT_BANNER_HTML, unsafe_allow_html=True)
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    # ── Render chat history ──
    for entry in st.session_state.chat_display:
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"])
            ts   = entry.get("time", "")
            side = "ts-right" if entry["role"] == "user" else "ts-left"
            tick = "✓" if entry["role"] == "user" else "⬡"
            st.markdown(f'<span class="ts {side}">{tick} {ts}</span>', unsafe_allow_html=True)
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

        # ── Retrieval with animated thinking ──
        think_placeholder = st.empty()
        think_placeholder.markdown(THINKING_HTML, unsafe_allow_html=True)

        q_emb = np.array([model.encode(question)]).astype('float32')
        _, idxs = st.session_state.index.search(q_emb, 3)
        context = "\n\n".join(st.session_state.chunks[i] for i in idxs[0])

        # ── Show retrieved chunks ──
        think_placeholder.markdown(
            retrieval_html([int(i)+1 for i in idxs[0]]),
            unsafe_allow_html=True
        )
        time.sleep(0.4)
        think_placeholder.empty()

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
            st.markdown(f'<span class="ts ts-left">⬡ {reply_time}</span>', unsafe_allow_html=True)
            if images_to_show:
                for img_data in images_to_show:
                    st.markdown(
                        f'<div class="img-frame"><div class="img-frame-label">'
                        f'<span>// IMAGE {img_data["index"]+1}</span>'
                        f'<span>PAGE {img_data["page"]}</span></div></div>',
                        unsafe_allow_html=True
                    )
                    st.image(base64.b64decode(img_data["b64"]), use_column_width=False)