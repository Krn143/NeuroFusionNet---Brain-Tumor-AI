"""
🧠 NeuroFusionNet — Premium Brain Tumor AI Dashboard
Segmentation → Classification → XAI → LLM Clinical Reports
"""

import os, sys, json, io
from pathlib import Path
import streamlit as st
import numpy as np
import torch
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import cv2
import pandas as pd

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from neurofusionnet.model import NeuroFusionNet, create_model
from neurofusionnet.dataset import get_eval_transforms, get_raw_transforms
from neurofusionnet.utils import CLASS_NAMES, NUM_CLASSES, TUMOR_INFO, get_device, count_parameters, load_checkpoint

# ── Page Config ──
st.set_page_config(page_title="NeuroFusionNet — Brain Tumor AI", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")
st.session_state["sidebar_state"] = "expanded"
# ── Premium CSS ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --bg-primary: #06060f; --bg-secondary: #0d0d1a; --bg-card: rgba(15,15,35,0.7);
  --accent-1: #6366f1; --accent-2: #8b5cf6; --accent-3: #a78bfa;
  --text-primary: #f1f5f9; --text-secondary: #94a3b8; --text-muted: #64748b;
  --border: rgba(99,102,241,0.15); --glow: rgba(99,102,241,0.25);
  --success: #22c55e; --warning: #f59e0b; --danger: #ef4444; --info: #3b82f6;
  --glass: rgba(15,15,40,0.6); --glass-border: rgba(99,102,241,0.2);
}

*, .stApp { font-family: 'Inter', -apple-system, sans-serif !important; }
.stApp { background: var(--bg-primary); background-image: radial-gradient(ellipse at 20% 50%, rgba(99,102,241,0.06) 0%, transparent 50%), radial-gradient(ellipse at 80% 20%, rgba(139,92,246,0.04) 0%, transparent 50%); }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0a0a1f 0%, #0d0d24 100%); border-right: 1px solid var(--border); }
#MainMenu, footer { visibility: hidden; }
header { background: transparent !important; }
[data-testid="stHeaderActionElements"] { display: none; }

/* Custom Premium Sidebar Toggle Buttons */
[data-testid="collapsedControl"] svg,
[data-testid="stSidebarCollapseButton"] svg,
[data-testid="collapsedControl"] span[data-testid="stIconMaterial"],
[data-testid="stSidebarCollapseButton"] span[data-testid="stIconMaterial"] {
  display: none !important;
}

[data-testid="collapsedControl"], 
[data-testid="stSidebarCollapseButton"] {
  background-color: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 50% !important;
  width: 38px !important;
  height: 38px !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  transition: all 0.3s ease !important;
  z-index: 999999 !important;
  box-shadow: 0 4px 12px rgba(0,0,0,0.5) !important;
}

[data-testid="collapsedControl"]:hover,
[data-testid="stSidebarCollapseButton"]:hover {
  background-color: rgba(99,102,241,0.2) !important;
  border-color: var(--accent-1) !important;
  transform: scale(1.05) !important;
}

/* Open Sidebar Arrow */
[data-testid="collapsedControl"]::before {
  content: "❯" !important;
  font-size: 1.2rem !important;
  font-weight: 800 !important;
  color: var(--accent-3) !important;
  line-height: 1 !important;
  margin-left: 3px !important;
}

/* Close Sidebar Arrow */
[data-testid="stSidebarCollapseButton"]::before {
  content: "❮" !important;
  font-size: 1.2rem !important;
  font-weight: 800 !important;
  color: var(--accent-3) !important;
  line-height: 1 !important;
  margin-right: 3px !important;
}

.hero-title { background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 30%, #a78bfa 60%, #c084fc 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 3.2rem; font-weight: 900; text-align: center; letter-spacing: -1.5px; line-height: 1.1; padding: 0.5rem 0; }
.hero-sub { color: var(--text-secondary); text-align: center; font-size: 1.05rem; font-weight: 400; margin-bottom: 2rem; letter-spacing: 0.5px; }

.glass-card { background: var(--glass); backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px); border: 1px solid var(--glass-border); border-radius: 16px; padding: 1.5rem; transition: all 0.4s cubic-bezier(0.16,1,0.3,1); position: relative; overflow: hidden; }
.glass-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px; background: linear-gradient(90deg, transparent, rgba(99,102,241,0.4), transparent); }
.glass-card:hover { border-color: rgba(99,102,241,0.35); box-shadow: 0 8px 32px var(--glow); transform: translateY(-3px); }

.stat-value { font-size: 2.2rem; font-weight: 800; background: linear-gradient(135deg, var(--accent-1), var(--accent-2)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; line-height: 1.2; }
.stat-label { color: var(--text-muted); font-size: 0.8rem; font-weight: 500; text-transform: uppercase; letter-spacing: 1px; margin-top: 0.3rem; }

.result-card { background: var(--glass); border: 1px solid var(--glass-border); border-radius: 16px; padding: 2rem; text-align: center; }
.result-emoji { font-size: 3.5rem; margin-bottom: 0.5rem; }
.result-class { font-size: 1.8rem; font-weight: 800; color: var(--success); }
.result-conf { color: var(--text-secondary); font-size: 1.1rem; margin-top: 0.25rem; }

.step-indicator { display: flex; justify-content: center; gap: 0; margin: 1.5rem 0; }
.step { display: flex; align-items: center; gap: 0.5rem; padding: 0.6rem 1.2rem; font-size: 0.85rem; font-weight: 600; color: var(--text-muted); background: var(--bg-card); border: 1px solid var(--border); }
.step:first-child { border-radius: 10px 0 0 10px; }
.step:last-child { border-radius: 0 10px 10px 0; }
.step.active { color: var(--accent-3); background: rgba(99,102,241,0.15); border-color: var(--accent-1); }
.step.done { color: var(--success); background: rgba(34,197,94,0.1); border-color: rgba(34,197,94,0.3); }

.divider { height: 1px; background: linear-gradient(90deg, transparent, var(--accent-1), transparent); margin: 2rem 0; opacity: 0.4; }

.report-container { background: rgba(10,10,30,0.8); border: 1px solid var(--glass-border); border-radius: 12px; padding: 1.5rem; font-family: 'Inter', sans-serif; color: var(--text-primary); line-height: 1.7; white-space: pre-wrap; }

.chat-user { background: rgba(99,102,241,0.12); border: 1px solid rgba(99,102,241,0.2); border-radius: 12px 12px 2px 12px; padding: 1rem 1.2rem; margin: 0.5rem 0; color: var(--text-primary); }
.chat-ai { background: rgba(34,197,94,0.08); border: 1px solid rgba(34,197,94,0.15); border-left: 3px solid var(--success); border-radius: 2px 12px 12px 12px; padding: 1rem 1.2rem; margin: 0.5rem 0; color: var(--text-primary); }

.feat-icon { font-size: 2rem; margin-bottom: 0.5rem; }
.feat-title { color: var(--text-primary); font-weight: 700; font-size: 1rem; margin: 0.4rem 0; }
.feat-desc { color: var(--text-muted); font-size: 0.85rem; line-height: 1.4; }

.tag { display: inline-block; background: rgba(99,102,241,0.15); border: 1px solid rgba(99,102,241,0.25); border-radius: 20px; padding: 0.2rem 0.8rem; margin: 0.15rem; font-size: 0.78rem; color: var(--accent-3); }

.stButton > button { background: linear-gradient(135deg, var(--accent-1) 0%, var(--accent-2) 100%) !important; color: white !important; border: none !important; border-radius: 12px !important; padding: 0.7rem 2rem !important; font-weight: 600 !important; font-family: 'Inter', sans-serif !important; transition: all 0.3s ease !important; }
.stButton > button:hover { box-shadow: 0 0 25px var(--glow) !important; transform: translateY(-2px) !important; }

.stTabs [data-baseweb="tab-list"] { gap: 4px; background: var(--bg-secondary); padding: 4px; border-radius: 12px; }
.stTabs [data-baseweb="tab"] { background: transparent; border-radius: 8px; padding: 8px 16px; color: var(--text-muted); font-weight: 500; }
.stTabs [aria-selected="true"] { background: rgba(99,102,241,0.2); color: var(--accent-3); }

.severity-high { color: var(--danger); font-weight: 700; } .severity-moderate { color: var(--warning); font-weight: 700; } .severity-low { color: var(--success); font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ── Model Loading ──
@st.cache_resource
def load_models():
    device = get_device()
    cls_model = create_model("base", num_classes=NUM_CLASSES, pretrained=True)
    ckpt = PROJECT_ROOT / "checkpoints" / "best_model.pth"
    cls_loaded = False
    if ckpt.exists():
        load_checkpoint(cls_model, str(ckpt), device)
        cls_loaded = True
    cls_model = cls_model.to(device).eval()
    return cls_model, device, cls_loaded

@st.cache_resource
def load_pipeline():
    from neurofusionnet.combination_of_segmentation_CNN.pipeline import SegClassPipeline
    ckpt_dir = PROJECT_ROOT / "checkpoints"
    return SegClassPipeline(
        seg_checkpoint=str(ckpt_dir / "unet_segmentation.pth"),
        cls_checkpoint=str(ckpt_dir / "best_model.pth"),
    )

@st.cache_resource
def load_llm():
    from neurofusionnet.combination_of_segmentation_CNN.llm_engine import FreeLLMEngine
    return FreeLLMEngine()

def predict_image(model, image_pil, device):
    transform = get_eval_transforms()
    tensor = transform(image_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        pred_class = probs.argmax(dim=1).item()
        confidence = probs[0, pred_class].item()
    all_probs = {CLASS_NAMES[i]: probs[0, i].item() for i in range(NUM_CLASSES)}
    return CLASS_NAMES[pred_class], confidence, all_probs, tensor

def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = tensor.cpu() * std + mean
    return torch.clamp(img, 0, 1).numpy().transpose(1, 2, 0)

def overlay_heatmap(img, heatmap, alpha=0.5):
    colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB) / 255.0
    if img.shape[:2] != heatmap.shape[:2]:
        colored = cv2.resize(colored, (img.shape[1], img.shape[0]))
    return np.clip((1 - alpha) * img + alpha * colored, 0, 1)

EMOJI = {"Glioma": "🔴", "Meningioma": "🟡", "Pituitary": "🟠", "No Tumor": "🟢"}
SEV_CLASS = {"High": "severity-high", "Low to Moderate": "severity-moderate", "Low": "severity-low", "None": "severity-low"}

# ── Sidebar ──
# Load llm BEFORE the sidebar block so it's available everywhere
cls_model, device, cls_loaded = load_models()
params = count_parameters(cls_model)

# Wrap llm loading in try/except so one bad API key doesn't crash the whole app
try:
    llm = load_llm()
except Exception as e:
    llm = None
    st.error(f"⚠️ LLM failed to load: {e}")

with st.sidebar:
    st.markdown('<div class="hero-title" style="font-size:1.6rem;">🧠 NeuroFusionNet</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#94a3b8;text-align:center;font-size:0.78rem;letter-spacing:1px;">HYBRID CNN-ViT • SEGMENTATION • XAI</p>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    # page = st.radio("Navigate", ["Dashboard", "Analyze", "Segmentation", "AI Consultant", "Explainability", "Analytics", "Architecture", "About"], label_visibility="collapsed")
    page = st.radio("Navigate", [
                    "Dashboard",
                    "Analyze",
                    "Segmentation",
                    "AI Consultant",
                    "Explainability",
                    "Analytics",
                    "Architecture",
                    "About"
                ], label_visibility="collapsed")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    cls_model, device, cls_loaded = load_models()
    params = count_parameters(cls_model)
    st.markdown(f"**System**")
    st.markdown(f"💻 `{device}` &nbsp; 📐 `{params['total_millions']:.1f}M` params")
    st.markdown(f"{'✅ Model trained' if cls_loaded else '⚠️ Untrained model'}")
    llm = load_llm()
    st.markdown(f"🤖 `{llm.provider_name}`")

# ═══════════════ PAGE: DASHBOARD ═══════════════
if page == "Dashboard":
    st.markdown('<div class="hero-title">🧠 NeuroFusionNet</div>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">Hybrid CNN-ViT Brain Tumor Detection with AI-Powered Segmentation,<br>Explainable AI, and Clinical Intelligence</p>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, val, label in [(c1,"4","Tumor Classes"),(c2,f"{params['total_millions']:.0f}M","Parameters"),(c3,"6K","MRI Scans"),(c4,"5","XAI Methods")]:
        with col: st.markdown(f'<div class="glass-card" style="text-align:center"><div class="stat-value">{val}</div><div class="stat-label">{label}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("### 🏗️ How It Works")
    svg_path = PROJECT_ROOT / "pipeline_horizontal.svg"
    if svg_path.exists():
        with open(svg_path, "r", encoding="utf-8") as f:
            svg_content = f.read()
        st.markdown(
            f'<div style="width:100%; overflow-x:auto">{svg_content}</div>',
            unsafe_allow_html=True
    )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown("### ✨ Key Features")
    feats = [("🧬","Hybrid Architecture","MobileNetV2 CNN + 4-layer Transformer with SE-gated fusion"),("🔬","5 XAI Methods","Grad-CAM, Grad-CAM++, Attention Rollout, SHAP, LIME"),("🤖","AI Reports",f"Clinical report generation via {llm.provider_name}"),("⚡","Lightweight",f"Only {params['total_millions']:.0f}M params — real-time inference"),("🫁","Segmentation","U-Net tumor localization before classification"),("📊","Full Analytics","Confusion matrix, ROC curves, per-class metrics")]
    cols = st.columns(3)
    for i, (icon, title, desc) in enumerate(feats):
        with cols[i % 3]:
            st.markdown(f'<div class="glass-card" style="text-align:center;margin-bottom:1rem"><div class="feat-icon">{icon}</div><div class="feat-title">{title}</div><div class="feat-desc">{desc}</div></div>', unsafe_allow_html=True)

# ═══════════════ PAGE: ANALYZE ═══════════════
elif page == "Analyze":
    st.markdown("## 🔬 Full Pipeline Analysis")
    st.markdown("Upload a brain MRI for end-to-end analysis: **Segmentation → Classification → XAI → Clinical Report**")

    uploaded = st.file_uploader("Upload Brain MRI", type=["jpg","jpeg","png","bmp"], key="analyze")
    if uploaded:
        image = Image.open(uploaded).convert("RGB")

        st.markdown('<div class="step-indicator"><div class="step done">① Upload</div><div class="step active">② Segment</div><div class="step">③ Classify</div><div class="step">④ Explain</div><div class="step">⑤ Report</div></div>', unsafe_allow_html=True)

        with st.spinner("🧠 Running full pipeline..."):
            pipeline = load_pipeline()
            results = pipeline.full_pipeline(image)
            xai_maps = pipeline.get_xai_explanations(results["tensor"])

        st.markdown('<div class="step-indicator"><div class="step done">① Upload</div><div class="step done">② Segment</div><div class="step done">③ Classify</div><div class="step done">④ Explain</div><div class="step done">⑤ Report</div></div>', unsafe_allow_html=True)

        pred, conf = results["prediction"], results["confidence"]
        emoji = EMOJI.get(pred, "🧠")

        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            st.image(np.array(image.resize((224,224))), caption="Original MRI", use_container_width=True)
        with c2:
            mask_vis = (results["prob_map"] * 255).astype(np.uint8)
            mask_colored = cv2.applyColorMap(mask_vis, cv2.COLORMAP_MAGMA)
            mask_colored = cv2.cvtColor(mask_colored, cv2.COLOR_BGR2RGB)
            st.image(mask_colored, caption=f"Segmentation ({results['tumor_percentage']:.1f}% area)", use_container_width=True)
        with c3:
            st.image(np.array(results["masked_image"]), caption="Masked ROI", use_container_width=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        c1, c2 = st.columns([1, 2])
        with c1:
            sev = TUMOR_INFO.get(pred, {}).get("severity", "N/A")
            sev_cls = SEV_CLASS.get(sev, "")
            st.markdown(f'<div class="result-card"><div class="result-emoji">{emoji}</div><div class="result-class">{pred}</div><div class="result-conf">{conf:.1%} confidence</div><p style="margin-top:0.5rem"><span class="{sev_cls}">Severity: {sev}</span></p></div>', unsafe_allow_html=True)
        with c2:
            st.markdown("#### 📊 Class Probabilities")
            for cls in sorted(results["all_probs"], key=results["all_probs"].get, reverse=True):
                p = results["all_probs"][cls]
                st.progress(p, text=f"{EMOJI.get(cls,'')} {cls}: {p:.1%}")

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("### 🔍 Explainability")
        orig_img = denormalize(results["tensor"][0])
        xai_cols = st.columns(4)
        with xai_cols[0]: st.image(orig_img, caption="Original", use_container_width=True)
        for i, (name, label) in enumerate([("gradcam","Grad-CAM"),("gradcam++","Grad-CAM++"),("attention_rollout","Attention")]):
            if name in xai_maps:
                with xai_cols[i+1]:
                    st.image(overlay_heatmap(orig_img, xai_maps[name]), caption=label, use_container_width=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("### 📋 AI Clinical Report")
        with st.spinner("📝 Generating report..."):
            xai_summary = ""
            if "gradcam" in xai_maps:
                hm = xai_maps["gradcam"]
                py, px_ = np.unravel_index(hm.argmax(), hm.shape)
                xai_summary = f"Grad-CAM peak activation in {'upper' if py < 112 else 'lower'}-{'left' if px_ < 112 else 'right'} region ({hm.max():.2f} intensity)."
            report = llm.generate_report(pred, conf, results["all_probs"], xai_summary=xai_summary, seg_info=results["seg_info"])
        st.markdown(f'<div class="report-container">{report}</div>', unsafe_allow_html=True)
        st.download_button("📥 Download Report", report, file_name=f"report_{pred.lower().replace(' ','_')}.txt", mime="text/plain")
    else:
        st.markdown('<div class="glass-card" style="text-align:center;padding:4rem"><div style="font-size:4rem">🏥</div><div class="feat-title" style="font-size:1.3rem;margin:1rem 0">Upload a Brain MRI to Begin Analysis</div><div class="feat-desc">Supports JPG, JPEG, PNG, BMP formats<br>The pipeline will automatically segment, classify, explain, and report.</div></div>', unsafe_allow_html=True)

# ═══════════════ PAGE: SEGMENTATION ═══════════════
elif page == "Segmentation":
    st.markdown("## 🧠 Tumor Segmentation")
    st.markdown("U-Net with MobileNetV2 encoder for pixel-wise tumor localization.")
    uploaded = st.file_uploader("Upload MRI", type=["jpg","jpeg","png"], key="seg")
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        with st.spinner("🔍 Segmenting..."):
            pipeline = load_pipeline()
            mask, prob_map = pipeline.segment(image)
            masked_img = pipeline.apply_mask(image, mask)
        c1, c2, c3 = st.columns(3)
        with c1: st.image(np.array(image.resize((224,224))), caption="Original", use_container_width=True)
        with c2:
            prob_vis = cv2.applyColorMap((prob_map * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
            st.image(cv2.cvtColor(prob_vis, cv2.COLOR_BGR2RGB), caption="Probability Map", use_container_width=True)
        with c3: st.image(np.array(masked_img), caption="Masked ROI", use_container_width=True)
        tumor_pct = mask.sum() / mask.size * 100
        st.info(f"📊 Tumor area: **{tumor_pct:.1f}%** of scan | Mask pixels: **{mask.sum():,}** / {mask.size:,}")

# ═══════════════ PAGE: AI CONSULTANT ═══════════════
elif page == "AI Consultant":
    st.markdown("## 🧬 AI Clinical Consultant")
    st.markdown(f"Interactive medical Q&A powered by **{llm.provider_name}**")
    if llm.is_llm_available:
        st.success(f"✅ Connected to {llm.provider_name}")
    else:
        st.info("ℹ️ Set `GEMINI_API_KEY` or `GROQ_API_KEY` for AI-powered responses. Using template engine.")

    uploaded = st.file_uploader("Upload MRI for consultation", type=["jpg","jpeg","png"], key="consult")
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        pred, conf, probs, tensor = predict_image(cls_model, image, device)
        c1, c2 = st.columns([1, 3])
        with c1: st.image(image, caption=f"{EMOJI.get(pred,'')} {pred} ({conf:.1%})", use_container_width=True)

        with c2:
            if "chat_history" not in st.session_state: st.session_state.chat_history = []
            # Display history
            for role, msg in st.session_state.chat_history:
                css = "chat-user" if role == "user" else "chat-ai"
                icon = "🧑‍⚕️" if role == "user" else "🤖"
                st.markdown(f'<div class="{css}">{icon} {msg}</div>', unsafe_allow_html=True)

            # Suggestions
            st.markdown("**Quick questions:**")
            suggestions = ["What are the treatment options?", "What is the prognosis?", "Explain the symptoms", "How confident is the model?"]
            scols = st.columns(4)
            for i, q in enumerate(suggestions):
                with scols[i]:
                    if st.button(q, key=f"sq_{i}"): st.session_state.pending_q = q

            question = st.chat_input("Ask about the diagnosis...")
            if not question and "pending_q" in st.session_state:
                question = st.session_state.pop("pending_q")

            if question:
                st.session_state.chat_history.append(("user", question))
                with st.spinner("🤔 Thinking..."):
                    history_pairs = [(st.session_state.chat_history[i][1], st.session_state.chat_history[i+1][1]) for i in range(0, len(st.session_state.chat_history)-1, 2) if i+1 < len(st.session_state.chat_history)]
                    answer = llm.ask_question(question, pred, conf, history_pairs)
                st.session_state.chat_history.append(("ai", answer))
                st.rerun()
    else:
        st.markdown('<div class="glass-card" style="text-align:center;padding:3rem"><div style="font-size:3.5rem">💬</div><div class="feat-title" style="font-size:1.2rem;margin:0.8rem 0">Upload an MRI to Start Consultation</div><div class="feat-desc">Ask questions about diagnosis, treatment, prognosis, and more.</div></div>', unsafe_allow_html=True)

# ═══════════════ PAGE: EXPLAINABILITY ═══════════════
elif page == "Explainability":
    st.markdown("## 🔍 Explainable AI Explorer")
    st.markdown("Deep dive into model decisions with 5 complementary XAI methods.")
    uploaded = st.file_uploader("Upload MRI", type=["jpg","jpeg","png"], key="xai")
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        pred, conf, probs, tensor = predict_image(cls_model, image, device)
        st.markdown(f"**Prediction:** {EMOJI.get(pred,'')} {pred} ({conf:.1%})")
        methods = st.multiselect("XAI Methods", ["gradcam","gradcam++","attention_rollout","shap","lime"], default=["gradcam","gradcam++","attention_rollout"])
        if st.button("🔬 Generate Explanations", type="primary"):
            with st.spinner("Computing explanations..."):
                from neurofusionnet.xai import XAIEngine
                xai = XAIEngine(cls_model, device)
                results = xai.explain(tensor, input_image_pil=image, methods=methods)
            orig = denormalize(tensor[0])
            n = min(len(methods)+1, 4)
            cols = st.columns(n)
            with cols[0]: st.image(orig, caption="Original", use_container_width=True)
            names = {"gradcam":"Grad-CAM","gradcam++":"Grad-CAM++","attention_rollout":"Attention Rollout","shap":"SHAP","lime":"LIME"}
            for i, m in enumerate(methods):
                if m in results and isinstance(results[m], np.ndarray):
                    with cols[(i+1) % n]:
                        st.image(overlay_heatmap(orig, results[m]), caption=names.get(m,m), use_container_width=True)
            # Stats table
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown("### 📊 Activation Statistics")
            stats = {}
            for m in methods:
                if m in results and isinstance(results[m], np.ndarray):
                    hm = results[m]
                    stats[names.get(m,m)] = {"Max": f"{hm.max():.3f}", "Mean": f"{hm.mean():.3f}", "Coverage (>0.5)": f"{(hm>0.5).mean():.1%}"}
            if stats: st.dataframe(pd.DataFrame(stats).T, use_container_width=True)
            # XAI narrative
            with st.spinner("Generating narrative..."): narrative = llm.generate_xai_narrative(pred, conf, results)
            st.markdown("### 📝 Analysis Narrative")
            st.markdown(narrative)
    else:
        methods_info = [("🟠 Grad-CAM","Gradient-based","CNN activation regions"),("🟡 Grad-CAM++","Gradient-based","Improved localization"),("🔵 Attention Rollout","Attention-based","Transformer attention flow"),("🟣 SHAP","Perturbation","Per-pixel Shapley values"),("🟢 LIME","Perturbation","Superpixel importance")]
        for name, typ, desc in methods_info:
            st.markdown(f'<div class="glass-card" style="margin-bottom:0.5rem"><strong>{name}</strong> <span class="tag">{typ}</span><br><span style="color:var(--text-muted)">{desc}</span></div>', unsafe_allow_html=True)

# ═══════════════ PAGE: ANALYTICS ═══════════════
elif page == "Analytics":
    st.markdown("## 📊 Model Analytics")
    history_path = PROJECT_ROOT / "checkpoints" / "training_history.json"
    if history_path.exists():
        with open(history_path) as f: history = json.load(f)
        epochs = list(range(1, len(history["train_loss"])+1))
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=epochs, y=history["train_loss"], name="Train", line=dict(color="#6366f1", width=2)))
            fig.add_trace(go.Scatter(x=epochs, y=history["val_loss"], name="Val", line=dict(color="#ef4444", width=2)))
            fig.update_layout(title="Loss", template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=350)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=epochs, y=history["train_acc"], name="Train", line=dict(color="#22c55e", width=2)))
            fig.add_trace(go.Scatter(x=epochs, y=history["val_acc"], name="Val", line=dict(color="#f59e0b", width=2)))
            fig.update_layout(title="Accuracy", template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=350)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No training history found. Train the model first to see analytics.")

    # Seg history
    seg_path = PROJECT_ROOT / "checkpoints" / "seg_training_history.json"
    if seg_path.exists():
        st.markdown("### 🧠 Segmentation Training")
        with open(seg_path) as f: sh = json.load(f)
        epochs = list(range(1, len(sh["train_loss"])+1))
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=epochs, y=sh["train_dice"], name="Train Dice", line=dict(color="#6366f1")))
            fig.add_trace(go.Scatter(x=epochs, y=sh["val_dice"], name="Val Dice", line=dict(color="#22c55e")))
            fig.update_layout(title="Dice Score", template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=300)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=epochs, y=sh["train_iou"], name="Train IoU", line=dict(color="#8b5cf6")))
            fig.add_trace(go.Scatter(x=epochs, y=sh["val_iou"], name="Val IoU", line=dict(color="#f59e0b")))
            fig.update_layout(title="IoU Score", template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=300)
            st.plotly_chart(fig, use_container_width=True)

# ═══════════════ PAGE: ARCHITECTURE ═══════════════
elif page == "Architecture":
    st.markdown("## 🏗️ System Architecture")
    st.markdown("### End-to-End Pipeline")
    # Fix: add encoding="utf-8" to prevent the Â^ character issue on Windows
    with open("pipeline.svg", "r", encoding="utf-8") as f:
        svg_content = f.read()

    st.markdown(
        f'<div style="width:100%; overflow-x:auto">{svg_content}</div>',
        unsafe_allow_html=True
)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 🧠 Classification Model")
        st.markdown(f"- **Architecture**: MobileNetV2 + 4-layer Transformer\n- **Parameters**: {params['total_millions']:.1f}M total\n- **Trainable**: {params['trainable_millions']:.1f}M\n- **Input**: 224×224 RGB\n- **Output**: 4-class softmax")
    with c2:
        st.markdown("### 🫁 Segmentation Model")
        st.markdown("- **Architecture**: U-Net + MobileNetV2 encoder\n- **Bridge**: ASPP (multi-scale)\n- **Parameters**: ~4.5M\n- **Input**: 224×224 RGB\n- **Output**: Binary mask (sigmoid)")

# ═══════════════ PAGE: ABOUT ═══════════════
elif page == "About":
    st.markdown("## ℹ️ About NeuroFusionNet")
    st.markdown("""
**NeuroFusionNet** is a comprehensive AI system for brain tumor detection and classification, combining:
- **U-Net Segmentation** for precise tumor localization
- **Hybrid CNN-ViT Classification** for accurate tumor typing
- **5-Method XAI Suite** for transparent, explainable predictions
- **Free LLM Integration** for clinical report generation
""")
    st.markdown("### 📁 BRISC 2025 Dataset")
    st.markdown("""
| Property | Value |
|----------|-------|
| **Total Images** | 6,000 |
| **Split** | 5,000 train / 1,000 test |
| **Modality** | T1-weighted MRI |
| **Classes** | 4 (balanced) |
| **Planes** | Axial, Coronal, Sagittal |
| **Annotation** | Expert-reviewed pixel masks |
""")
    fig = go.Figure(data=[go.Bar(x=CLASS_NAMES, y=[1500]*4, marker_color=["#ef4444","#f59e0b","#3b82f6","#22c55e"], text=[1500]*4, textposition="auto")])
    fig.update_layout(title="Class Distribution (Train)", template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=300)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("### ⚠️ Disclaimer")
    st.warning("This system is for **educational and research purposes only**. It is NOT a medical device and should NOT be used for clinical diagnosis. Always consult qualified healthcare professionals.")
