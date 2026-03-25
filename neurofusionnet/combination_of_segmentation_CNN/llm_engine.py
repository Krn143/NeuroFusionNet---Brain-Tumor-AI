"""
Free LLM Engine for Clinical Report Generation and Interactive Q&A.

Supports multiple free LLM providers with automatic fallback:
  1. Google Gemini (free tier — 15 RPM)
  2. Groq (free tier — Llama 3)
  3. Template-based fallback (no API needed)

Set environment variable GEMINI_API_KEY or GROQ_API_KEY to enable.
"""

import os
import json
from typing import Optional, Dict

import numpy as np
from PIL import Image


# ── Tumor knowledge base ─────────────────────────────────────────────────────

TUMOR_INFO = {
    "Glioma": {
        "description": "Gliomas arise from glial cells in the brain or spine. They are the most common type of primary brain tumor, accounting for ~30% of all brain tumors.",
        "severity": "High",
        "grade": "WHO Grade I-IV",
        "treatments": ["Surgical resection", "Radiation therapy", "Chemotherapy (Temozolomide)", "Targeted therapy (Bevacizumab)", "Tumor Treating Fields (TTFields)"],
        "location": "Cerebral hemispheres, brainstem, cerebellum",
        "prognosis": "Variable — Low-grade (I-II): 5-15+ years median survival. High-grade (III-IV/Glioblastoma): 12-18 months median survival.",
        "symptoms": "Headaches, seizures, cognitive changes, personality changes, motor weakness, speech difficulties, vision problems",
    },
    "Meningioma": {
        "description": "Meningiomas develop from the meninges (protective membranes surrounding the brain and spinal cord). Most are benign and slow-growing.",
        "severity": "Low to Moderate",
        "grade": "WHO Grade I-III (most are Grade I)",
        "treatments": ["Observation (watch and wait)", "Surgical resection", "Stereotactic radiosurgery (Gamma Knife)", "Conventional radiation"],
        "location": "Convexity, parasagittal, sphenoid wing, posterior fossa, olfactory groove",
        "prognosis": "Generally favorable — 5-year survival >90% for Grade I. Recurrence rate: 7-25% for Grade I, higher for Grade II-III.",
        "symptoms": "Headaches, vision changes, hearing loss, memory problems, seizures, limb weakness",
    },
    "Pituitary": {
        "description": "Pituitary adenomas are tumors of the pituitary gland. Most are benign, slow-growing, and may cause hormonal imbalances.",
        "severity": "Low to Moderate",
        "grade": "Typically benign (adenoma)",
        "treatments": ["Medication (dopamine agonists for prolactinomas)", "Transsphenoidal surgery", "Radiation therapy", "Hormone replacement therapy"],
        "location": "Sella turcica (pituitary fossa at base of brain)",
        "prognosis": "Generally excellent — most are curable with surgery. 5-year recurrence: 10-25% depending on subtype.",
        "symptoms": "Vision changes (bitemporal hemianopia), headaches, hormonal imbalances (amenorrhea, acromegaly, Cushing's), fatigue, weight changes",
    },
    "No Tumor": {
        "description": "No tumorous growth detected in the MRI scan. Brain parenchyma appears within normal limits.",
        "severity": "None",
        "grade": "N/A",
        "treatments": ["No treatment required", "Routine follow-up if symptomatic"],
        "location": "N/A",
        "prognosis": "N/A — No pathological findings detected.",
        "symptoms": "N/A",
    },
}


class FreeLLMEngine:
    """Multi-provider free LLM engine for clinical AI reports.
    
    Automatically selects the best available provider:
    1. Google Gemini (GEMINI_API_KEY or GOOGLE_API_KEY)
    2. Groq (GROQ_API_KEY) 
    3. Template fallback (always available)
    """

    def __init__(self):
        self.provider = None
        self.client = None
        self._try_initialize()

    def _try_initialize(self):
        """Try providers in order of preference."""
        # Try Gemini first
        gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "AIzaSyDkmT6HnzXpWYTeMTeFrikmBTWHExEGdPE")
        if gemini_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=gemini_key)
                self.client = genai.GenerativeModel("gemini-2.0-flash")
                # Quick test
                self.client.generate_content("test", generation_config={"max_output_tokens": 5})
                self.provider = "gemini"
                print("✅ LLM Engine: Google Gemini (free tier) connected")
                return
            except Exception as e:
                print(f"⚠️  Gemini init failed: {e}")

        # Try Groq
        groq_key = os.environ.get("GROQ_API_KEY", "")
        if groq_key:
            try:
                from groq import Groq
                self.client = Groq(api_key=groq_key)
                self.provider = "groq"
                print("✅ LLM Engine: Groq (Llama 3) connected")
                return
            except Exception as e:
                print(f"⚠️  Groq init failed: {e}")

        # Fallback
        self.provider = "template"
        print("[INFO] LLM Engine: Using template-based reports (set GEMINI_API_KEY or GROQ_API_KEY for AI reports)")

    @property
    def is_llm_available(self) -> bool:
        return self.provider in ("gemini", "groq")

    @property
    def provider_name(self) -> str:
        names = {"gemini": "Google Gemini", "groq": "Groq (Llama 3)", "template": "Template Engine"}
        return names.get(self.provider, "Unknown")

    def _call_llm(self, prompt: str, max_tokens: int = 1024) -> str:
        """Send prompt to the active LLM provider."""
        if self.provider == "gemini":
            response = self.client.generate_content(
                prompt,
                generation_config={
                    "max_output_tokens": max_tokens,
                    "temperature": 0.4,
                },
            )
            return response.text

        elif self.provider == "groq":
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.4,
            )
            return response.choices[0].message.content

        return ""

    # ── Clinical Report ──────────────────────────────────────────────────────

    def generate_report(
        self,
        prediction: str,
        confidence: float,
        all_probs: Dict[str, float],
        xai_summary: str = "",
        seg_info: str = "",
    ) -> str:
        """Generate a clinical report for the prediction.
        
        Args:
            prediction: Predicted tumor class
            confidence: Prediction confidence (0-1)
            all_probs: All class probabilities
            xai_summary: XAI analysis summary text
            seg_info: Segmentation information (tumor area, location)
        """
        if self.is_llm_available:
            try:
                prompt = f"""You are an expert neuroradiology AI assistant. Generate a structured clinical report for a brain MRI analysis.

**AI Model Used:** NeuroFusionNet (Hybrid CNN-ViT with U-Net Segmentation)

**Classification Result:**
- Predicted class: {prediction}
- Confidence: {confidence:.1%}
- All probabilities: {json.dumps({k: f'{v:.1%}' for k, v in all_probs.items()})}

{f'**Segmentation Analysis:** {seg_info}' if seg_info else ''}
{f'**XAI Explainability:** {xai_summary}' if xai_summary else ''}

Generate a professional report with these sections:
1. **FINDINGS** — What was detected and key observations
2. **CLINICAL IMPRESSION** — Significance and assessment
3. **TUMOR CHARACTERISTICS** — If tumor detected, describe typical features
4. **DIFFERENTIAL DIAGNOSIS** — Alternative diagnoses to consider
5. **RECOMMENDED ACTIONS** — Follow-up steps and recommendations
6. **CONFIDENCE ASSESSMENT** — Model reliability for this case

End with an important disclaimer: This is AI-generated analysis for educational/research purposes only and should not replace professional medical diagnosis.

Keep the report professional, concise, and well-structured. Use medical terminology appropriately."""

                return self._call_llm(prompt, max_tokens=1024)
            except Exception as e:
                print(f"⚠️  LLM report failed: {e}")

        return self._template_report(prediction, confidence, all_probs, xai_summary, seg_info)

    def _template_report(self, prediction, confidence, all_probs, xai_summary="", seg_info=""):
        """Comprehensive template-based report (fallback)."""
        info = TUMOR_INFO.get(prediction, TUMOR_INFO["No Tumor"])

        if confidence >= 0.90:
            conf_level, conf_note = "Very High", "The model shows strong conviction in this classification."
        elif confidence >= 0.75:
            conf_level, conf_note = "High", "The model is fairly confident. Results are reliable."
        elif confidence >= 0.50:
            conf_level, conf_note = "Moderate", "Moderate confidence. Additional expert review recommended."
        else:
            conf_level, conf_note = "Low", "Low confidence. Manual expert review is strongly recommended."

        sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
        prob_lines = "\n".join([f"    • {cls}: {prob:.1%}" for cls, prob in sorted_probs])

        report = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  🧠 NEUROFUSIONNET — AI CLINICAL REPORT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

▸ FINDINGS
  {info['description']}

▸ CLASSIFICATION
  Primary Diagnosis: {prediction}
  Confidence: {confidence:.1%} ({conf_level})
  {conf_note}

▸ PROBABILITY DISTRIBUTION
{prob_lines}

▸ CLINICAL ASSESSMENT
  • Severity: {info['severity']}
  • Grade: {info['grade']}
  • Location: {info['location']}"""

        if seg_info:
            report += f"""

▸ SEGMENTATION ANALYSIS
  {seg_info}"""

        report += f"""

▸ TREATMENT OPTIONS
  {chr(10).join(f'  • {t}' for t in info['treatments'])}

▸ PROGNOSIS
  {info['prognosis']}

▸ DIFFERENTIAL DIAGNOSIS
  • Primary: {prediction} ({confidence:.1%})
  • Secondary: {sorted_probs[1][0] if len(sorted_probs) > 1 else 'N/A'} ({sorted_probs[1][1]:.1%} if len(sorted_probs) > 1 else '')"""

        if xai_summary:
            report += f"""

▸ XAI ANALYSIS
  {xai_summary}"""

        report += """

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ⚠️ DISCLAIMER: AI-generated analysis for
  educational/research purposes only. Not a
  substitute for professional medical diagnosis.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        return report

    # ── Interactive Q&A ──────────────────────────────────────────────────────

    def ask_question(
        self,
        question: str,
        prediction: str,
        confidence: float,
        chat_history: list = None,
    ) -> str:
        """Answer a medical question about the scan.
        
        Args:
            question: User's question
            prediction: Current prediction
            confidence: Prediction confidence
            chat_history: Previous Q&A pairs for context
        """
        if self.is_llm_available:
            try:
                history_text = ""
                if chat_history:
                    for q, a in chat_history[-3:]:  # Last 3 interactions for context
                        history_text += f"\nUser: {q}\nAssistant: {a}\n"

                prompt = f"""You are an expert neuroradiology AI assistant. A brain MRI has been analyzed by NeuroFusionNet (hybrid CNN-ViT model).

**Diagnosis:** {prediction} (Confidence: {confidence:.1%})

**Known information about {prediction}:**
{json.dumps(TUMOR_INFO.get(prediction, {}), indent=2)}

{f'**Previous conversation:**{history_text}' if history_text else ''}

**User question:** {question}

Provide a helpful, accurate, and educational response. Be detailed but accessible. Include relevant medical context. Always remind that this is educational information, not medical advice."""

                return self._call_llm(prompt, max_tokens=512)
            except Exception as e:
                print(f"⚠️  LLM Q&A failed: {e}")

        return self._template_qa(question, prediction, confidence)

    def _template_qa(self, question, prediction, confidence):
        """Template-based Q&A fallback."""
        info = TUMOR_INFO.get(prediction, TUMOR_INFO["No Tumor"])
        q = question.lower()

        if any(w in q for w in ["treatment", "treat", "cure", "therapy", "medicine"]):
            treatments = "\n".join(f"  • {t}" for t in info["treatments"])
            return f"**Treatment Options for {prediction}:**\n\n{treatments}\n\n_Treatment selection depends on tumor size, location, grade, patient age, and overall health. A multidisciplinary team should determine the final plan._\n\n⚠️ This is educational information, not medical advice."

        elif any(w in q for w in ["prognosis", "survival", "outcome", "chance", "life"]):
            return f"**Prognosis for {prediction}:**\n\n{info['prognosis']}\n\n**Factors affecting prognosis:** tumor grade/subtype, size, location, extent of resection, patient age, response to therapy.\n\n⚠️ Individual prognosis varies significantly. Consult a specialist."

        elif any(w in q for w in ["symptom", "sign", "feel", "pain"]):
            return f"**Common Symptoms of {prediction}:**\n\n{info['symptoms']}\n\n_Symptoms vary widely between individuals. Presence or absence of symptoms doesn't confirm or rule out a diagnosis._\n\n⚠️ Educational information only."

        elif any(w in q for w in ["location", "where", "region", "area", "brain"]):
            return f"**Typical Location of {prediction}:**\n\n{info['location']}\n\n{info['description']}\n\nModel confidence: {confidence:.1%}"

        elif any(w in q for w in ["confident", "sure", "accurate", "trust", "reliable"]):
            level = "high" if confidence >= 0.85 else ("moderate" if confidence >= 0.5 else "low")
            return f"**Model Confidence: {confidence:.1%} ({level})**\n\nNeuroFusionNet uses a hybrid CNN-ViT architecture combining local spatial features with global context for more reliable classifications.\n\n{'✅ High confidence — results are reliable.' if confidence >= 0.85 else '⚠️ Consider seeking expert review.'}"

        elif any(w in q for w in ["what", "explain", "about", "tell"]):
            return f"**About {prediction}:**\n\n{info['description']}\n\n• Severity: {info['severity']}\n• Grade: {info['grade']}\n• Location: {info['location']}\n• Confidence: {confidence:.1%}"

        else:
            return f"**{prediction} — Key Information:**\n\n{info['description']}\n\n• Severity: {info['severity']}\n• Location: {info['location']}\n\nTry asking about: treatments, prognosis, symptoms, location, or model confidence.\n\n⚠️ Educational information only."

    # ── XAI Narrative ────────────────────────────────────────────────────────

    def generate_xai_narrative(self, prediction, confidence, xai_results) -> str:
        """Generate natural language explanation of XAI findings."""
        if self.is_llm_available:
            try:
                # Summarize heatmap stats
                stats = {}
                for method, heatmap in xai_results.items():
                    if method == "prediction" or not isinstance(heatmap, np.ndarray):
                        continue
                    peak_y, peak_x = np.unravel_index(heatmap.argmax(), heatmap.shape)
                    stats[method] = {
                        "max_activation": float(heatmap.max()),
                        "mean_activation": float(heatmap.mean()),
                        "coverage_above_50pct": float((heatmap > 0.5).mean()),
                        "peak_region": _get_spatial_region(peak_x, peak_y, 224),
                    }

                prompt = f"""You are an XAI (Explainable AI) specialist analyzing brain tumor MRI classification results.

**Prediction:** {prediction} (Confidence: {confidence:.1%})

**XAI Method Results:**
{json.dumps(stats, indent=2)}

Write a clear, professional narrative explaining:
1. Where the model focused its attention (based on peak regions)
2. How different XAI methods agree or disagree
3. What this means for the reliability of the prediction
4. Whether the focused regions are clinically consistent with {prediction}

Keep it concise (3-4 paragraphs). Use medical context where relevant."""

                return self._call_llm(prompt, max_tokens=512)
            except Exception:
                pass

        # Template fallback
        return self._template_xai_narrative(prediction, confidence, xai_results)

    def _template_xai_narrative(self, prediction, confidence, xai_results):
        parts = [f"The model classified this MRI as **{prediction}** with **{confidence:.1%}** confidence."]

        for method, heatmap in xai_results.items():
            if method == "prediction" or not isinstance(heatmap, np.ndarray):
                continue
            peak_y, peak_x = np.unravel_index(heatmap.argmax(), heatmap.shape)
            region = _get_spatial_region(peak_x, peak_y, 224)
            names = {"gradcam": "Grad-CAM", "gradcam++": "Grad-CAM++",
                     "attention_rollout": "Attention Rollout", "shap": "SHAP", "lime": "LIME"}
            parts.append(f"**{names.get(method, method)}** focused on the **{region}** region (intensity: {heatmap.max():.2f}).")

        return "\n\n".join(parts)


def _get_spatial_region(x, y, size):
    col = "left" if x < size/3 else ("center" if x < 2*size/3 else "right")
    row = "upper" if y < size/3 else ("middle" if y < 2*size/3 else "lower")
    return "central" if col == "center" and row == "middle" else f"{row}-{col}"
