"""
MedGemma Engine — LLM-powered clinical insights for brain tumor classification.

Integrates Google's MedGemma 4B multimodal model to provide:
  1. Clinical report generation from MRI predictions
  2. Interactive medical Q&A about scans
  3. XAI narrative explanations

Uses HuggingFace Inference API with graceful fallback to template-based reports
when the model is unavailable.
"""

import os
import base64
import io
import json
from typing import Optional, Dict

import numpy as np
from PIL import Image

from .utils import CLASS_NAMES, TUMOR_INFO


# ── MedGemma Integration ─────────────────────────────────────────────────────

class MedGemmaEngine:
    """MedGemma-powered clinical report generator and medical Q&A.
    
    Attempts to use MedGemma via HuggingFace. Falls back to comprehensive
    template-based reports if the model is unavailable.
    
    Args:
        hf_token: HuggingFace API token (or set HF_TOKEN env var)
        model_id: MedGemma model identifier on HuggingFace
        use_api: Use HuggingFace Inference API (True) or local model (False)
    """

    def __init__(
        self,
        hf_token: Optional[str] = None,
        model_id: str = "google/medgemma-4b-it",
        use_api: bool = True,
    ):
        self.model_id = model_id
        self.hf_token = hf_token or os.environ.get("HF_TOKEN", "")
        self.use_api = use_api
        self.model = None
        self.processor = None
        self.pipeline = None
        self._is_available = False

        self._try_initialize()

    def _try_initialize(self):
        """Try to initialize MedGemma (API or local)."""
        if not self.hf_token:
            print("⚠️  MedGemma: No HuggingFace token found. Using template-based reports.")
            print("   Set HF_TOKEN environment variable or pass hf_token to enable MedGemma.")
            return

        try:
            if self.use_api:
                self._init_api()
            else:
                self._init_local()
        except Exception as e:
            print(f"⚠️  MedGemma initialization failed: {e}")
            print("   Falling back to template-based reports.")

    def _init_api(self):
        """Initialize HuggingFace Inference API client."""
        try:
            from huggingface_hub import InferenceClient
            self.client = InferenceClient(
                model=self.model_id,
                token=self.hf_token,
            )
            self._is_available = True
            print("✅ MedGemma: Connected via HuggingFace Inference API")
        except ImportError:
            print("⚠️  huggingface_hub not installed. Install with: pip install huggingface-hub")
        except Exception as e:
            print(f"⚠️  MedGemma API init failed: {e}")

    def _init_local(self):
        """Initialize local MedGemma model."""
        try:
            from transformers import AutoProcessor, AutoModelForImageTextToText
            import torch

            self.processor = AutoProcessor.from_pretrained(
                self.model_id, token=self.hf_token
            )
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_id,
                token=self.hf_token,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            self._is_available = True
            print("✅ MedGemma: Loaded locally")
        except Exception as e:
            print(f"⚠️  MedGemma local model load failed: {e}")

    @property
    def is_available(self) -> bool:
        return self._is_available

    # ── Report Generation ─────────────────────────────────────────────────────

    def generate_report(
        self,
        prediction: str,
        confidence: float,
        all_probs: Dict[str, float],
        image: Optional[Image.Image] = None,
        xai_summary: str = "",
    ) -> str:
        """Generate a clinical report for the given prediction.
        
        Tries MedGemma first, falls back to template-based report.
        """
        if self._is_available and image is not None:
            try:
                return self._medgemma_report(prediction, confidence, all_probs, image, xai_summary)
            except Exception as e:
                print(f"⚠️  MedGemma report generation failed: {e}")

        return self._template_report(prediction, confidence, all_probs, xai_summary)

    def _medgemma_report(self, prediction, confidence, all_probs, image, xai_summary):
        """Generate report using MedGemma."""
        prompt = f"""You are a neuroradiology AI assistant analyzing a brain MRI scan.

A deep learning model (NeuroFusionNet — a hybrid CNN-ViT architecture) has classified this brain MRI image.

**Classification Result:**
- Predicted class: {prediction}
- Confidence: {confidence:.1%}
- All probabilities: {json.dumps({k: f'{v:.1%}' for k, v in all_probs.items()})}

{f'**XAI Analysis:** {xai_summary}' if xai_summary else ''}

Please provide a structured clinical report including:
1. **Findings**: Description of what was detected
2. **Impression**: Clinical significance
3. **Tumor Characteristics**: If a tumor is detected, describe typical characteristics
4. **Differential Diagnosis**: Other possible diagnoses to consider
5. **Recommended Actions**: Suggested follow-up steps

Important: Include a disclaimer that this is an AI-generated analysis for educational purposes and should not replace professional medical diagnosis."""

        if self.use_api:
            # Encode image
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_b64 = base64.b64encode(buffered.getvalue()).decode()

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            response = self.client.chat_completion(
                messages=messages,
                max_tokens=1024,
                temperature=0.3,
            )
            return response.choices[0].message.content
        else:
            # Local inference
            import torch
            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt",
            ).to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, max_new_tokens=1024, temperature=0.3,
                    do_sample=True,
                )
            return self.processor.decode(outputs[0], skip_special_tokens=True)

    def _template_report(self, prediction, confidence, all_probs, xai_summary=""):
        """Generate comprehensive template-based report (fallback)."""
        info = TUMOR_INFO.get(prediction, TUMOR_INFO["No Tumor"])

        # Confidence assessment
        if confidence >= 0.90:
            conf_level = "Very High"
            conf_note = "The model shows strong conviction in this classification."
        elif confidence >= 0.75:
            conf_level = "High"
            conf_note = "The model is fairly confident in this classification."
        elif confidence >= 0.50:
            conf_level = "Moderate"
            conf_note = "The model shows moderate confidence. Additional review recommended."
        else:
            conf_level = "Low"
            conf_note = "The model shows low confidence. Manual expert review is strongly recommended."

        # Build probability ranking
        sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
        prob_lines = "\n".join([f"  - {cls}: {prob:.1%}" for cls, prob in sorted_probs])

        # Differential diagnosis (second most likely)
        diff_dx = sorted_probs[1][0] if len(sorted_probs) > 1 else "N/A"
        diff_prob = sorted_probs[1][1] if len(sorted_probs) > 1 else 0

        report = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🧠 NEUROFUSIONNET — AI CLINICAL REPORT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 CLASSIFICATION RESULT
  Primary Diagnosis: {prediction}
  Confidence: {confidence:.1%} ({conf_level})
  {conf_note}

📊 PROBABILITY DISTRIBUTION
{prob_lines}

🔬 FINDINGS
  {info['description']}

🏥 CLINICAL ASSESSMENT
  • Severity Level: {info['severity']}
  • Typical Location: {info['typical_location']}

💊 RECOMMENDED TREATMENT OPTIONS
  {chr(10).join(f'  • {t}' for t in info['common_treatments'])}

📈 PROGNOSIS
  {info['prognosis']}

🔄 DIFFERENTIAL DIAGNOSIS
  • Primary: {prediction} ({confidence:.1%})
  • Secondary: {diff_dx} ({diff_prob:.1%})
  {'• Consider further imaging or biopsy if confidence < 75%' if confidence < 0.75 else ''}
"""

        if xai_summary:
            report += f"""
🔍 XAI ANALYSIS SUMMARY
  {xai_summary}
"""

        report += """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️  DISCLAIMER: This report is generated by an AI system
for educational and research purposes only. It should NOT
be used as a substitute for professional medical diagnosis.
Always consult a qualified healthcare provider.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        return report

    # ── Q&A Interface ─────────────────────────────────────────────────────────

    def ask_question(
        self,
        question: str,
        prediction: str,
        confidence: float,
        image: Optional[Image.Image] = None,
    ) -> str:
        """Answer a medical question about the scan.
        
        Uses MedGemma if available, otherwise uses knowledge base.
        """
        if self._is_available and image is not None:
            try:
                return self._medgemma_qa(question, prediction, confidence, image)
            except Exception as e:
                print(f"⚠️  MedGemma Q&A failed: {e}")

        return self._template_qa(question, prediction, confidence)

    def _medgemma_qa(self, question, prediction, confidence, image):
        """Answer question using MedGemma."""
        prompt = f"""You are a neuroradiology AI assistant. A brain MRI has been classified as '{prediction}' with {confidence:.1%} confidence by a hybrid CNN-ViT model.

The user asks: {question}

Please provide a helpful, accurate, and educational response. Include relevant medical context but note that this is for educational purposes only."""

        if self.use_api:
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_b64 = base64.b64encode(buffered.getvalue()).decode()

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            response = self.client.chat_completion(
                messages=messages, max_tokens=512, temperature=0.4,
            )
            return response.choices[0].message.content
        else:
            import torch
            inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=512, temperature=0.4, do_sample=True)
            return self.processor.decode(outputs[0], skip_special_tokens=True)

    def _template_qa(self, question, prediction, confidence):
        """Answer common questions using knowledge base (fallback)."""
        info = TUMOR_INFO.get(prediction, TUMOR_INFO["No Tumor"])
        question_lower = question.lower()

        # Pattern matching for common questions
        if any(w in question_lower for w in ["treatment", "treat", "cure", "therapy"]):
            treatments = "\n".join(f"  • {t}" for t in info["common_treatments"])
            return f"""**Treatment Options for {prediction}:**

{treatments}

**Note:** Treatment selection depends on multiple factors including tumor size, location, grade, patient age, and overall health. The final treatment plan should be determined by a multidisciplinary medical team.

⚠️ This is AI-generated educational information, not medical advice."""

        elif any(w in question_lower for w in ["prognosis", "survival", "outcome", "chance"]):
            return f"""**Prognosis for {prediction}:**

{info['prognosis']}

**Factors Affecting Prognosis:**
  • Tumor grade and subtype
  • Size and location
  • Extent of surgical resection
  • Patient's age and overall health
  • Response to adjuvant therapy

⚠️ Individual prognosis varies significantly. Consult a specialist for personalized assessment."""

        elif any(w in question_lower for w in ["location", "where", "region", "area"]):
            return f"""**Typical Location of {prediction}:**

{info['typical_location']}

**About the Detection:**
{info['description']}

The AI model identified features consistent with this type of tumor with {confidence:.1%} confidence."""

        elif any(w in question_lower for w in ["symptom", "sign", "feel"]):
            symptoms = {
                "Glioma": "headaches, seizures, cognitive changes, personality changes, weakness, speech difficulties, vision problems",
                "Meningioma": "headaches, vision changes, hearing loss, memory problems, seizures, weakness in limbs",
                "Pituitary": "vision changes (especially peripheral), headaches, hormonal imbalances, fatigue, unexplained weight changes",
                "No Tumor": "N/A — No tumor detected",
            }
            return f"""**Common Symptoms of {prediction}:**

Typical symptoms include: {symptoms.get(prediction, 'consult a healthcare provider')}.

**Important:** Symptoms vary widely between individuals. The presence or absence of symptoms does not confirm or rule out a diagnosis.

⚠️ This is AI-generated educational information."""

        elif any(w in question_lower for w in ["confident", "sure", "accurate", "trust", "reliability"]):
            return f"""**Model Confidence Analysis:**

  • Predicted: {prediction}
  • Confidence: {confidence:.1%}

{'✅ High confidence — The model is very certain about this classification.' if confidence >= 0.85 else ''}
{'⚡ Moderate confidence — Consider seeking additional expert review.' if 0.50 <= confidence < 0.85 else ''}
{'⚠️ Low confidence — Expert review is strongly recommended.' if confidence < 0.50 else ''}

**About the Model:**
NeuroFusionNet is a hybrid CNN-ViT architecture that combines local spatial features (from MobileNetV2 CNN) with global contextual understanding (from a Vision Transformer). This dual approach often provides more reliable classifications than either approach alone.

⚠️ AI confidence scores are statistical measures, not guarantees of accuracy."""

        else:
            return f"""**About {prediction}:**

{info['description']}

**Key Facts:**
  • Severity: {info['severity']}
  • Location: {info['typical_location']}
  • Model Confidence: {confidence:.1%}

For more specific questions, try asking about:
  • Treatment options
  • Prognosis and outcomes
  • Common symptoms
  • Model confidence and reliability

⚠️ This is AI-generated educational information, not medical advice."""

    # ── XAI Narrative ─────────────────────────────────────────────────────────

    def generate_xai_narrative(
        self,
        prediction: str,
        confidence: float,
        xai_results: dict,
    ) -> str:
        """Generate natural language explanation of XAI findings."""
        parts = []
        parts.append(f"The model classified this MRI as **{prediction}** with "
                     f"**{confidence:.1%}** confidence.")

        if "gradcam" in xai_results:
            heatmap = xai_results["gradcam"]
            # Find peak activation region
            peak_y, peak_x = np.unravel_index(heatmap.argmax(), heatmap.shape)
            region = _get_spatial_region(peak_x, peak_y, 224)
            intensity = heatmap.max()
            parts.append(
                f"**Grad-CAM analysis** shows the CNN focused most intensely on the "
                f"**{region}** region of the scan (activation intensity: {intensity:.2f})."
            )

        if "attention_rollout" in xai_results:
            heatmap = xai_results["attention_rollout"]
            peak_y, peak_x = np.unravel_index(heatmap.argmax(), heatmap.shape)
            region = _get_spatial_region(peak_x, peak_y, 224)
            spread = (heatmap > 0.5).mean()
            parts.append(
                f"**Transformer attention** is concentrated on the **{region}** area, "
                f"covering approximately **{spread:.0%}** of the image. "
                f"{'This focused attention suggests a well-defined region of interest.' if spread < 0.3 else 'The broader attention pattern suggests diffuse features were considered.'}"
            )

        if "shap" in xai_results:
            heatmap = xai_results["shap"]
            high_importance = (heatmap > 0.7).mean()
            parts.append(
                f"**SHAP analysis** identified **{high_importance:.1%}** of pixels as "
                f"highly important for this decision, indicating "
                f"{'focal diagnostic features' if high_importance < 0.15 else 'distributed evidence across the scan'}."
            )

        return "\n\n".join(parts)


def _get_spatial_region(x, y, img_size):
    """Map pixel coordinates to intuitive spatial region names."""
    # Divide into 3×3 grid
    col = "left" if x < img_size / 3 else ("center" if x < 2 * img_size / 3 else "right")
    row = "upper" if y < img_size / 3 else ("middle" if y < 2 * img_size / 3 else "lower")
    if col == "center" and row == "middle":
        return "central"
    return f"{row}-{col}"
