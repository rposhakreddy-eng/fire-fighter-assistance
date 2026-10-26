"""
Groq AI Multimodal Vision Analysis Module
Provides secondary reasoning on thermal imagery, occluded victim signatures,
and rescue recommendations using Groq vision models.
Fails gracefully if GROQ_API_KEY is not configured or network is unreachable.
"""

import os
import base64
import re
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root or current dir
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR.parent / ".env"
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)
else:
    load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY", "").strip()

# Lazy client initialization
_client = None

def get_groq_client():
    global _client
    if _client is not None:
        return _client
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        from groq import Groq
        _client = Groq(api_key=api_key)
        return _client
    except Exception as e:
        print(f"[Groq AI] Client initialization error: {e}")
        return None


def is_groq_configured() -> bool:
    """Check if Groq API key is present."""
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    return bool(api_key and not api_key.startswith("your_"))


def analyze_with_grok(image_path: str, detections: list) -> dict:
    """
    Perform multimodal analysis on thermal / drone image.
    Returns structured analysis dict.
    """
    if not is_groq_configured():
        return {
            "configured": False,
            "human_assessment": "NOT CONFIGURED",
            "visible_body_part": "Groq AI reasoning not configured in .env",
            "evidence": "Add a valid GROQ_API_KEY to the .env file to enable secondary multimodal LLM vision analysis.",
            "confidence": "N/A",
            "recommendation": "Rely on primary YOLO11 detections and standard emergency response protocol.",
            "raw_text": "Groq API key not configured."
        }

    client = get_groq_client()
    if client is None:
        return {
            "configured": False,
            "human_assessment": "ERROR",
            "visible_body_part": "Client initialization failed",
            "evidence": "Unable to initialize Groq client.",
            "confidence": "N/A",
            "recommendation": "Check Python groq package installation.",
            "raw_text": "Error initializing Groq client."
        }

    try:
        # Read and encode image to base64
        with open(image_path, "rb") as file:
            image_bytes = file.read()

        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        ext = Path(image_path).suffix.lower()
        mime_type = "image/png" if ext == ".png" else "image/jpeg"
        image_data = f"data:{mime_type};base64,{base64_image}"

        prompt = f"""
You are an expert AI assistant for a Fire Fighter Assistance Drone System.
Analyze this thermal or drone camera image carefully.

The local computer-vision model detected:
{detections if detections else "No full person detected by primary bounding box model."}

Analyze whether this image contains visual or thermal evidence consistent with a trapped human victim.
In smoke-filled buildings, victims are often partially occluded by debris, furniture, or smoke.
Look for visible body parts (head, face, hand, arm, leg, foot, torso).
Do not assume every hot region is a human. Note false positives (radiators, fires, electronics).

Provide your assessment in the following exact format:
Human assessment: YES / NO / UNCERTAIN
Visible body part: [Specify exact body parts observed, or None]
Evidence: [1-2 concise sentences describing visual/thermal findings]
Confidence: LOW / MEDIUM / HIGH
Firefighter recommendation: [Concise, actionable advice for firefighters on-scene]
"""

        response = client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_data}
                        }
                    ]
                }
            ],
            temperature=0.2,
            timeout=15.0
        )

        content = response.choices[0].message.content or ""
        return parse_groq_response(content)

    except Exception as e:
        print(f"[Groq AI] Inference warning/error: {e}")
        return {
            "configured": True,
            "human_assessment": "ANALYSIS UNAVAILABLE",
            "visible_body_part": "N/A",
            "evidence": f"Secondary vision model request failed: {str(e)[:120]}",
            "confidence": "N/A",
            "recommendation": "Proceed with caution using primary YOLO detections.",
            "raw_text": str(e)
        }


def parse_groq_response(text: str) -> dict:
    """Parse structured fields from LLM response text."""
    assessment_match = re.search(r"Human assessment:\s*(YES|NO|UNCERTAIN)", text, re.IGNORECASE)
    body_match = re.search(r"Visible body part:\s*([^\n\r]+)", text, re.IGNORECASE)
    evidence_match = re.search(r"Evidence:\s*([^\n\r]+(?:\n[^\n\r]+)?)", text, re.IGNORECASE)
    confidence_match = re.search(r"Confidence:\s*(LOW|MEDIUM|HIGH)", text, re.IGNORECASE)
    rec_match = re.search(r"Firefighter recommendation:\s*([^\n\r]+(?:\n[^\n\r]+)?)", text, re.IGNORECASE)

    return {
        "configured": True,
        "human_assessment": assessment_match.group(1).upper() if assessment_match else "UNCERTAIN",
        "visible_body_part": body_match.group(1).strip() if body_match else "None specified",
        "evidence": evidence_match.group(1).strip() if evidence_match else text[:200],
        "confidence": confidence_match.group(1).upper() if confidence_match else "MEDIUM",
        "recommendation": rec_match.group(1).strip() if rec_match else "Inspect area with thermal camera.",
        "raw_text": text
    }