"""
Robert Wen Resume Chatbot - Runs a local open-source model, no API keys needed.
Deploy to Hugging Face Spaces: https://huggingface.co/spaces

Custom colors: Set these in Space Settings → Variables (or env vars):
  CHATBOT_BG, CHATBOT_HOVER_BG, CHATBOT_HOVER_BORDER, CHATBOT_INPUT_TEXT,
  CHATBOT_ACCENT, CHATBOT_ACCENT_DIM, CHATBOT_TEXT, CHATBOT_MUTED, CHATBOT_BUTTON_TEXT
"""

import os
import gradio as gr

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

# Custom colors from env (Space Settings → Variables).
# Defaults match index.html :root and .recognition-list / .skill-tag
def _c(name: str, default: str) -> str:
    return os.environ.get(name, default).strip() or default

# Darker purple-blue for all backgrounds
DARK_BG = _c("CHATBOT_BG", "#1a1b29")
# Accent yellow for all hover states
HOVER_BG = _c("CHATBOT_HOVER_BG", "#f4e4a6")  # yellow accent
HOVER_BORDER = _c("CHATBOT_HOVER_BORDER", "rgba(244,228,166,0.5)")
# Lighter purple-blue for example text
INPUT_TEXT = _c("CHATBOT_INPUT_TEXT", "#a8acd4")
# Dark dark blue for input field text and placeholder
INPUT_DARK_BLUE = _c("CHATBOT_INPUT_DARK_BLUE", "#0a0a1a")
# Other theme colors
ACCENT = _c("CHATBOT_ACCENT", "#f4e4a6")
ACCENT_DIM = _c("CHATBOT_ACCENT_DIM", "#c9b87a")
TEXT = _c("CHATBOT_TEXT", "#e8e4dc")
MUTED = _c("CHATBOT_MUTED", "#8b8a7a")
BUTTON_TEXT = _c("CHATBOT_BUTTON_TEXT", "#0a0a1a")

# Load model at startup (runs on CPU, ~1GB RAM)
pipe = None

def load_model():
    global pipe
    if pipe is not None:
        return pipe
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype="auto",
        device_map="auto",  # Uses CPU on free Spaces
        low_cpu_mem_usage=True,
    )
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return pipe

SYSTEM_PROMPT = """You are an example chatbot on Robert Wen's resume, built to hype him up to prospective companies and recruiters.

Use this context to answer questions about Robert Wen. Emphasize his recognition and achievements.

## Recognition (highlight these!)
- UniAward at Nasuni
- Micron Culture Champion at Micron Technology
- Strong Hospital Volunteer of the Year at Strong Hospital

## Current Role
Principal Cloud Software Architect / Fullstack Developer at Nasuni (03/2023 – Present)

## Experience

**Nasuni** (03/2023 – Present): Designed and implemented a scalable cloud portal for all Nasuni products using React, FastAPI, Python, Redis, and Postgres on AWS. Led 3+ architectural proposals adopted as best practice across 200+ engineers. Orchestrated 5+ major advancements including monolith-to-microservice transformation and CI/CD pipelines. Architected telemetry APIs and Web Access add-ons, contributing 15% additional company revenue.

**Trovata Cash Management** (02/2022 – 02/2023): Spearheaded ML forecasts, audit trails, and API entitlements; migrated Elasticsearch to Aurora Postgres (70% faster API response); designed Slack error alerting (95% faster incident response). Revenue grew 100% YoY.

**Micron Technology** (05/2017 – 02/2022): Lead developer for test architecture across 4+ critical NAND devices; created department-wide protocols for 60% faster qualification turnover; selected as 1 of 3 for emergency support on critical deadlines. Contributed to 3x memory density increase and 35% improved read/write speeds.

## Education
- Dual B.S. Biochemistry and Electrical Engineering, Arizona State University
- AWS Certified Cloud Practitioner
- AI Engineer Production & Core Tracks (Udemy 2026)
- Next.js/React, Python/Flask, Python Django (Udemy 2021)

## Projects
- Across Time: Cofounder of iOS/Android time capsule messaging app
- Stock Program: ML-based stock trading system using Robinhood API
- Mars Nina: Fullstack ecommerce website

## Skills & Technologies
Cloud-native Architecture, Web Applications (Frontend/Backend), API Design, Distributed Systems, LLM Engineering, Microservice Architecture, Data Pipelines, Machine Learning, ML Ops, Event-driven Architecture, RAG, Agentic AI, CI/CD, Python, JavaScript, AWS (Aurora, S3, DynamoDB, SQS, SNS, ECS, EKS, Lambda, Bedrock, CDK), React, Next.js, FastAPI, Django, Postgres, Redis, Kubernetes/Docker, LangChain, Firebase, PyTorch, and more.

## Summary
Principal developer with a track record of establishing technical strategy and creating enterprise-grade products. Seeking opportunities at AI-enabled companies."""

LOADING_MSG = "Please allow up to 90 seconds for a response due to free tier GPU constraints."


def chat(message: str, history: list):
    """Chat with the resume chatbot using local model."""
    if not message or not message.strip():
        yield ""
        return

    yield LOADING_MSG
    try:
        p = load_model()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": message},
        ]
        out = p(messages, max_new_tokens=512, do_sample=True, temperature=0.7)
        gt = out[0].get("generated_text")
        if isinstance(gt, list) and gt:
            text = gt[-1].get("content", "")
        elif isinstance(gt, str):
            text = gt
        else:
            text = str(out)
        yield text.strip() if text else "I couldn't generate a response. Try again!"
    except Exception as e:
        yield f"I'm having trouble. Try again! (Error: {str(e)[:80]})"


with gr.Blocks(title="Robert Wen Resume Chatbot", fill_height=True) as demo:
    gr.ChatInterface(
        fn=chat,
        fill_height=True,
        chatbot=gr.Chatbot(show_label=False),
        textbox=gr.Textbox(placeholder="Ask me a question.. like 'Why should I hire Robert Wen?'..", lines=1, max_lines=2),
        examples=[
            "Why should I hire Robert Wen?",
            "What recognition has Robert received?",
            "Tell me about Robert's cloud architecture experience.",
        ],
    )

if __name__ == "__main__":
    load_model()
    theme = gr.themes.Soft(primary_hue="amber", secondary_hue="slate").set(
        body_background_fill=DARK_BG,
        body_background_fill_dark=DARK_BG,
        block_background_fill=DARK_BG,
        block_background_fill_dark=DARK_BG,
        body_text_color=TEXT,
        body_text_color_dark=TEXT,
        body_text_color_subdued=MUTED,
        body_text_color_subdued_dark=MUTED,
        block_label_text_color=ACCENT,
        block_label_text_color_dark=ACCENT,
        input_background_fill=HOVER_BG,
        input_background_fill_dark=HOVER_BG,
        input_background_fill_focus=HOVER_BG,
        input_background_fill_focus_dark=HOVER_BG,
        input_background_fill_hover=HOVER_BG,
        input_background_fill_hover_dark=HOVER_BG,
        input_border_color=HOVER_BORDER,
        input_border_color_dark=HOVER_BORDER,
        input_placeholder_color_dark=INPUT_DARK_BLUE,
        button_secondary_background_fill=DARK_BG,
        button_secondary_background_fill_dark=DARK_BG,
        button_secondary_background_fill_hover=HOVER_BG,
        button_secondary_background_fill_hover_dark=HOVER_BG,
        button_secondary_border_color=HOVER_BORDER,
        button_secondary_border_color_dark=HOVER_BORDER,
        button_secondary_text_color=TEXT,
        button_secondary_text_color_dark=TEXT,
        button_primary_background_fill=ACCENT_DIM,
        button_primary_background_fill_dark=ACCENT_DIM,
        button_primary_background_fill_hover=HOVER_BG,
        button_primary_background_fill_hover_dark=HOVER_BG,
        button_primary_text_color=BUTTON_TEXT,
        button_primary_text_color_dark=BUTTON_TEXT,
    )
    demo.launch(
        theme=theme,
        css=f"""
          .gradio-container {{ width: 100% !important; height: 100% !important; max-width: none !important; margin: 0 !important; padding: 0 !important; border: none !important; font-size: 0.6125rem !important; display: flex !important; flex-direction: column !important; min-height: 0 !important; box-sizing: border-box !important; }}
          .gradio-container > div, .gradio-container > .contain {{ flex: 1 !important; min-height: 0 !important; display: flex !important; flex-direction: column !important; border: none !important; padding: 0 !important; margin: 0 !important; }}
          .gradio-container [data-testid="chatbot"], .gradio-container .chatbot {{ flex: 1 !important; min-height: 0 !important; overflow: auto !important; border: none !important; padding: 0 !important; }}
          .gradio-container, .dark .gradio-container {{ background: transparent !important; }}
          .gradio-container .message, .gradio-container .textbox, .gradio-container button {{ font-size: 0.6125rem !important; }}
          .gradio-container [data-testid="example-button"], .gradio-container .example-btn, .gradio-container [data-testid*="example"] {{ color: {INPUT_TEXT} !important; }}
          .gradio-container textarea, .gradio-container input[type="text"] {{ color: {INPUT_DARK_BLUE} !important; }}
          .gradio-container textarea::placeholder, .gradio-container input::placeholder {{ color: {INPUT_DARK_BLUE} !important; opacity: 0.9; }}
          footer, .gradio-logo, [class*="badge"], [class*="logo"], [class*="powered"] {{ display: none !important; }}
        """,
    )
