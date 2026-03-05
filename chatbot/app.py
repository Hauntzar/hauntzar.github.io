"""
Robert Wen Resume Chatbot - Runs a local open-source model, no API keys needed.
Deploy to Hugging Face Spaces: https://huggingface.co/spaces
"""

import os
import gradio as gr

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

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
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
    )
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


def chat(message: str, history: list) -> str:
    """Chat with the resume chatbot using local model."""
    if not message or not message.strip():
        return ""

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
        return text.strip() if text else "I couldn't generate a response. Try again!"
    except Exception as e:
        return f"I'm having trouble. Try again! (Error: {str(e)[:80]})"


with gr.Blocks(title="Robert Wen Resume Chatbot") as demo:
    gr.Markdown("### Robert's Resume Chatbot")
    gr.ChatInterface(
        fn=chat,
        textbox=gr.Textbox(placeholder="Ask me a question.. like 'Why should I hire Robert Wen?'.."),
        examples=[
            "Why should I hire Robert Wen?",
            "What recognition has Robert received?",
            "Tell me about Robert's cloud architecture experience.",
        ],
    )

if __name__ == "__main__":
    load_model()
    demo.launch(
        theme=gr.themes.Soft(
            primary_hue="amber",
            secondary_hue="slate",
        ).set(
            body_background_fill="#0a0a1a",
            block_background_fill="#1a1a3e",
        ),
        css="""
          .gradio-container { max-width: 520px !important; margin: 0 auto !important; padding: 12px !important; }
          .gradio-container, .dark .gradio-container { background: transparent !important; }
        """,
    )
