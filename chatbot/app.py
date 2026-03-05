"""
Robert Wen Resume Chatbot - Open source Gradio app
Deploy to Hugging Face Spaces: https://huggingface.co/spaces
"""

import gradio as gr

# Try Hugging Face Inference API; fallback to keyword responses if unavailable
try:
    from huggingface_hub import InferenceClient
    client = InferenceClient()
    MODEL = "HuggingFaceH4/zephyr-7b-beta"
    USE_LLM = True
except Exception:
    USE_LLM = False

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
    """Chat with the resume chatbot."""
    if not message or not message.strip():
        return ""

    if USE_LLM:
        try:
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            for h in history:
                if isinstance(h, (list, tuple)) and len(h) >= 2:
                    messages.append({"role": "user", "content": h[0]})
                    messages.append({"role": "assistant", "content": h[1]})
                elif isinstance(h, dict):
                    messages.append({"role": h.get("role", "user"), "content": h.get("content", "")})
            messages.append({"role": "user", "content": message})
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                max_tokens=512,
                temperature=0.7,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"I'm having trouble reaching the AI right now. Try again in a moment! (Error: {str(e)[:80]})"
    else:
        # Fallback: simple keyword responses
        msg = message.lower()
        if "hire" in msg or "why" in msg:
            return "Robert Wen is a Principal Cloud Software Architect with proven impact: UniAward at Nasuni, Micron Culture Champion, and Strong Hospital Volunteer of the Year. He's led architectural proposals adopted across 200+ engineers, contributed 15% additional revenue at Nasuni, and driven 100% YoY revenue growth at Trovata. His expertise in cloud architecture, microservices, and LLM engineering makes him an exceptional hire."
        if "recognition" in msg or "award" in msg:
            return "Robert has been recognized with: UniAward (Nasuni), Micron Culture Champion (Micron Technology), and Strong Hospital Volunteer of the Year (Strong Hospital)."
        if "experience" in msg or "background" in msg:
            return "Robert has held senior roles at Nasuni (Principal Architect), Trovata (Senior Architect), and Micron (Software Architect). He specializes in cloud-native architecture, microservices, and AI/LLM engineering."
        return "Robert Wen is a Principal Cloud Software Architect with strong recognition at Nasuni, Micron, and Strong Hospital. Ask about his experience, skills, or why he'd be a great hire!"


with gr.Blocks(
    title="Robert Wen Resume Chatbot",
    theme=gr.themes.Soft(
        primary_hue="amber",
        secondary_hue="slate",
    ).set(
        body_background_fill="*neutral_50",
        block_background_fill="*neutral_100",
    ),
    css="""
    .gradio-container { max-width: 720px !important; margin: 0 auto !important; }
    """,
) as demo:
    gr.Markdown("### Chat with Robert's Resume")
    gr.ChatInterface(
        fn=chat,
        type="messages",
        placeholder="Ask me a question.. like why should I hire Robert Wen..",
        examples=[
            "Why should I hire Robert Wen?",
            "What recognition has Robert received?",
            "Tell me about Robert's cloud architecture experience.",
        ],
    )

if __name__ == "__main__":
    demo.launch()
