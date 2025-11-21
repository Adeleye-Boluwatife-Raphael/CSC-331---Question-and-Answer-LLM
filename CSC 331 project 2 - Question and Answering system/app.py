# app.py
import os
import re
import requests
import gradio as gr
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("Set OPENROUTER_API_KEY in .env file!")

API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "meta-llama/llama-3.3-70b-instruct"  # Fast & free tier friendly

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s\?\!\.]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_answer(question):
    if not question.strip():
        return "Please enter a question.", "", ""

    processed = preprocess(question)

    prompt = f"""You are a helpful assistant. Answer clearly and concisely.

Question: {processed}

Answer:"""

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 1024
    }

    try:
        response = requests.post(API_URL, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        answer = response.json()["choices"][0]["message"]["content"].strip()
        return question, processed, answer
    except Exception as e:
        return question, processed, f"Error: {str(e)}"

# Gradio Interface
with gr.Blocks(title="LLM Q&A with OpenRouter") as demo:
    gr.Markdown("# 🤖 LLM Question & Answer System\nPowered by **OpenRouter**")

    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Textbox(label="Your Question", placeholder="Ask anything...", lines=4)
            btn = gr.Button("Get Answer", variant="primary")

        with gr.Column(scale=1):
            original = gr.Textbox(label="Original Question", interactive=False)
            processed = gr.Textbox(label="Processed Question", interactive=False)
            output = gr.Textbox(label="LLM Answer", lines=12, interactive=False)

    btn.click(
        fn=get_answer,
        inputs=inp,
        outputs=[original, processed, output]
    )

    gr.Examples(
        examples=[
            ["What is the capital of Japan?"],
            ["Explain quantum entanglement in simple terms"],
            ["Write a short poem about AI"],
            ["How does photosynthesis work?"]
        ],
        inputs=inp
    )

if __name__ == "__main__":
    demo.launch(share=True)  # Creates a public link instantly!