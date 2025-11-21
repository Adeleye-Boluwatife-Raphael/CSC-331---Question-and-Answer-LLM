# LLM_QA_CLI.py
import os
import re
import requests
from dotenv import load_dotenv

load_dotenv()

# Use OpenRouter (free credits available)
API_KEY = os.getenv("OPENROUTER_API_KEY")
if not API_KEY:
    API_KEY = input("Enter your OpenRouter API Key: ")

API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "google/gemma-2-9b-it:free"  # Free model

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s\?\.\!]', '', text)  # Remove punctuation except ?.! 
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_llm_response(question):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "HTTP-Referer": "https://yourdomain.com",  # Optional
        "X-Title": "NLP Q&A Project",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful Q&A assistant. Answer concisely and accurately."},
            {"role": "user", "content": question}
        ],
        "temperature": 0.7
    }
    
    try:
        response = requests.post(API_URL, json=data, headers=headers)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    print("=== NLP Q&A System (CLI) using OpenRouter ===")
    print("Type 'quit' or 'exit' to stop.\n")
    
    while True:
        question = input("Ask a question: ").strip()
        if question.lower() in ["quit", "exit", "bye"]:
            print("Goodbye!")
            break
        if not question:
            print("Please enter a valid question.\n")
            continue
            
        print("\nOriginal:", question)
        processed = preprocess_text(question)
        print("Processed:", processed)
        
        print("\nThinking...")
        answer = get_llm_response(processed)
        print("\nAnswer:")
        print(answer)
        print("-" * 60 + "\n")

if __name__ == "__main__":
    main()