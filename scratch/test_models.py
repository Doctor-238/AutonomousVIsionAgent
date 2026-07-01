import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get("OPENROUTER_API_KEY")

models = [
    "google/gemma-2-9b-it:free",
    "meta-llama/llama-3.1-8b-instruct:free",
    "qwen/qwen-2-7b-instruct:free",
    "microsoft/phi-3-mini-128k-instruct:free",
    "openrouter/auto:free"
]

prompt = """
Extract the constraints from this text and return ONLY a JSON object.
Text: "노란색은 제외하고 빨간색과 파란색을 포함하여 2개만 배달해"
Format: {"delivery_limit": int, "priority_colors": ["color1"], "skipped_colors": ["color2"]}
"""

for model in models:
    print(f"\n--- Testing {model} ---")
    try:
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=15
        )
        if resp.status_code == 200:
            print("Response:", resp.json()["choices"][0]["message"]["content"])
        else:
            print("Error:", resp.text)
    except Exception as e:
        print("Exception:", str(e))
