import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

api_key = os.getenv("DASHSCOPE_API_KEY")

if not api_key:
    raise RuntimeError("DASHSCOPE_API_KEY is missing")

client = OpenAI(
    api_key=api_key,
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
)

response = client.chat.completions.create(
    model="qwen3.7-plus",
    messages=[
        {
            "role": "user",
            "content": "Reply with exactly: ProofPilot is connected.",
        }
    ],
)

print(response.choices[0].message.content)
