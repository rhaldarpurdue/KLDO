import os
from openai import OpenAI

import sys

client = OpenAI(
    api_key="988e77ea0a9549928c607b3bedbab192",
    base_url="https://api.aimlapi.com",
)

response = client.chat.completions.create(
    model="cognitivecomputations/dolphin-2.5-mixtral-8x7b",
    messages=[
        {
            "role": "system",
            "content": "You are an AI assistant who knows everything.",
        },
        {
            "role": "user",
            "content": str(sys.argv[1])
        },
    ],
)

message = response.choices[0].message.content
print(f"Assistant: {message}")