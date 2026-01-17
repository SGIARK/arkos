import os
import sys

from openai import OpenAI

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config_module.loader import config

# Point to your running ArkOS agent
client = OpenAI(base_url=f"http://localhost:{config.get('app.port')}/v1", api_key="not-needed")


def test_agent(prompt: str):
    response = client.chat.completions.create(
        model="ark-agent", messages=[{"role": "user", "content": prompt}]
    )

    message = response.choices[0].message.content
    print("=== Agent Response ===")
    print(message)
    print("======================")
    return message


if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() in ["exit", "quit"]:
            break
        test_agent(user_input)
