import os
import sys
from openai import OpenAI

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config_module.loader import config


# Point to your running ArkOS agent
client = OpenAI(
    base_url=f"http://localhost:{config.get('app.port')}/v1", api_key="not-needed"
)


def chat_stream(prompt: str, user_id: str = "anonymous"):
    """Send a message and stream the response."""
    print("ARK: ", end="", flush=True)

    stream = client.chat.completions.create(
        model="ark-agent",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        user=user_id,  # Standard OpenAI parameter
    )

    full_response = ""
    for chunk in stream:
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            print(content, end="", flush=True)
            full_response += content

    print()  # newline after response
    return full_response


def chat(prompt: str, user_id: str = "anonymous"):
    """Send a message and get full response (no streaming)."""
    response = client.chat.completions.create(
        model="ark-agent",
        messages=[{"role": "user", "content": prompt}],
        stream=False,
        user=user_id,  # Standard OpenAI parameter
    )

    message = response.choices[0].message.content
    print(f"ARK: {message}")
    return message


if __name__ == "__main__":
    print("ArkOS Chat Interface (type 'exit' to quit)")
    print("-" * 40)

    user_id = config.get("app.user_id") or "anonymous"
    print(f"Authenticated as: {user_id}")
    print("-" * 40)

    use_streaming = True  # Set to False to disable streaming

    while True:
        try:
            user_input = input("You: ")
            if user_input.strip().lower() in ["exit", "quit"]:
                break
            if not user_input.strip():
                continue

            if use_streaming:
                chat_stream(user_input, user_id=user_id)
            else:
                chat(user_input, user_id=user_id)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
