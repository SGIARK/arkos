from flask import Flask, request, jsonify, render_template
from llama_cpp import Llama

app = Flask(__name__)


### TUT: use the following (flask --app hello run) to start the server


# Initialize your Llama model
llm = Llama(
    model_path="/Users/velocity/Documents/Holder/Project/CodingStuff/VICUNA/llama.cpp/models/Mistral/mistral-7b-instruct-v0.2.Q6_K.gguf",
    n_ctx=2048,
)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate_text():
    try:
        data = request.json
        chat_header = "You a helpful AI assistant, answer the following in a complete sentence. Do not generate more than instructed\n\n"
        prompt = chat_header + data["prompt"]

        # Generate text using llama_cpp model
        results = llm(prompt, max_tokens=128)["choices"][0]["text"]
        print(results)

        return jsonify(
            {"generatedText": results}
        )  # Ensure 'generatedText' matches what frontend expects
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
