from huggingface_hub import InferenceClient


class Parse:

    client = InferenceClient("")

    def __init__(self, message, json):
        self.message = message
        self.json = json

    def fill_json(self):
        resp = client.text_generation(
            f'convert to JSON: {self.message}. please use the following schema: {self.json}" ',
            max_new_tokens=1000,  # arbitrary number
            seed=42,
            grammar={"type": "json", "value": self.json},
        )

        return resp
