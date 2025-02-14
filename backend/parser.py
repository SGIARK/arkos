"""
dependencies:
* huggingface_hub

classes:
* Parse

"""
# TODO: determine what this actually does

from huggingface_hub import InferenceClient

class Parse:
    """
    attributes:
    * message
    * json
    * client

    methods:
    * __init__(self, message, json)
    * fill_json(self)
    """
    def __init__(self, message, json):
        self.message = message
        self.json = json
        self.client = InferenceClient("")

    def fill_json(self):
        resp = self.client.text_generation(
            f'convert to JSON: {self.message}. please use the following schema: {self.json}" ',
            max_new_tokens=1000,  # arbitrary number
            seed=42,
            grammar={"type": "json", "value": self.json},
        )

        return resp
