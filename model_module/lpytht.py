from llama_cpp import Llama
import yaml

with open("../config_module/config.yaml", "r") as file:
    configuration = yaml.safe_load(file)

model_id = configuration["model_path"]

llm = Llama(model_path=model_id)


print(
    llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": "You are an assistant who does humans bidding",
            },
            {"role": "user", "content": "Give me a bedtime story."},
        ]
    )
)
