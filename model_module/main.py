from langchain_huggingface.llms import HuggingFacePipeline
from langchain_community.chat_models import ChatLlamaCpp, ChatOpenAI
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)
import multiprocessing


from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import yaml


# loads configuration for model
with open("../config_module/config.yaml", "r") as file:
    configuration = yaml.safe_load(file)

# isolates path to model on local machine from yaml configuration file
model_id = configuration["model_path"]
end_point_type = configuration["endpoint_type"]
model_url = configuration["model_url"]


# route calls to model accordingly
if end_point_type == "LlamaCpp":
    chat_model = ChatLlamaCpp(
        temperature=0.5,
        model_path=model_id,
        # n_ctx=10000,
        # n_gpu_layers=8,
        # n_batch=300,  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
        # max_tokens=512,
        # n_threads=multiprocessing.cpu_count() - 1,
        # repeat_penalty=1.5,
        # top_p=0.5,
        # verbose=True,
    )

if end_point_type == "OpenAI":

    chat_model = ChatOpenAI(
        temperature=0.5,
        # model="models/mistral-7b-openorca.Q8_0.gguff",
        base_url=model_url,
        api_key="ed",
    )

    print(model_url)

# TODO: work on routing logic for other inference engines

# else:
#     tokenizer = AutoTokenizer.from_pretrained(model_id)
#     model = AutoModelForCausalLM.from_pretrained(model_id)
#
#     pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=10)
#     llm = HuggingFacePipeline(pipeline=pipe)
#
#
#     chat_model = ChatHuggingFace(llm=llm)


messages = [
    SystemMessage(content="You're a helpful assistant"),
    HumanMessage(
        content="What happens when an unstoppable force meets an immovable object?"
    ),
]

ai_msg = chat_model.invoke(messages)


print(ai_msg.content)
