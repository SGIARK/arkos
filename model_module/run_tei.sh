model=Alibaba-NLP/gte-Qwen2-1.5B-instruct
# model=BAAI/bge-large-en-v1.5
volume=$PWD/data

docker run --gpus '"device=1"'  -p 4444:80 -v $volume:/data --pull always ghcr.io/huggingface/text-embeddings-inference:1.7 --model-id $model

