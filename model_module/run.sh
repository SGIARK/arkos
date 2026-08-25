#!/bin/bash


export HF_TOKEN=""

docker run --gpus '"device=1"' \
  --shm-size 32g \
  -p 30000:30000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --env HF_TOKEN=$HF_TOKEN \
  --env PYTHONPATH=/sgl-workspace/sglang/python \
  --ipc=host \
  --entrypoint "" \
  lmsysorg/sglang:latest \
  python3 -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B \
    --tool-call-parser qwen25 \
    --reasoning-parser qwen3 \
    --host 0.0.0.0 \
    --port 30000


