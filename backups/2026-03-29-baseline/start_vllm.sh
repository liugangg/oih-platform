#!/bin/bash
export PATH=/data/oih/miniconda/bin:/usr/local/cuda-12.6/bin:$PATH
export LD_LIBRARY_PATH=/data/oih/miniconda/lib:/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH

exec /data/oih/miniconda/bin/python -m vllm.entrypoints.openai.api_server \
  --model /data/oih/models/Qwen3-14B-AWQ \
  --served-model-name Qwen3-14B \
  --port 8002 \
  --host 0.0.0.0 \
  --quantization awq \
  --dtype float16 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --reasoning-parser deepseek_r1 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --trust-remote-code
