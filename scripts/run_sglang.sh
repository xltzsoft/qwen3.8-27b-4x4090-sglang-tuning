#!/bin/bash
export PATH=/opt/sgenv/bin:$PATH
cd /root
export CUDA_VISIBLE_DEVICES=0,1,2,3
exec /opt/sgenv/bin/python -m sglang.launch_server \
  --model-path /root/models/Qwen3.8-27B-FP8 \
  --tensor-parallel-size 4 \
  --context-length 262144 \
  --chunked-prefill-size 8192 \
  --max-prefill-tokens 32768 \
  --host 0.0.0.0 \
  --port 8000 \
  --mem-fraction-static 0.80 \
  --disable-prefill-cuda-graph \
  --load-format safetensors \
  --kv-cache-dtype fp8_e4m3 \
  --mm-feature-transport cpu \
  --attention-backend flashinfer \
  --mamba-ssm-dtype bfloat16 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --api-key sk-CHANGE-ME \
  --enable-metrics \
  --enable-cache-report \
  --log-level info
