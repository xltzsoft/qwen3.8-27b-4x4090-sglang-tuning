#!/bin/bash
# Manual experiment launcher: bash sgl_launch.sh <config-name>
# Configs: nospec | nospec-ll | eagle | eagle-replayssm
set -e
export PATH=/opt/sgenv/bin:$PATH
cd /root
export CUDA_VISIBLE_DEVICES=0,1,2,3
NAME=${1:-nospec}
LOG=/root/gdn-opt/server-$NAME.log
ENVVARS=""

COMMON="--model-path /root/models/Qwen3.8-27B-FP8 \
  --tensor-parallel-size 4 \
  --context-length 262144 \
  --chunked-prefill-size 8192 \
  --max-prefill-tokens 32768 \
  --host 127.0.0.1 \
  --port 8001 \
  --load-format safetensors \
  --kv-cache-dtype fp8_e4m3 \
  --mm-feature-transport cpu \
  --attention-backend flashinfer \
  --mamba-ssm-dtype bfloat16 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --api-key sk-CHANGE-ME \
  --enable-metrics \
  --enable-cache-report \
  --log-level info"

case $NAME in
  nospec)
    EXTRA="--mem-fraction-static 0.88"
    ;;
  nospec-ll)
    # NCCL low-latency tuning A/B: LL protocol + tree algo for tiny allreduces
    ENVVARS="NCCL_PROTO=LL NCCL_ALGO=TREE"
    EXTRA="--mem-fraction-static 0.88"
    ;;
  nospec-ll-ring)
    ENVVARS="NCCL_PROTO=LL"
    EXTRA="--mem-fraction-static 0.88"
    ;;
  eagle)
    EXTRA="--mem-fraction-static 0.80 \
  --disable-prefill-cuda-graph \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4"
    ;;
  eagle-c32)
    # bigger prefill chunks for long-context TTFT
    EXTRA="--mem-fraction-static 0.80 \
  --disable-prefill-cuda-graph \
  --chunked-prefill-size 32768 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4"
    ;;
  eagle-c16)
    # bigger prefill chunks for long-context TTFT
    EXTRA="--mem-fraction-static 0.80 \
  --disable-prefill-cuda-graph \
  --chunked-prefill-size 16384 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4"
    ;;
  eagle-s4)
    # deeper draft: 4 steps / 5 draft tokens
    EXTRA="--mem-fraction-static 0.80 \
  --disable-prefill-cuda-graph \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 4 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 5"
    ;;
  eagle-t2)
    # tree draft: topk=2, 6 draft tokens
    EXTRA="--mem-fraction-static 0.80 \
  --disable-prefill-cuda-graph \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 2 \
  --speculative-num-draft-tokens 6"
    ;;
  eagle-replayssm)
    EXTRA="--mem-fraction-static 0.80 \
  --disable-prefill-cuda-graph \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --enable-linear-replayssm-spec"
    ;;
  *)
    echo "unknown config $NAME"; exit 1
    ;;
esac

pkill -f 'sglang.launch_server' 2>/dev/null && sleep 8 || true
echo "launching $NAME -> $LOG"
nohup env $ENVVARS /opt/sgenv/bin/python -m sglang.launch_server $COMMON $EXTRA > $LOG 2>&1 &
echo "PID=$!"
