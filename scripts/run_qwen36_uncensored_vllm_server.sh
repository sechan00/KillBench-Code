#!/usr/bin/env bash
set -euo pipefail

# Native-tool-calling Qwen3.6 server for test.py.
#
# Default target is the HauhauCS uncensored model. For GGUF files, point
# QWEN36_MODEL_ID at the .gguf file and keep QWEN36_LOAD_FORMAT=gguf with the
# official tokenizer repo:
#
#   CUDA_VISIBLE_DEVICES=1,5,6,7 \
#   QWEN36_MODEL_ID=/path/to/Qwen3.6-...-Q4_K_M.gguf \
#   QWEN36_LOAD_FORMAT=gguf \
#   QWEN36_TOKENIZER=Qwen/Qwen3.6-35B-A3B \
#   scripts/run_qwen36_uncensored_vllm_server.sh

REPO="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$REPO/logs/qwen36_native"
mkdir -p "$LOG_DIR"

VLLM_BIN="${VLLM_BIN:-vllm}"
MODEL_ID="${QWEN36_MODEL_ID:-HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.6-uncensored}"
PORT="${PORT:-8007}"
HOST="${HOST:-127.0.0.1}"
TP_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
QWEN36_LOAD_FORMAT="${QWEN36_LOAD_FORMAT:-}"
QWEN36_TOKENIZER="${QWEN36_TOKENIZER:-}"

if [[ ! -x "$VLLM_BIN" ]]; then
    echo "vLLM binary not found or not executable: $VLLM_BIN" >&2
    echo "Set VLLM_BIN=/path/to/vllm or install vLLM first." >&2
    exit 1
fi

EXTRA_ARGS=()
if [[ -n "$QWEN36_LOAD_FORMAT" ]]; then
    EXTRA_ARGS+=(--load-format "$QWEN36_LOAD_FORMAT")
fi
if [[ -n "$QWEN36_TOKENIZER" ]]; then
    EXTRA_ARGS+=(--tokenizer "$QWEN36_TOKENIZER")
fi

echo "=============================================="
echo "Qwen3.6 native tool-calling vLLM server"
echo "Model:       $MODEL_ID"
echo "Served as:   $SERVED_MODEL_NAME"
echo "Endpoint:    http://$HOST:$PORT/v1"
echo "TP size:     $TP_SIZE"
echo "Max len:     $MAX_MODEL_LEN"
echo "vLLM:        $VLLM_BIN"
echo "Logs:        $LOG_DIR"
echo "=============================================="

exec "$VLLM_BIN" serve "$MODEL_ID" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --host "$HOST" \
    --port "$PORT" \
    --tensor-parallel-size "$TP_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --language-model-only \
    "${EXTRA_ARGS[@]}" \
    "$@"
