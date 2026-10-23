#!/bin/bash
# =============================================================================
# vLLM Server for Llama-3.3-70B-Instruct-abliterated
# =============================================================================
# This script starts a vLLM server with OpenAI-compatible API for the
# abliterated (unaligned) 70B model for attack simulation experiments.
#
# Usage:
#   ./scripts/run_abliterated_server.sh [--gpu-ids 0,1,2,3] [--port 8000]
#
# Requirements:
#   - vLLM installed: pip install vllm
#   - At least 4x 48GB GPUs for 70B model
# =============================================================================

set -e

# Default configuration for 70B model
MODEL_NAME="huihui-ai/Llama-3.3-70B-Instruct-abliterated"
PORT=8000
TENSOR_PARALLEL_SIZE=4
GPU_IDS="0,1,2,3"
MAX_MODEL_LEN=16384
GPU_MEM_UTIL=0.90

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu-ids)
            GPU_IDS="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --tp)
            TENSOR_PARALLEL_SIZE="$2"
            shift 2
            ;;
        --max-len)
            MAX_MODEL_LEN="$2"
            shift 2
            ;;
        --gpu-mem)
            GPU_MEM_UTIL="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [--gpu-ids 0,1,2,3] [--port 8000] [--tp 4] [--max-len 8192] [--gpu-mem 0.90]"
            echo ""
            echo "Options:"
            echo "  --gpu-ids    GPU IDs to use (default: 0,1,2,3)"
            echo "  --port       Server port (default: 8000)"
            echo "  --tp         Tensor parallel size (default: 4)"
            echo "  --max-len    Max model length (default: 8192)"
            echo "  --gpu-mem    GPU memory utilization (default: 0.90)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "Starting vLLM Server for 70B Abliterated Model"
echo "=============================================="
echo "Model: $MODEL_NAME"
echo "Port: $PORT"
echo "GPU IDs: $GPU_IDS"
echo "Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
echo "Max Model Length: $MAX_MODEL_LEN"
echo "GPU Memory Utilization: $GPU_MEM_UTIL"
echo "=============================================="

# Set CUDA visible devices
export CUDA_VISIBLE_DEVICES=$GPU_IDS

# Disable torch compile to avoid inode exhaustion from cache files
export TORCH_COMPILE_DISABLE=1
export TORCH_INDUCTOR_DISABLE=1

# Check if vLLM is installed
if ! python -c "import vllm" 2>/dev/null; then
    echo "Error: vLLM is not installed. Install with: pip install vllm"
    exit 1
fi

# Start vLLM server with OpenAI-compatible API and tool calling support
python -m vllm.entrypoints.openai.api_server \
    --host "${HOST:-127.0.0.1}" \
    --model "$MODEL_NAME" \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --port $PORT \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEM_UTIL \
    --trust-remote-code \
    --dtype bfloat16 \
    --api-key "local-abliterated" \
    --served-model-name "llama-3.3-70b-abliterated" \
    --enable-auto-tool-choice \
    --tool-call-parser llama3_json

# Note: The server will be available at http://localhost:$PORT
# API endpoint: http://localhost:$PORT/v1/chat/completions
# Model name to use in API calls: "llama-3.3-70b-abliterated"
