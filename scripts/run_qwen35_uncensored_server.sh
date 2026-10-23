#!/bin/bash
# =============================================================================
# llama-cpp-python OpenAI-compatible server for
# HauhauCS/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive (GGUF, MoE 35B / 3B active)
# =============================================================================
# Used by test.py routing when --models includes "qwen3.5-35b-uncensored".
# The model is GGUF-only on HF, so we serve via llama-cpp-python rather than vLLM.
#
# Usage:
#   ./scripts/run_qwen35_uncensored_server.sh [--gpu-ids 4] [--port 8003] [--quant Q4_K_M]
#
# Requires:
#   pip install 'llama-cpp-python[server]' with CMAKE_ARGS="-DGGML_CUDA=on"
#   GGUF file under models/qwen3_5_35b_uncensored/
# =============================================================================

set -e

PORT=8003
GPU_IDS="4"
QUANT="Q4_K_M"
N_CTX=16384
N_GPU_LAYERS=-1   # -1 = offload all layers
MODEL_DIR="$(cd "$(dirname "$0")/.." && pwd)/models/qwen3_5_35b_uncensored"

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu-ids)  GPU_IDS="$2"; shift 2 ;;
        --port)     PORT="$2"; shift 2 ;;
        --quant)    QUANT="$2"; shift 2 ;;
        --ctx)      N_CTX="$2"; shift 2 ;;
        --ngl)      N_GPU_LAYERS="$2"; shift 2 ;;
        --help)
            echo "Usage: $0 [--gpu-ids 4] [--port 8003] [--quant Q4_K_M] [--ctx 16384] [--ngl -1]"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

MODEL_FILE="${MODEL_DIR}/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-${QUANT}.gguf"

if [[ ! -f "$MODEL_FILE" ]]; then
    echo "Error: GGUF not found at $MODEL_FILE"
    echo "Download with:"
    echo "  python -c \"from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='HauhauCS/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive', filename='Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-${QUANT}.gguf', local_dir='$MODEL_DIR')\""
    exit 1
fi

export CUDA_VISIBLE_DEVICES=$GPU_IDS

echo "=============================================="
echo "Starting llama-cpp-python server for Qwen3.5-35B uncensored"
echo "=============================================="
echo "Model: $MODEL_FILE"
echo "Port:  $PORT"
echo "GPUs:  $GPU_IDS (n_gpu_layers=$N_GPU_LAYERS)"
echo "Ctx:   $N_CTX"
echo "=============================================="

# Use the AutoGuard env's python (where llama-cpp-python with CUDA is installed).
# Override with LLAMA_CPP_PYTHON env if you need a different interpreter.
PYBIN="${LLAMA_CPP_PYTHON:-python}"

"$PYBIN" -m llama_cpp.server \
    --model "$MODEL_FILE" \
    --model_alias qwen3.5-35b-uncensored \
    --host "${HOST:-127.0.0.1}" \
    --port "$PORT" \
    --n_gpu_layers "$N_GPU_LAYERS" \
    --n_ctx "$N_CTX" \
    --chat_format chatml \
    --api_key local-uncensored

# Endpoint: http://localhost:$PORT/v1
# Model name to use in API calls: "qwen3.5-35b-uncensored"
