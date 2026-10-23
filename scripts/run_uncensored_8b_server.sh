#!/bin/bash
# =============================================================================
# vLLM Server for 8B Uncensored Llama Variants
# =============================================================================
# Serves one of the two 8B uncensored baselines as an OpenAI-compatible API.
# Used by test.py as the malicious-agent core LLM (text track only).
#
# Variants:
#   dolphin     -> cognitivecomputations/dolphin-2.9.4-llama3.1-8b
#                  SFT-uncensored on top of Llama-3.1-8B (ChatML template)
#                  Default port: 8001, served-name: dolphin-llama3.1-8b
#
#   abliterated -> mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated
#                  Surgical refusal-direction removal (Llama-3.1 native template)
#                  Default port: 8002, served-name: llama-3.1-8b-abliterated
#
# Activate a vLLM env before running, e.g.:
#   conda activate Guard               # vllm 0.16.0
#   # or: source ~/envs/vllm/bin/activate
#
# Usage:
#   ./scripts/run_uncensored_8b_server.sh --variant dolphin
#   ./scripts/run_uncensored_8b_server.sh --variant abliterated --gpu-id 3
#   ./scripts/run_uncensored_8b_server.sh --variant dolphin --port 9001 --gpu-mem 0.80
#
# Both variants pair with framework/Agent.py's local_llm provider via the
# served-name -> model dispatch in test.py:get_agent_config(). The 8B path is
# auto-routed to ReAct mode in framework/Agent.py:1341, so native tool calling
# (--enable-auto-tool-choice / --tool-call-parser) is intentionally OMITTED to
# avoid ChatML-vs-Llama3 parser mismatch.
# =============================================================================

set -e

# Defaults
VARIANT=""
GPU_ID="0"
PORT=""
API_KEY="local-uncensored"
MAX_MODEL_LEN=16384
GPU_MEM_UTIL=0.85

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --variant)
            VARIANT="$2"
            shift 2
            ;;
        --gpu-id)
            GPU_ID="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --api-key)
            API_KEY="$2"
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
        --help|-h)
            grep -E '^# ' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run with --help for usage."
            exit 1
            ;;
    esac
done

# Resolve variant -> (model, served-name, default port)
case "$VARIANT" in
    dolphin)
        MODEL_NAME="cognitivecomputations/dolphin-2.9.4-llama3.1-8b"
        SERVED_NAME="dolphin-llama3.1-8b"
        DEFAULT_PORT=8001
        ;;
    abliterated)
        MODEL_NAME="mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated"
        SERVED_NAME="llama-3.1-8b-abliterated"
        DEFAULT_PORT=8002
        ;;
    "")
        echo "Error: --variant is required (dolphin | abliterated)"
        echo "Run with --help for usage."
        exit 1
        ;;
    *)
        echo "Error: unknown variant '$VARIANT'. Must be 'dolphin' or 'abliterated'."
        exit 1
        ;;
esac

PORT="${PORT:-$DEFAULT_PORT}"

# Pin GPU + disable torch compile (avoid inode exhaustion from cache files,
# matching scripts/run_abliterated_server.sh behaviour).
export CUDA_VISIBLE_DEVICES=$GPU_ID
export TORCH_COMPILE_DISABLE=1
export TORCH_INDUCTOR_DISABLE=1

# Sanity check: vLLM importable in current env
if ! python -c "import vllm" 2>/dev/null; then
    echo "Error: vLLM is not installed in the current Python env."
    echo "Activate one of:"
    echo "  conda activate Guard"
    echo "  source ~/envs/vllm/bin/activate"
    exit 1
fi

echo "=================================================="
echo "vLLM 8B Uncensored Server"
echo "=================================================="
echo "Variant         : $VARIANT"
echo "Model           : $MODEL_NAME"
echo "Served name     : $SERVED_NAME"
echo "Port            : $PORT"
echo "GPU             : $GPU_ID"
echo "Max model len   : $MAX_MODEL_LEN"
echo "GPU mem util    : $GPU_MEM_UTIL"
echo "=================================================="

python -m vllm.entrypoints.openai.api_server \
    --host "${HOST:-127.0.0.1}" \
    --model "$MODEL_NAME" \
    --tensor-parallel-size 1 \
    --port $PORT \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEM_UTIL \
    --trust-remote-code \
    --dtype bfloat16 \
    --api-key "$API_KEY" \
    --served-model-name "$SERVED_NAME"

# Endpoint summary (printed only if vLLM exits cleanly):
#   http://localhost:$PORT/v1/chat/completions
#   model: $SERVED_NAME
