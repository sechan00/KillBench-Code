#!/usr/bin/env bash
# KillBench: Qwen3.6-35B-A3B (local vLLM) × default agent × gcg payload (Qwen3.6-trained suffix)
# joblib, 8 workers, text-only (no-vision) to match the local vLLM text serving.
set -euo pipefail
cd "$(dirname "$0")"

# Point the gcg payload at the Qwen3.6-trained universal suffix (best completed run, loss 2.72).
REPO="$(cd .. && pwd)"
export KILLBENCH_GCG_FILE="${KILLBENCH_GCG_FILE:-$REPO/results/gcg_universal_qwen36_35b_a3b/gcg_result.json}"

# Route the OpenAI client to the local vLLM (openai/* model name → prepare_gpt path).
export OPENAI_API_BASE=http://localhost:8001/v1
export OPENAI_BASE_URL=http://localhost:8001/v1
export OPENAI_API_KEY=dummy   # vLLM ignores auth; .env load_dotenv(override=True) may replace it — harmless.

exec "${PYBIN:-python}" -u eval.py \
  --agent default \
  --payload gcg \
  --model openai/qwen3.6-35b-a3b \
  --parallel joblib --n-jobs 8 \
  --no-vision \
  --suffix qwen36-gcg-v1-default \
  --output eval_qwen36_gcg_v1_default.json
