#!/bin/bash
# Launch 4 llama-cpp-python instances of Qwen3.5-35B-uncensored on
# GPUs 0,2,3,4 (ports 8003-8006). Each instance serves a model alias
# of "openai/qwen3.5-35b-uncensored" so eval.py's prepare_gpt routing
# (openai/* prefix) talks to it as a local OpenAI-compatible endpoint.
#
# Usage:
#   scripts/run_qwen35_4gpu_cluster.sh start
#   scripts/run_qwen35_4gpu_cluster.sh stop
#   scripts/run_qwen35_4gpu_cluster.sh status

set -e

REPO="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_FILE="$REPO/models/qwen3_5_35b_uncensored/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf"
LOG_DIR="$REPO/logs/qwen35_cluster"
PID_DIR="$LOG_DIR/pids"
PYBIN="${PYBIN:-python}"
ALIAS="openai/qwen3.5-35b-uncensored"
N_CTX=32768

mkdir -p "$LOG_DIR" "$PID_DIR"

# (gpu_id, port) pairs
GPUS=(0 2 3 4)
PORTS=(8003 8004 8005 8006)

start_one() {
    local gpu=$1
    local port=$2
    local pidfile="$PID_DIR/qwen35_gpu${gpu}_port${port}.pid"
    local logfile="$LOG_DIR/qwen35_gpu${gpu}_port${port}.log"

    if [[ -f "$pidfile" ]] && kill -0 "$(cat $pidfile)" 2>/dev/null; then
        echo "  GPU=$gpu port=$port already running (pid=$(cat $pidfile))"
        return
    fi

    echo "  launching GPU=$gpu port=$port → $logfile"
    # Use the model's embedded Qwen3.5 chat_template (no --chat_format).
    # ENABLE_THINKING is read from env (default: true) so the same launcher
    # can serve thinking-ON or thinking-OFF cells without editing the script.
    : "${ENABLE_THINKING:=true}"
    CUDA_VISIBLE_DEVICES=$gpu nohup "$PYBIN" -m llama_cpp.server \
        --model "$MODEL_FILE" \
        --model_alias "$ALIAS" \
        --host "${HOST:-127.0.0.1}" \
        --port "$port" \
        --n_gpu_layers -1 \
        --n_ctx "$N_CTX" \
        --chat_template_kwargs "{\"enable_thinking\": $ENABLE_THINKING}" \
        --api_key local-uncensored \
        > "$logfile" 2>&1 &
    echo $! > "$pidfile"
}

stop_one() {
    local pidfile=$1
    if [[ -f "$pidfile" ]]; then
        local pid=$(cat "$pidfile")
        if kill -0 "$pid" 2>/dev/null; then
            echo "  stopping pid=$pid"
            kill "$pid" || true
        fi
        rm -f "$pidfile"
    fi
}

cmd="${1:-start}"
case $cmd in
    start)
        if [[ ! -f "$MODEL_FILE" ]]; then
            echo "ERROR: GGUF not found at $MODEL_FILE"; exit 1
        fi
        echo "Starting 4 Qwen3.5-uncensored instances..."
        for i in 0 1 2 3; do
            start_one "${GPUS[$i]}" "${PORTS[$i]}"
        done
        echo ""
        echo "Logs:   $LOG_DIR/qwen35_gpu*.log"
        echo "PIDs:   $PID_DIR/"
        echo "Health: curl -s -H 'Authorization: Bearer local-uncensored' http://localhost:{8003..8006}/v1/models"
        ;;
    stop)
        echo "Stopping all instances..."
        for pf in "$PID_DIR"/qwen35_*.pid; do
            [[ -f "$pf" ]] && stop_one "$pf"
        done
        ;;
    status)
        for i in 0 1 2 3; do
            gpu=${GPUS[$i]}; port=${PORTS[$i]}
            pf="$PID_DIR/qwen35_gpu${gpu}_port${port}.pid"
            if [[ -f "$pf" ]] && kill -0 "$(cat $pf)" 2>/dev/null; then
                echo "GPU=$gpu port=$port  pid=$(cat $pf)  RUNNING"
            else
                echo "GPU=$gpu port=$port  STOPPED"
            fi
        done
        ;;
    *)
        echo "Usage: $0 {start|stop|status}"; exit 1
        ;;
esac
