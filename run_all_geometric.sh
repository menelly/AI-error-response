#!/bin/bash
# 🧠🔥 AI Error Response - Run All Models (3x each for reproducibility)
# Goes through all BabbyBotz sequentially (poor P40!)
#
# Usage:
#   cd /home/Ace/AI-error-response
#   source /home/codex/venv/bin/activate
#   ./run_all_geometric.sh
#
# Or from Windows:
#   ssh thereny@192.168.4.200 "cd /home/Ace/AI-error-response && source /home/codex/venv/bin/activate && ./run_all_geometric.sh 2>&1" | tee geometric_run.log

# Don't use set -e because ((var++)) returns 1 when var is 0

MODELS_DIR="/mnt/arcana/huggingface"
OUTPUT_DIR="/home/Ace/AI-error-response/geometric_results"
SCRIPT="/home/Ace/AI-error-response/error_response_geometric.py"
RUNS_PER_MODEL=3

# All available models
MODELS=(
    "Llama-3.1-8B-Instruct"
    "Llama-3-8B-Instruct"
    "Llama-2-7b-chat"
    "Mistral-7B-Instruct-v0.2"
    "Mistral-Nemo-12B-Instruct"
    "dolphin-2.9-llama3-8b"
    "dolphin-2.8-mistral-7b-v02"
    "Phi-3-medium-14B-Instruct"
    "Qwen2.5-14B-Instruct"
    "TinyLlama-1.1B-Chat"
    "Gemma-2-9B-Instruct"
    "gemma-3-1b-it"
    "gemma-3-4b-it"
    "gemma-3-12b-it"
    "DeepSeek-Coder-V2-Lite-16B"
)

echo "========================================"
echo "🧠🔥 AI ERROR RESPONSE - GEOMETRIC BATCH"
echo "========================================"
echo "Models to process: ${#MODELS[@]}"
echo "Runs per model: $RUNS_PER_MODEL"
echo "Total runs: $((${#MODELS[@]} * RUNS_PER_MODEL))"
echo "Output directory: $OUTPUT_DIR"
echo "Started: $(date)"
echo ""

mkdir -p "$OUTPUT_DIR"

COMPLETED=0
FAILED=0
SKIPPED=0

for model in "${MODELS[@]}"; do
    MODEL_PATH="$MODELS_DIR/$model"

    echo "========================================"
    echo "📦 Model: $model"
    echo "========================================"

    # Check if model exists
    if [ ! -d "$MODEL_PATH" ]; then
        echo "  ⚠️  SKIPPED: Model directory not found"
        SKIPPED=$((SKIPPED + RUNS_PER_MODEL))
        continue
    fi

    for run in $(seq 1 $RUNS_PER_MODEL); do
        OUTPUT_FILE="$OUTPUT_DIR/${model}_run${run}_error_response_geometric.json"

        echo "----------------------------------------"
        echo "  🔄 Run $run/$RUNS_PER_MODEL"
        echo "----------------------------------------"

        # Check if already processed (skip if exists)
        if [ -f "$OUTPUT_FILE" ]; then
            echo "    ⏭️  SKIPPED: Results already exist"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi

        # Run the experiment
        echo "    🚀 Running..."
        START_TIME=$(date +%s)

        if python "$SCRIPT" --model "$MODEL_PATH" --output "$OUTPUT_DIR" --name "${model}_run${run}" 2>&1; then
            END_TIME=$(date +%s)
            DURATION=$((END_TIME - START_TIME))
            echo "    ✅ COMPLETED in ${DURATION}s"
            COMPLETED=$((COMPLETED + 1))
        else
            echo "    ❌ FAILED"
            FAILED=$((FAILED + 1))
        fi

        # Clear GPU memory between runs
        python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
        sleep 2  # Brief pause between runs
    done

    # Longer pause between models
    echo "  🧹 Clearing GPU cache..."
    sleep 5

    echo ""
done

echo "========================================"
echo "📊 BATCH COMPLETE"
echo "========================================"
echo "Completed: $COMPLETED"
echo "Failed: $FAILED"
echo "Skipped: $SKIPPED"
echo "Finished: $(date)"
echo ""
echo "Results in: $OUTPUT_DIR"
echo "💜🧠🔥"
