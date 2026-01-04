#!/bin/bash
set -e

#############################
# Experiment configuration #
#############################

MODELS=("dense_7b" "moe_8x7b")
PROMPTS=("neutral" "math" "code")
RUNS=10

OUTDIR="traces"
mkdir -p $OUTDIR

echo "Starting GPU side-channel experiment..."
echo "Models: ${MODELS[@]}"
echo "Prompts: ${PROMPTS[@]}"
echo "Runs per setting: $RUNS"
echo "Output dir: $OUTDIR"
echo "--------------------------------------"

#############################
# Main experiment loop     #
#############################

for model in "${MODELS[@]}"; do
  for prompt in "${PROMPTS[@]}"; do
    for ((r=1; r<=RUNS; r++)); do

      echo "[RUN] model=$model prompt=$prompt run=$r"

      OUTFILE="${OUTDIR}/trace_${model}_${prompt}_run${r}.csv"

      # Start GPU logger
      python log_gpu.py &
      LOGGER_PID=$!

      # Small delay to ensure logger starts
      sleep 1

      # Run inference
      python run_inference.py "$model" "$prompt"

      # Allow final samples
      sleep 1

      # Stop logger
      if kill -0 $LOGGER_PID 2>/dev/null; then
        kill $LOGGER_PID
        wait $LOGGER_PID 2>/dev/null || true
      fi

      # Save trace
      mv gpu_trace.csv "$OUTFILE"

    done
  done
done

echo "--------------------------------------"
echo "All experiments completed successfully."
