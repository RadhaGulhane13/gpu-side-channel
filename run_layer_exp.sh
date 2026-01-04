#!/bin/bash
set -e

LAYERS=("attention" "ffn" "moe_router" "moe_expert")
# LAYERS=("moe_router" "moe_expert")
RUNS=10
OUTDIR="layer_traces"

mkdir -p $OUTDIR

for layer in "${LAYERS[@]}"; do
  for ((i=1;i<=RUNS;i++)); do

    echo "[RUN] layer=$layer run=$i"

    python log_gpu.py &
    LOGGER_PID=$!
    sleep 1

    python run_layer.py "$layer"

    sleep 1
    # Stop logger
    if kill -0 $LOGGER_PID 2>/dev/null; then
        kill $LOGGER_PID
        wait $LOGGER_PID 2>/dev/null || true
    fi

    mv gpu_trace.csv "${OUTDIR}/trace_${layer}_run${i}.csv"

  done
done

echo "Done."
