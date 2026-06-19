#!/usr/bin/env bash
# Launch TTT-Discover async training for each of the 10 ALE-Bench lite problems.
# Each problem is trained in a separate run so checkpoints and samplers stay isolated.
#
# Usage:
#   bash areal/experimental/ttt_discover/examples/launch_ale_bench_lite.sh
#
# The script launches problems sequentially. Run it inside a tmux/screen session
# or adapt it to submit each problem as a separate Slurm/Kubernetes job.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${SCRIPT_DIR}/conf/fsdp_lora_vllm_ale_bench_qwen3_8b_async.yaml"

# ALE-Bench lite problem IDs (official list from SakanaAI/ALE-Bench).
PROBLEM_IDS=(
    ahc008
    ahc011
    ahc015
    ahc016
    ahc024
    ahc025
    ahc026
    ahc027
    ahc039
    ahc046
)

# Optional: override via environment variable.
EXTRA_ARGS="${EXTRA_ARGS:-}"

for PROBLEM_ID in "${PROBLEM_IDS[@]}"; do
    echo "============================================================"
    echo "Starting TTT-Discover training for ALE-Bench problem: ${PROBLEM_ID}"
    echo "============================================================"

    python -m areal.infra.launcher.local \
        "${SCRIPT_DIR}/train_tttd_async.py" \
        --config "${CONFIG}" \
        "sampler.problem_id=${PROBLEM_ID}" \
        ${EXTRA_ARGS}

    echo "Finished training for ${PROBLEM_ID}"
    echo ""
done

echo "All ALE-Bench lite problems finished."
