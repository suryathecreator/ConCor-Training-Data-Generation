#!/usr/bin/env bash
#SBATCH --job-name=bcc-q38-python-pack
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

set -euo pipefail
REPO_ROOT="${REPO_ROOT:?REPO_ROOT is required}"
cd "$REPO_ROOT"
exec "$REPO_ROOT/scripts/package_qwen_python_runtime.sh"
