#!/usr/bin/env bash
#SBATCH --job-name=bcc10k-site
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=10080
#SBATCH --requeue
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

set -euo pipefail
REPO_ROOT="${REPO_ROOT:?REPO_ROOT is required}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:?CAMPAIGN_ROOT is required}"
PORT="${SITE_PORT:-8765}"
SSH_TARGET="${SITE_SSH_TARGET:-user@login.cluster.example}"
READY="$CAMPAIGN_ROOT/site/READY.json"
PYTHON_BIN="${VLLM_PYTHON:-$REPO_ROOT/.venv/bin/python}"
unset VLLM_PYTHON

while [ ! -s "$READY" ]; do
  sleep 30
done

HOST="$(hostname)"
CAMPAIGN_ROOT="$CAMPAIGN_ROOT" HOST="$HOST" PORT="$PORT" SSH_TARGET="$SSH_TARGET" "$PYTHON_BIN" -c 'import json,os,time; from pathlib import Path; root=Path(os.environ["CAMPAIGN_ROOT"]); port=os.environ["PORT"]; host=os.environ["HOST"]; target=os.environ["SSH_TARGET"]; payload={"host":host,"port":int(port),"started_at":time.time(),"tunnel":"ssh -N -o ExitOnForwardFailure=yes -L {0}:{1}:{0} {2}".format(port,host,target)}; (root/"site"/"HOST.json").write_text(json.dumps(payload,indent=2)+"\n")'
exec "$PYTHON_BIN" -m http.server "$PORT" --directory "$CAMPAIGN_ROOT/site"
