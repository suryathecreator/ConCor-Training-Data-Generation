#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-python3.11}"
VENV="${VENV:-$REPO_ROOT/.venv}"
SAM3_VENV="${SAM3_VENV:-$REPO_ROOT/.venv-sam3}"
INSTALL_GPU="${INSTALL_GPU:-1}"
INSTALL_SAM3="${INSTALL_SAM3:-1}"

"$PYTHON" -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip wheel setuptools
"$VENV/bin/python" -m pip install -e "$REPO_ROOT[campaign,dev]"
"$VENV/bin/python" -m spacy download en_core_web_sm

if [ "$INSTALL_GPU" = "1" ]; then
  "$VENV/bin/python" -m pip install "vllm==0.26.0"
fi

if [ "$INSTALL_SAM3" = "1" ]; then
  SAM3_DIR="${SAM3_REPO_ROOT:-$REPO_ROOT/third_party/sam3}"
  if [ ! -d "$SAM3_DIR/.git" ]; then
    mkdir -p "$(dirname "$SAM3_DIR")"
    git clone https://github.com/facebookresearch/sam3.git "$SAM3_DIR"
  fi
  "$PYTHON" -m venv "$SAM3_VENV"
  "$SAM3_VENV/bin/python" -m pip install --upgrade pip wheel setuptools
  "$SAM3_VENV/bin/python" -m pip install -e "$REPO_ROOT[campaign]"
  "$SAM3_VENV/bin/python" -m pip install -e "$SAM3_DIR"
  "$SAM3_VENV/bin/python" -m spacy download en_core_web_sm
fi

VLLM_PYTHON="$VENV/bin/python" "$REPO_ROOT/scripts/install_wordnet.sh"
echo "Setup complete. Activate with: source $VENV/bin/activate"
echo "SAM3 environment: $SAM3_VENV"
echo "Before the first run, authenticate with Hugging Face and request access to facebook/sam3."
