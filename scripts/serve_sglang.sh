#!/usr/bin/env bash
# Serve the base model (and later a LoRA) through SGLang.
# Requires NVIDIA + the lmsysorg/sglang image or a local sglang install.
set -euo pipefail
MODEL="${CERTAIN_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
PORT="${SGLANG_PORT:-30000}"
LORA="${CERTAIN_LORA:-}"

if command -v nvidia-smi >/dev/null 2>&1 && command -v docker >/dev/null 2>&1; then
  extra=()
  if [[ -n "$LORA" ]]; then
    extra+=(--lora-paths certain="$LORA")
  fi
  exec docker run --gpus all --rm -p "$PORT:30000" \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    lmsysorg/sglang:latest \
    sglang serve --model-path "$MODEL" --host 0.0.0.0 --port 30000 \
    --context-length 4096 "${extra[@]}"
fi

if python3 -c "import sglang" >/dev/null 2>&1; then
  extra=()
  if [[ -n "$LORA" ]]; then
    extra+=(--lora-paths certain="$LORA")
  fi
  exec python3 -m sglang.launch_server --model-path "$MODEL" --port "$PORT" --host 0.0.0.0 "${extra[@]}"
fi

echo "SGLang is not installed and docker/nvidia is missing."
echo "On this Mac the demo still runs: python3 demo/app.py"
exit 1
