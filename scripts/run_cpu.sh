#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m unittest discover -s tests -v
python3 scripts/kill_test.py
python3 scripts/generate_data.py
python3 eval/run_audit.py --constructed
python3 scripts/train_sft.py --dry-run
python3 scripts/train_grpo.py --reward certificate --dry-run
python3 scripts/train_grpo.py --reward outcome --dry-run
echo "CPU pipeline ok. Demo: python3 demo/app.py"
