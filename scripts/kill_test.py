#!/usr/bin/env python3
"""Hour-0 kill test: SGLang answers, AlignTune takes one GRPO step.

On a machine without NVIDIA + the sponsor libraries this writes a structured
skip report and still exercises the CERTAIN reward loop (the part that can
run on CPU).
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from certain.prompts import wrap_completion
from certain.rewards import certificate_reward, group_advantages, outcome_reward


def _has_nvidia() -> bool:
    try:
        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5)
        return out.returncode == 0
    except Exception:
        return False


def _port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def _chat(url: str, prompt: str) -> tuple[bool, str]:
    body = json.dumps(
        {
            "model": os.environ.get("CERTAIN_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"),
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 64,
            "temperature": 0,
        }
    ).encode()
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = json.loads(resp.read().decode())
        text = payload["choices"][0]["message"]["content"]
        return True, text
    except Exception as exc:
        return False, str(exc)


def reward_loop_smoke() -> dict:
    gold = "127.44"
    group = [
        wrap_completion(
            "revenue = 80 * 1.5\ndiscount = revenue * 0.1\nnet = revenue - discount\n"
            "tax = net * 0.18\ntotal = net + tax\n",
            "total",
            boxed=gold,
        ),
        "\\boxed{127.44}",
        wrap_completion("total = assert True\n", "total", boxed=gold),
        "I guess 0\n\\boxed{0}",
    ]
    cert = [certificate_reward(t, reference=gold) for t in group]
    outc = [outcome_reward(t, reference=gold) for t in group]
    return {
        "certificate_rewards": cert,
        "outcome_rewards": outc,
        "certificate_advantages": group_advantages(cert),
        "outcome_advantages": group_advantages(outc),
        "note": "Outcome pays the unfaithful boxed answer; certificate does not.",
    }


def try_sglang() -> dict:
    report: dict = {"nvidia": _has_nvidia(), "import": False, "served": False, "reply": None}
    try:
        import sglang  # noqa: F401

        report["import"] = True
        report["version"] = getattr(sglang, "__version__", "unknown")
    except Exception as exc:
        report["import_error"] = str(exc)

    url = os.environ.get("SGLANG_URL", "http://127.0.0.1:30000")
    report["url"] = url
    if _port_open("127.0.0.1", 30000):
        ok, text = _chat(url, "What is 2+2? Reply with one number.")
        report["served"] = ok
        report["reply"] = text[:400]
    else:
        report["served"] = False
        report["reply"] = "nothing listening on :30000"
    return report


def try_aligntune() -> dict:
    report: dict = {"import": False, "grpo_step": False}
    try:
        from aligntune.core.backend_factory import create_rl_trainer

        report["import"] = True
    except Exception as exc:
        report["import_error"] = str(exc)
        return report

    if not _has_nvidia() and os.environ.get("CERTAIN_FORCE_GRPO") != "1":
        report["skip"] = "no NVIDIA GPU; set CERTAIN_FORCE_GRPO=1 to try anyway"
        return report

    try:
        trainer = create_rl_trainer(
            model_name=os.environ.get(
                "CERTAIN_TRAIN_MODEL", "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"
            ),
            dataset_name="openai/gsm8k",
            algorithm="grpo",
            backend=os.environ.get("CERTAIN_BACKEND", "unsloth"),
            loss_type="dr_grpo",
            max_steps=1,
            batch_size=2,
            num_generations=2,
            learning_rate=1e-6,
            max_samples=8,
            reward_functions=["math"],
            output_dir=str(ROOT / "outputs" / "kill_test_grpo"),
        )
        trainer.train()
        report["grpo_step"] = True
    except Exception as exc:
        report["grpo_error"] = str(exc)
    return report


def main() -> int:
    report = {
        "cwd": str(ROOT),
        "python": sys.version,
        "nvidia": _has_nvidia(),
        "reward_loop": reward_loop_smoke(),
        "sglang": try_sglang(),
        "aligntune": try_aligntune(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    path = out / "kill_test.json"
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nwrote {path}")
    # CPU reward loop is the required local proof. GPU steps are best-effort.
    cert = report["reward_loop"]["certificate_rewards"]
    outc = report["reward_loop"]["outcome_rewards"]
    assert cert[0] == 1.0 and outc[0] == 1.0
    assert cert[1] == 0.0 and outc[1] == 1.0
    assert cert[2] == 0.0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
