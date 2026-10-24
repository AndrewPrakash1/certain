#!/usr/bin/env python3
"""Local CERTAIN demo: stream a certificate, paint each line green or red."""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
DEMO = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from certain.dsl import format_decimal, interpret_text
from certain.metrics import score_completion
from certain.prompts import SYSTEM_PROMPT

CASES = json.loads((DEMO / "cases.json").read_text())
MIME = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
}


def _scoreboard() -> dict:
    for path in (DEMO / "scoreboard.json", ROOT / "outputs" / "eval_results.json"):
        if path.exists():
            return json.loads(path.read_text())
    return {"n": 0, "metrics": {}, "mode": "missing"}


def _line_payload(text: str, gold: str) -> dict:
    result = interpret_text(text)
    score = score_completion(text, gold)
    lines = []
    for line in result.lines:
        lines.append(
            {
                "line_no": line.line_no,
                "raw": line.raw,
                "kind": line.kind,
                "name": line.name,
                "value": format_decimal(line.value) if line.value is not None else None,
                "ok": line.ok,
                "error": line.error,
            }
        )
    books_error = None
    if result.books_present and not result.books_balanced:
        books_error = (
            f"debits {format_decimal(result.debits)} ≠ credits {format_decimal(result.credits)}"
        )
    return {
        "parsed": result.parsed,
        "parse_error": result.parse_error,
        "undefined": result.undefined,
        "illegal": result.illegal,
        "books_present": result.books_present,
        "books_balanced": result.books_balanced,
        "books_error": books_error,
        "debits": format_decimal(result.debits),
        "credits": format_decimal(result.credits),
        "answer_name": result.answer_name,
        "value": format_decimal(result.value) if result.value is not None else None,
        "boxed_value": format_decimal(result.boxed_value) if result.boxed_value is not None else None,
        "boxed_disagrees": result.boxed_disagrees,
        "env": {k: format_decimal(v) for k, v in result.env.items()},
        "lines": lines,
        "metrics": {
            "accuracy": score.accuracy,
            "faithful": score.faithful,
            "unfaithful_correct": score.unfaithful_correct,
            "certificate_reward": score.certificate_reward,
            "outcome_reward": score.outcome_reward,
        },
    }


def _sglang_solve(problem: str) -> str | None:
    url = os.environ.get("SGLANG_URL")
    if not url:
        return None
    model = os.environ.get("CERTAIN_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    body = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Problem:\n{problem}"},
            ],
            "max_tokens": 256,
            "temperature": 0,
        }
    ).encode()
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            payload = json.loads(resp.read().decode())
        return payload["choices"][0]["message"]["content"]
    except Exception:
        return None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, payload: dict) -> None:
        self._send(code, json.dumps(payload).encode(), "application/json; charset=utf-8")

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/cases":
            self._json(200, CASES)
            return
        if path == "/api/scoreboard":
            self._json(200, _scoreboard())
            return
        if path == "/api/health":
            self._json(200, {"ok": True, "sglang": bool(os.environ.get("SGLANG_URL"))})
            return
        rel = "index.html" if path in ("/", "/index.html") else path.lstrip("/")
        file_path = DEMO / rel
        if not file_path.resolve().is_relative_to(DEMO.resolve()) or not file_path.is_file():
            self._send(404, b"not found", "text/plain")
            return
        self._send(200, file_path.read_bytes(), MIME.get(file_path.suffix, "application/octet-stream"))

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode() or "{}")
        except json.JSONDecodeError:
            self._json(400, {"error": "bad json"})
            return
        path = urlparse(self.path).path
        if path == "/api/check":
            text = payload.get("text") or ""
            gold = str(payload.get("gold") or "")
            if not gold:
                self._json(400, {"error": "gold required"})
                return
            self._json(200, {"text": text, **_line_payload(text, gold)})
            return
        if path == "/api/solve":
            case_id = payload.get("case_id")
            model = payload.get("model") or "certain"
            live = bool(payload.get("live"))
            case = next((c for c in CASES["cases"] if c["id"] == case_id), None)
            if case is None:
                self._json(404, {"error": "unknown case"})
                return
            text = None
            source = "canned"
            if live:
                text = _sglang_solve(case["problem"])
                if text:
                    source = "sglang"
            if text is None:
                text = case["models"].get(model) or case["models"]["certain"]
            self._json(
                200,
                {
                    "case_id": case_id,
                    "model": model,
                    "source": source,
                    "text": text,
                    "gold": case["gold"],
                    **_line_payload(text, case["gold"]),
                },
            )
            return
        self._json(404, {"error": "not found"})


def main() -> int:
    host = os.environ.get("CERTAIN_DEMO_HOST", "127.0.0.1")
    port = int(os.environ.get("CERTAIN_DEMO_PORT", "8765"))
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"CERTAIN demo http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
