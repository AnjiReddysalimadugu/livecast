#!/usr/bin/env python3
"""Standalone answer service: question text → LLM text answer (Ollama).

Run on any machine (multi-server):
  python answer/server.py --host 0.0.0.0 --port 8091

SFU points here with:
  set ANSWER_URL=http://<this-host>:8091/answer
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
SYSTEM = os.environ.get(
    "ANSWER_SYSTEM",
    "You are a helpful assistant for a live video call demo. "
    "Answer briefly and clearly in the same language as the question. "
    "If the question is unclear or noise, say you did not catch it.",
)


def ask_ollama(question: str) -> str:
    body = json.dumps(
        {
            "model": MODEL,
            "stream": False,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": question},
            ],
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as res:
        data = json.loads(res.read().decode("utf-8"))
    msg = data.get("message") or {}
    return (msg.get("content") or "").strip()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(200)
        self._cors()
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/", "/health"):
            payload = {
                "ok": True,
                "service": "answer",
                "model": MODEL,
                "ollama": OLLAMA_URL,
            }
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self._cors()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        if self.path.rstrip("/") != "/answer":
            self.send_error(404)
            return
        n = int(self.headers.get("Content-Length", "0"))
        try:
            body = json.loads(self.rfile.read(n).decode("utf-8") or "{}")
        except Exception:
            self.send_error(400, "invalid json")
            return
        question = (body.get("question") or body.get("text") or "").strip()
        if not question:
            raw = json.dumps({"error": "question required"}).encode("utf-8")
            self.send_response(400)
            self._cors()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(raw)
            return
        try:
            answer = ask_ollama(question)
            out = {
                "question": question,
                "answer": answer,
                "model": MODEL,
                "status": "ok",
            }
            code = 200
        except urllib.error.URLError as exc:
            out = {
                "question": question,
                "answer": "",
                "status": "error",
                "error": f"ollama unreachable: {exc}",
            }
            code = 502
        except Exception as exc:
            traceback.print_exc()
            out = {
                "question": question,
                "answer": "",
                "status": "error",
                "error": str(exc),
            }
            code = 500
        raw = json.dumps(out, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self._cors()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=os.environ.get("ANSWER_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("ANSWER_PORT", "8091")))
    args = ap.parse_args()
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    sys.stderr.write(
        f"answer-service: http://{args.host}:{args.port}/answer  model={MODEL}\n"
    )
    sys.stderr.flush()
    httpd.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
