#!/usr/bin/env python3
"""Decode streamed WebRTC Opus → continuous Whisper captions."""

from __future__ import annotations

import json
import os
import struct
import sys
import threading
import time
import traceback
from collections import deque

import av
import numpy as np

MODEL_NAME = os.environ.get("WHISPER_MODEL", "base")
CHUNK_SEC = float(os.environ.get("WHISPER_CHUNK_SEC", "1.5"))
STEP_SEC = float(os.environ.get("WHISPER_STEP_SEC", "0.75"))
MIN_INTERVAL = float(os.environ.get("WHISPER_MIN_INTERVAL", "0.12"))
MIN_RMS = float(os.environ.get("WHISPER_MIN_RMS", "0.0035"))
TARGET_RATE = 16000
MAX_BUF_SEC = float(os.environ.get("WHISPER_MAX_BUF_SEC", "4.5"))
MAX_LAG_SEC = float(os.environ.get("WHISPER_MAX_LAG_SEC", "2.2"))
# Force language for speed+clarity (set empty for auto-detect).
LANGUAGE = os.environ.get("WHISPER_LANGUAGE", "en").strip() or None
INITIAL_PROMPT = os.environ.get(
    "WHISPER_PROMPT",
    "Live meeting captions with names and clear punctuation.",
)


def read_exact(stream, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            raise EOFError("stdin closed")
        buf.extend(chunk)
    return bytes(buf)


def emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def normalize(s: str) -> str:
    return " ".join(s.strip().split())


def merge_transcript(prev: str, new: str) -> str:
    prev_n = normalize(prev)
    new_n = normalize(new)
    if not new_n:
        return prev_n
    if not prev_n:
        return collapse_repeats(new_n)
    new_n = collapse_repeats(new_n)
    if new_n == prev_n or new_n in prev_n:
        return prev_n
    if prev_n in new_n:
        return new_n
    prev_words = prev_n.split()
    new_words = new_n.split()
    best = 0
    max_k = min(len(prev_words), len(new_words), 12)
    for k in range(max_k, 0, -1):
        if prev_words[-k:] == new_words[:k]:
            best = k
            break
    if best:
        return collapse_repeats(" ".join(prev_words + new_words[best:]))
    # Avoid appending near-duplicates of the last phrase.
    tail = " ".join(prev_words[-8:]).lower()
    if new_n.lower() in tail or tail in new_n.lower():
        return prev_n
    return collapse_repeats((prev_n + " " + new_n).strip())


def collapse_repeats(s: str) -> str:
    words = s.split()
    if not words:
        return s
    out = [words[0]]
    for w in words[1:]:
        if w.lower() == out[-1].lower():
            continue
        # drop immediate "this is this is"
        if len(out) >= 2 and w.lower() == out[-2].lower() and out[-1].lower() == out[-2].lower():
            continue
        out.append(w)
    # collapse repeated bigrams: "this is this is"
    cleaned: list[str] = []
    i = 0
    while i < len(out):
        if i + 3 < len(out) and out[i : i + 2] == out[i + 2 : i + 4]:
            cleaned.extend(out[i : i + 2])
            i += 4
            continue
        cleaned.append(out[i])
        i += 1
    return " ".join(cleaned)


def opus_head(channels: int, sample_rate: int = 48000) -> bytes:
    # RFC 7845 OpusHead (19 bytes)
    return (
        b"OpusHead"
        + bytes([1, channels & 0xFF])
        + struct.pack("<H", 0)  # pre-skip
        + struct.pack("<I", sample_rate)
        + struct.pack("<h", 0)  # gain
        + bytes([0])  # channel mapping family
    )


def make_opus_decoder(channels: int):
    codec = av.CodecContext.create("libopus", "r")
    codec.sample_rate = 48000
    codec.extradata = opus_head(channels)
    try:
        codec.layout = "stereo" if channels == 2 else "mono"
    except Exception:
        pass
    try:
        codec.format = "s16"
    except Exception:
        pass
    return codec


def packet_to_pcm(decoders: list, resampler, payload: bytes) -> np.ndarray | None:
    last_err = None
    for dec in decoders:
        try:
            packet = av.Packet(payload)
            frames = dec.decode(packet)
        except Exception as exc:
            last_err = exc
            continue
        pcm_parts = []
        for frame in frames:
            try:
                outs = resampler.resample(frame)
            except Exception:
                # Fallback: convert manually.
                try:
                    arr = frame.to_ndarray()
                    if arr.ndim > 1:
                        arr = arr.mean(axis=0)
                    pcm_parts.append(arr.astype(np.float32) / 32768.0)
                except Exception as exc:
                    last_err = exc
                continue
            for of in outs:
                arr = of.to_ndarray()
                if arr.ndim > 1:
                    arr = arr.reshape(-1)
                pcm_parts.append(arr.astype(np.float32) / 32768.0)
        if pcm_parts:
            return np.concatenate(pcm_parts)
    if last_err is not None:
        return None
    return None


def main() -> int:
    from faster_whisper import WhisperModel

    sys.stderr.write(
        f"whisper-worker: loading model={MODEL_NAME} chunk={CHUNK_SEC}s…\n"
    )
    sys.stderr.flush()
    model = WhisperModel(MODEL_NAME, device="cpu", compute_type="int8")
    sys.stderr.write("whisper-worker: ready (continuous captions)\n")
    sys.stderr.flush()
    emit({"text": "", "partial": False, "status": "ready", "model": MODEL_NAME})

    # Prefer stereo (Chrome default), keep mono as fallback decoder.
    decoders = [make_opus_decoder(2), make_opus_decoder(1)]
    resampler = av.AudioResampler(format="s16", layout="mono", rate=TARGET_RATE)

    pcm_lock = threading.Lock()
    pcm_buf: deque[np.ndarray] = deque()
    pcm_samples = 0
    cursor = 0
    full_text = ""
    last_emitted = ""
    stop = threading.Event()

    stats = {
        "pkt": 0,
        "ok": 0,
        "fail": 0,
        "last_rms": 0.0,
    }

    def append_pcm(pcm: np.ndarray) -> None:
        nonlocal pcm_samples, cursor
        with pcm_lock:
            pcm_buf.append(pcm)
            pcm_samples += len(pcm)
            max_keep = int(MAX_BUF_SEC * TARGET_RATE)
            while pcm_samples > max_keep and pcm_buf:
                dropped = pcm_buf.popleft()
                n = len(dropped)
                pcm_samples -= n
                cursor = max(0, cursor - n)

    def take_window() -> np.ndarray | None:
        nonlocal cursor, pcm_samples
        need = int(CHUNK_SEC * TARGET_RATE)
        step = int(STEP_SEC * TARGET_RATE)
        with pcm_lock:
            if pcm_samples - cursor < need:
                return None
            parts = list(pcm_buf)
            audio = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
            if cursor > len(audio):
                cursor = max(0, len(audio) - need)
            end = cursor + need
            if end > len(audio):
                return None
            window = audio[cursor:end].copy()
            cursor += step
            drop = cursor
            while drop > 0 and pcm_buf:
                first = pcm_buf[0]
                if len(first) <= drop:
                    pcm_buf.popleft()
                    pcm_samples -= len(first)
                    cursor -= len(first)
                    drop -= len(first)
                else:
                    pcm_buf[0] = first[drop:]
                    pcm_samples -= drop
                    cursor = 0
                    drop = 0
            return window

    def catch_up_if_behind() -> float:
        """If inference lagged, jump near live edge so captions stay ~2s late, not 20s."""
        nonlocal cursor
        with pcm_lock:
            lag = (pcm_samples - cursor) / TARGET_RATE
            if lag > MAX_LAG_SEC:
                keep = int(CHUNK_SEC * TARGET_RATE)
                cursor = max(0, pcm_samples - keep)
                return lag
        return 0.0

    def lag_sec() -> float:
        with pcm_lock:
            return max(0.0, (pcm_samples - cursor) / TARGET_RATE)

    def infer_loop() -> None:
        nonlocal full_text, last_emitted
        last_status = 0.0
        while not stop.is_set():
            now = time.time()
            if now - last_status > 1.2:
                with pcm_lock:
                    buffered = pcm_samples / TARGET_RATE
                lag = lag_sec()
                emit(
                    {
                        "text": last_emitted,
                        "partial": True,
                        "status": "listening" if stats["ok"] else "waiting_audio",
                        "pkt": stats["pkt"],
                        "decoded": stats["ok"],
                        "decode_fail": stats["fail"],
                        "buffered_sec": round(lag, 2),
                        "queued_sec": round(buffered, 2),
                        "rms": round(stats["last_rms"], 4),
                    }
                )
                sys.stderr.write(
                    f"whisper-worker: pkt={stats['pkt']} ok={stats['ok']} "
                    f"fail={stats['fail']} lag={lag:.2f}s rms={stats['last_rms']:.4f}\n"
                )
                sys.stderr.flush()
                last_status = now

            skipped = catch_up_if_behind()
            if skipped:
                sys.stderr.write(f"whisper-worker: catch-up skipped {skipped:.1f}s lag\n")
                sys.stderr.flush()

            window = take_window()
            if window is None:
                time.sleep(0.03)
                continue

            rms = float(np.sqrt(np.mean(window * window))) if len(window) else 0.0
            stats["last_rms"] = rms
            if rms < MIN_RMS:
                continue

            t0 = time.time()
            try:
                segments, info = model.transcribe(
                    window,
                    language=LANGUAGE,
                    beam_size=1,
                    best_of=1,
                    temperature=0.0,
                    vad_filter=False,
                    condition_on_previous_text=False,
                    without_timestamps=True,
                    initial_prompt=INITIAL_PROMPT,
                )
                chunk_text = " ".join(s.text.strip() for s in segments).strip()
                lang = getattr(info, "language", None) or (LANGUAGE or "")
            except Exception as exc:
                sys.stderr.write(f"whisper-worker infer error: {exc}\n")
                sys.stderr.flush()
                traceback.print_exc(file=sys.stderr)
                time.sleep(MIN_INTERVAL)
                continue

            elapsed = time.time() - t0
            if chunk_text:
                full_text = merge_transcript(full_text, chunk_text)
                show = full_text if len(full_text) <= 500 else ("…" + full_text[-500:])
                if show != last_emitted:
                    last_emitted = show
                    emit(
                        {
                            "text": show,
                            "partial": True,
                            "status": "ok",
                            "lang": lang,
                            "rms": round(rms, 4),
                            "infer_ms": int(elapsed * 1000),
                            "buffered_sec": round(lag_sec(), 2),
                        }
                    )
            # Prefer staying near live over waiting out a long min-interval.
            time.sleep(0.02 if elapsed > STEP_SEC else max(0.02, MIN_INTERVAL - elapsed))

    worker = threading.Thread(target=infer_loop, name="whisper-infer", daemon=True)
    worker.start()

    stdin = sys.stdin.buffer
    try:
        while True:
            try:
                header = read_exact(stdin, 4)
                (length,) = struct.unpack(">I", header)
                if length == 0 or length > 1_000_000:
                    continue
                payload = read_exact(stdin, length)
            except EOFError:
                break
            except Exception as exc:
                sys.stderr.write(f"whisper-worker read error: {exc}\n")
                sys.stderr.flush()
                break

            stats["pkt"] += 1
            pcm = packet_to_pcm(decoders, resampler, payload)
            if pcm is None or len(pcm) == 0:
                stats["fail"] += 1
                continue
            stats["ok"] += 1
            append_pcm(pcm)
    finally:
        stop.set()

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
