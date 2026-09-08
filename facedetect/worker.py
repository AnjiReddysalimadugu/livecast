#!/usr/bin/env python3
"""YuNet detect + SFace re-ID: unique faces in a configurable time window."""

from __future__ import annotations

import json
import struct
import sys
import time
import traceback
from pathlib import Path

import av
import cv2
import numpy as np

CONFIG_PATH = Path(__file__).resolve().parent / "config.json"
YUNET = Path(__file__).resolve().parent / "models" / "face_detection_yunet_2023mar.onnx"
SFACE = Path(__file__).resolve().parent / "models" / "face_recognition_sface_2021dec.onnx"

MATCH_COSINE = 0.40
MIN_DETECT = 0.50
MIN_FRAC = 0.03
MIN_HITS = 2


def read_exact(stream, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            raise EOFError("stdin closed")
        buf.extend(chunk)
    return bytes(buf)


def load_window_sec(default: int = 300) -> int:
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        sec = int(data.get("window_sec", default))
        return max(30, min(sec, 24 * 3600))
    except Exception:
        return default


class UniqueTracker:
    def __init__(self) -> None:
        self.next_id = 1
        self.gallery: dict[int, dict] = {}

    def update(self, recognizer, emb: np.ndarray, now: float) -> int:
        best_id, best = None, -1.0
        for fid, item in self.gallery.items():
            score = float(
                recognizer.match(emb, item["emb"], cv2.FaceRecognizerSF_FR_COSINE)
            )
            if score > best:
                best, best_id = score, fid
        if best_id is not None and best >= MATCH_COSINE:
            item = self.gallery[best_id]
            item["emb"] = 0.85 * item["emb"] + 0.15 * emb
            item["last_seen"] = now
            item["hits"] += 1
            return best_id

        fid = self.next_id
        self.next_id += 1
        self.gallery[fid] = {"emb": emb, "last_seen": now, "hits": 1}
        return fid

    def unique_in_window(self, now: float, window_sec: int) -> list[int]:
        cutoff = now - window_sec
        return sorted(
            fid
            for fid, item in self.gallery.items()
            if item["last_seen"] >= cutoff and item.get("hits", 0) >= MIN_HITS
        )

    def prune(self, now: float, window_sec: int) -> None:
        cutoff = now - (window_sec * 2)
        for fid in [i for i, it in self.gallery.items() if it["last_seen"] < cutoff]:
            del self.gallery[fid]


def detect_rows(detector, bgr: np.ndarray) -> list:
    h, w = bgr.shape[:2]
    detector.setInputSize((w, h))
    _, faces_mat = detector.detect(bgr)
    if faces_mat is None:
        return []
    rows = []
    for row in faces_mat:
        bw, bh = float(row[2]), float(row[3])
        score = float(row[-1])
        if score < MIN_DETECT:
            continue
        if bw < w * MIN_FRAC or bh < h * MIN_FRAC:
            continue
        rows.append(row)
    return rows


def embed_faces(recognizer, tracker, bgr, rows, scale_w, scale_h, now):
    out = []
    for row in rows:
        try:
            aligned = recognizer.alignCrop(bgr, row)
            emb = recognizer.feature(aligned)
            fid = tracker.update(recognizer, emb, now)
            out.append(
                {
                    "id": fid,
                    "x": float(row[0]) / scale_w,
                    "y": float(row[1]) / scale_h,
                    "width": float(row[2]) / scale_w,
                    "height": float(row[3]) / scale_h,
                    "score": float(row[-1]),
                }
            )
        except Exception:
            continue
    # one box per id
    by_id = {}
    for f in out:
        prev = by_id.get(f["id"])
        if prev is None or f["score"] > prev["score"]:
            by_id[f["id"]] = f
    return sorted(by_id.values(), key=lambda f: f["score"], reverse=True)[:5]


def main() -> int:
    if not YUNET.exists():
        raise FileNotFoundError(YUNET)
    if not SFACE.exists():
        raise FileNotFoundError(SFACE)

    detector = cv2.FaceDetectorYN.create(str(YUNET), "", (320, 320), MIN_DETECT, 0.35, 8)
    recognizer = cv2.FaceRecognizerSF.create(str(SFACE), "")
    tracker = UniqueTracker()

    codec = av.CodecContext.create("libvpx", "r")
    last_seen_any = 0.0
    last_live: list[dict] = []
    hold_seconds = 3.0
    last_line = ""
    last_config_check = 0.0
    window_sec = load_window_sec()

    sys.stderr.write(f"face-worker: ready (YuNet+SFace unique, window={window_sec}s)\n")
    sys.stderr.flush()

    stdin = sys.stdin.buffer
    stdout = sys.stdout

    while True:
        try:
            header = read_exact(stdin, 4)
            (length,) = struct.unpack(">I", header)
            if length == 0 or length > 5_000_000:
                continue
            payload = read_exact(stdin, length)
        except EOFError:
            break
        except Exception as exc:
            sys.stderr.write(f"face-worker read error: {exc}\n")
            sys.stderr.flush()
            break

        try:
            frames = codec.decode(av.Packet(payload))
        except Exception as exc:
            sys.stderr.write(f"face-worker decode skip: {exc}\n")
            sys.stderr.flush()
            continue

        now = time.time()
        if now - last_config_check > 2.0:
            window_sec = load_window_sec(window_sec)
            last_config_check = now
            tracker.prune(now, window_sec)

        for frame in frames:
            bgr = frame.to_ndarray(format="bgr24")
            h, w = bgr.shape[:2]

            rows = detect_rows(detector, bgr)
            face_boxes = embed_faces(recognizer, tracker, bgr, rows, w, h, now)

            if 0 < len(face_boxes) < 3:
                up = cv2.resize(bgr, (w * 2, h * 2), interpolation=cv2.INTER_LINEAR)
                up_rows = detect_rows(detector, up)
                up_boxes = embed_faces(
                    recognizer, tracker, up, up_rows, w * 2, h * 2, now
                )
                # merge by id
                by_id = {f["id"]: f for f in face_boxes}
                for f in up_boxes:
                    prev = by_id.get(f["id"])
                    if prev is None or f["score"] > prev["score"]:
                        by_id[f["id"]] = f
                face_boxes = sorted(
                    by_id.values(), key=lambda f: f["score"], reverse=True
                )[:5]

            if face_boxes:
                last_seen_any = now
                last_live = face_boxes

            present = last_seen_any > 0 and (now - last_seen_any) <= hold_seconds
            use = face_boxes if face_boxes else (last_live if present else [])
            unique_ids = tracker.unique_in_window(now, window_sec)
            best = use[0]["score"] if use else 0.0

            out = {
                "w": w,
                "h": h,
                "faces": use,
                "count": len(use) if present else 0,
                "present": bool(present and use),
                "confidence": round(best, 3) if present and use else 0.0,
                "unique_count": len(unique_ids),
                "unique_ids": unique_ids,
                "window_sec": window_sec,
                "status": (
                    f"{len(use)} live · {len(unique_ids)} unique / {window_sec}s"
                    if present and use
                    else f"No face · {len(unique_ids)} unique / {window_sec}s"
                ),
            }
            line = json.dumps(out)
            if line != last_line:
                stdout.write(line + "\n")
                stdout.flush()
                last_line = line

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise SystemExit(1)
