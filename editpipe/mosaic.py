"""모자이크 적용: ffmpeg rawvideo 파이프 ↔ numpy 픽셀화."""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from .media import probe, run


def pixelate(frame: np.ndarray, box: list[float], block: int) -> None:
    h, w = frame.shape[:2]
    x0, y0 = int(box[0] * w), int(box[1] * h)
    x1, y1 = int(np.ceil(box[2] * w)), int(np.ceil(box[3] * h))
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return
    region = frame[y0:y1, x0:x1]
    rh, rw = region.shape[:2]
    block = max(block, rh // 5, rw // 40)   # 영역이 크면 블록도 키워 판독 불가하게
    bh, bw = max(1, rh // max(1, rh // block)), max(1, rw // max(1, rw // block))
    # 블록 평균
    ph, pw = (bh - rh % bh) % bh, (bw - rw % bw) % bw
    padded = np.pad(region, ((0, ph), (0, pw), (0, 0)), mode="edge")
    H, W = padded.shape[:2]
    small = padded.reshape(H // bh, bh, W // bw, bw, 3).mean(axis=(1, 3))
    big = np.repeat(np.repeat(small, bh, axis=0), bw, axis=1)[:rh, :rw]
    frame[y0:y1, x0:x1] = big.astype(np.uint8)


def apply_mosaic_video(video: Path, tracks: list[dict], out: Path, cfg: dict) -> Path:
    if not tracks:
        # 모자이크 대상 없음 → 그대로 복사
        run(["ffmpeg", "-y", "-v", "error", "-i", str(video), "-c", "copy", str(out)])
        return out
    info = probe(video)
    w, h, fps = info["width"], info["height"], info["fps"]
    block = max(4, int(cfg["mosaic"]["block"] * h / 1080))
    dec = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(video), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                           stdout=subprocess.PIPE, bufsize=10 ** 8)
    enc_cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", f"{fps:.6f}",
               "-i", "-", "-i", str(video), "-map", "0:v", "-map", "1:a?", "-c:v", "libx264",
               "-crf", str(cfg["mosaic"]["crf"]), "-preset", cfg["mosaic"]["preset"], "-pix_fmt", "yuv420p",
               "-c:a", "copy", "-movflags", "+faststart", "-shortest", str(out)]
    enc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    nbytes = w * h * 3
    idx = 0
    tracks = sorted(tracks, key=lambda t: t["start"])
    try:
        while True:
            buf = dec.stdout.read(nbytes)
            if len(buf) < nbytes:
                break
            t = idx / fps
            active = [tr for tr in tracks if tr["start"] <= t <= tr["end"]]
            if active:
                frame = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3).copy()
                for tr in active:
                    pixelate(frame, tr["box"], block)
                enc.stdin.write(frame.tobytes())
            else:
                enc.stdin.write(buf)
            idx += 1
    finally:
        enc.stdin.close()
        dec.stdout.close()
        dec.wait()
        err = enc.stderr.read().decode(errors="ignore")
        if enc.wait() != 0:
            raise RuntimeError(f"인코딩 실패: {err[-1500:]}")
    return out


def apply_mosaic_image(image: Path, hits: list[dict], out: Path, cfg: dict) -> Path:
    im = Image.open(image).convert("RGB")
    arr = np.asarray(im).copy()
    block = max(4, int(cfg["mosaic"]["block"] * arr.shape[0] / 1080))
    for h in hits:
        pixelate(arr, h["box"], block)
    Image.fromarray(arr).save(out, quality=95)
    return out
