"""PNG 오버레이 합성기: ffmpeg rawvideo 파이프로 프레임을 읽어 numpy/PIL 로 알파 합성 후 재인코딩.

ffmpeg 필터 그래프 대신 쓰는 이유: 이 빌드에 libass(자막)·drawtext 가 없고, 오버레이 수백 개도 한 번의 인코딩으로 처리하기 위함.
items: [{"png": 경로, "start": 초, "end": 초, "x": px, "y": px, "w": px, "h": px, "fade": 초}]
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from .media import probe, run


class _Layer:
    __slots__ = ("start", "end", "x", "y", "rgb", "alpha", "fade")

    def __init__(self, it: dict):
        im = Image.open(it["png"]).convert("RGBA")
        w, h = int(it["w"]), int(it["h"])
        if im.size != (w, h):
            im = im.resize((w, h), Image.LANCZOS)
        arr = np.asarray(im, dtype=np.float32)
        self.rgb = arr[:, :, :3]
        self.alpha = arr[:, :, 3:4] / 255.0
        self.x, self.y = int(it["x"]), int(it["y"])
        self.start, self.end = float(it["start"]), float(it["end"])
        self.fade = float(it.get("fade", 0.0))

    def opacity(self, t: float) -> float:
        if self.fade <= 0:
            return 1.0
        return max(0.0, min(1.0, (t - self.start) / self.fade, (self.end - t) / self.fade))


def blend(frame: np.ndarray, layer: _Layer, t: float) -> None:
    """frame(H,W,3 uint8) 위에 layer 를 제자리 합성."""
    H, W = frame.shape[:2]
    lh, lw = layer.rgb.shape[:2]
    x0, y0 = max(0, layer.x), max(0, layer.y)
    x1, y1 = min(W, layer.x + lw), min(H, layer.y + lh)
    if x1 <= x0 or y1 <= y0:
        return
    sx, sy = x0 - layer.x, y0 - layer.y
    a = layer.alpha[sy:sy + (y1 - y0), sx:sx + (x1 - x0)] * layer.opacity(t)
    rgb = layer.rgb[sy:sy + (y1 - y0), sx:sx + (x1 - x0)]
    dst = frame[y0:y1, x0:x1].astype(np.float32)
    frame[y0:y1, x0:x1] = (dst * (1 - a) + rgb * a).astype(np.uint8)


def compose(video: Path, items: list[dict], out: Path, cfg: dict) -> Path:
    items = [it for it in items if Path(it["png"]).exists() and float(it["end"]) > float(it["start"])]
    if not items:
        run(["ffmpeg", "-y", "-v", "error", "-i", str(video), "-c", "copy", str(out)])
        return out
    info = probe(video)
    w, h, fps = info["width"], info["height"], info["fps"]
    layers = sorted((_Layer(it) for it in items), key=lambda l: l.start)
    m = cfg["mosaic"]
    dec = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(video), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                           stdout=subprocess.PIPE, bufsize=10 ** 8)
    enc = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                            "-r", f"{fps:.6f}", "-i", "-", "-i", str(video), "-map", "0:v", "-map", "1:a?",
                            "-c:v", "libx264", "-crf", str(m["crf"]), "-preset", m["preset"], "-pix_fmt", "yuv420p",
                            "-c:a", "copy", "-movflags", "+faststart", "-shortest", str(out)],
                           stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    nbytes, idx = w * h * 3, 0
    try:
        while True:
            buf = dec.stdout.read(nbytes)
            if len(buf) < nbytes:
                break
            t = idx / fps
            active = [l for l in layers if l.start <= t < l.end]
            if active:
                frame = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3).copy()
                for l in active:
                    blend(frame, l, t)
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
