"""ffmpeg/ffprobe 헬퍼."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic"}
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm", ".mts", ".m2ts"}


def run(cmd: list[str], quiet: bool = True) -> subprocess.CompletedProcess:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"명령 실패: {' '.join(cmd[:6])} ...\n{r.stderr[-2000:]}")
    return r


def is_image(path: str | Path) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXT


def probe(path: str | Path) -> dict:
    r = run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)])
    data = json.loads(r.stdout)
    v = next((s for s in data["streams"] if s["codec_type"] == "video"), None)
    a = next((s for s in data["streams"] if s["codec_type"] == "audio"), None)
    if v is None:
        raise RuntimeError("비디오 스트림이 없습니다.")
    num, den = (v.get("avg_frame_rate") or v.get("r_frame_rate") or "30/1").split("/")
    fps = float(num) / float(den) if float(den) else 30.0
    duration = float(data["format"].get("duration") or v.get("duration") or 0)
    if a and a.get("duration"):
        duration = min(duration, float(a["duration"]))   # 오디오가 더 짧으면 그 길이에 맞춤 (A/V 길이 불일치 방지)
    return {"width": int(v["width"]), "height": int(v["height"]), "fps": fps, "duration": duration,
            "has_audio": a is not None, "sample_rate": int(a["sample_rate"]) if a else 0}


def extract_audio(video: str | Path, wav: str | Path, sr: int = 16000) -> Path:
    wav = Path(wav)
    wav.parent.mkdir(parents=True, exist_ok=True)
    run(["ffmpeg", "-y", "-v", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", str(sr), "-c:a", "pcm_s16le", str(wav)])
    return wav


def extract_frames(video: str | Path, out_dir: str | Path, interval: float, width: int) -> list[tuple[float, Path]]:
    """interval 초 간격으로 프레임 추출 → [(time, path)]"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in out_dir.glob("f_*.jpg"):
        p.unlink()
    run(["ffmpeg", "-y", "-v", "error", "-i", str(video),
         "-vf", f"fps=1/{interval},scale={width}:-2", "-q:v", "3", str(out_dir / "f_%06d.jpg")])
    frames = sorted(out_dir.glob("f_*.jpg"))
    # fps=1/interval 필터는 각 구간의 첫 프레임을 뽑음 → 시각 = idx*interval
    return [(i * interval, p) for i, p in enumerate(frames)]


def fmt_time(t: float) -> str:
    m, s = divmod(max(0.0, t), 60)
    h, m = divmod(int(m), 60)
    return f"{h:02d}:{m:02d}:{s:05.2f}"
