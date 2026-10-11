"""컷 편집 1차: 무음 구간 + 필러워드 제거."""
from __future__ import annotations

import csv
import json
import re
import subprocess
from pathlib import Path

from .media import filter_script_args, fmt_time, probe, run


# ---------- 무음 검출 (ffmpeg silencedetect) ----------
def detect_silence(video: Path, db: float, min_len: float) -> list[tuple[float, float]]:
    cmd = ["ffmpeg", "-hide_banner", "-nostats", "-i", str(video), "-af",
           f"silencedetect=noise={db}dB:d={min_len}", "-f", "null", "-"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    starts = [float(m) for m in re.findall(r"silence_start:\s*([\d.]+)", r.stderr)]
    ends = [float(m) for m in re.findall(r"silence_end:\s*([\d.]+)", r.stderr)]
    out = []
    for i, s in enumerate(starts):
        e = ends[i] if i < len(ends) else None
        out.append((s, e))
    return out


def _norm(w: str) -> str:
    return re.sub(r"[^\w가-힣]", "", w).lower()


def find_fillers(words: list[dict], fillers: list[str], max_len: float, pad: float) -> list[dict]:
    fset = {_norm(f) for f in fillers}
    out = []
    for w in words:
        n = _norm(w["word"])
        if n in fset and (w["end"] - w["start"]) <= max_len:
            out.append({"start": max(0.0, w["start"] - pad), "end": w["end"] + pad, "reason": f"필러 '{w['word']}'"})
    return out


def _merge(ranges: list[tuple[float, float]]) -> list[tuple[float, float]]:
    ranges = sorted(r for r in ranges if r[1] > r[0])
    out: list[list[float]] = []
    for s, e in ranges:
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def build_cut_list(duration: float, silences: list[tuple[float, float | None]], fillers: list[dict], c: dict) -> dict:
    keep_pad = float(c["keep_silence"])
    removed: list[tuple[float, float, str]] = []
    for s, e in silences:
        e = duration if e is None else e
        s2 = 0.0 if s <= 0.01 else s + keep_pad            # 맨 앞 무음은 통째로 제거
        e2 = duration if e >= duration - 0.01 else e - keep_pad  # 맨 뒤 무음도 통째로 제거
        if e2 - s2 > 0.05:
            removed.append((s2, e2, "무음"))
    for f in fillers:
        removed.append((f["start"], f["end"], f["reason"]))
    merged = _merge([(s, e) for s, e, _ in removed])
    # 유지 구간 계산
    keep, cur = [], 0.0
    for s, e in merged:
        if s - cur >= float(c["min_segment"]):
            keep.append({"start": round(cur, 3), "end": round(s, 3)})
        cur = max(cur, e)
    if duration - cur >= float(c["min_segment"]):
        keep.append({"start": round(cur, 3), "end": round(duration, 3)})
    kept = sum(k["end"] - k["start"] for k in keep)
    return {
        "source_duration": round(duration, 3), "result_duration": round(kept, 3),
        "removed_seconds": round(duration - kept, 3),
        "keep": keep,
        "removed": [{"start": round(s, 3), "end": round(e, 3), "reason": r} for s, e, r in sorted(removed)],
    }


def plan_cuts(video: Path, transcript: dict, work: Path, cfg: dict, force: bool = False) -> dict:
    out = work / "cuts.json"
    if out.exists() and not force:
        return json.loads(out.read_text(encoding="utf-8"))
    c = cfg["cut"]
    info = probe(video)
    words = transcript.get("words") or []
    note = ""
    if words:
        silences = detect_silence(video, float(c["silence_db"]), float(c["min_silence"]))
    else:
        # 말소리가 전혀 없는 영상(제품 촬영·BGM 등)은 '무음' 기준이 의미 없으므로 컷하지 않고 전체를 유지
        silences = []
        note = "음성(전사 단어)이 없어 무음 컷을 생략했습니다. 그래도 자르려면 cuts.json 을 직접 편집하세요."
        print(f"  [컷 편집] {note}")
    fillers = find_fillers(words, c["fillers"], float(c["filler_max_len"]), float(c["filler_pad"]))
    plan = build_cut_list(info["duration"], silences, fillers, c)
    if note:
        plan["note"] = note
    out.write_text(json.dumps(plan, ensure_ascii=False, indent=1), encoding="utf-8")
    with open(work / "cuts_review.csv", "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f)
        wr.writerow(["시작", "끝", "길이(초)", "사유"])
        for r in plan["removed"]:
            wr.writerow([fmt_time(r["start"]), fmt_time(r["end"]), round(r["end"] - r["start"], 2), r["reason"]])
    return plan


def apply_cuts(video: Path, plan: dict, out: Path, cfg: dict) -> Path:
    """keep 구간만 이어붙여 새 영상 생성 (trim/atrim + concat, 샘플 정확)."""
    keep = plan["keep"]
    if not keep:
        raise RuntimeError("남길 구간이 없습니다.")
    info = probe(video)
    has_a = info["has_audio"]
    parts, labels = [], ""
    for i, k in enumerate(keep):
        parts.append(f"[0:v]trim=start={k['start']}:end={k['end']},setpts=PTS-STARTPTS[v{i}]")
        labels += f"[v{i}]"
        if has_a:
            parts.append(f"[0:a]atrim=start={k['start']}:end={k['end']},asetpts=PTS-STARTPTS[a{i}]")
            labels += f"[a{i}]"
    n = len(keep)
    parts.append(f"{labels}concat=n={n}:v=1:a={1 if has_a else 0}[vout]" + ("[aout]" if has_a else ""))
    script = out.with_suffix(".filter.txt")
    script.write_text(";\n".join(parts), encoding="utf-8")
    m = cfg["mosaic"]
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(video), *filter_script_args(script), "-map", "[vout]"]
    if has_a:
        cmd += ["-map", "[aout]", "-c:a", "aac", "-b:a", "192k"]
    cmd += ["-c:v", "libx264", "-crf", str(m.get("crf", 18)), "-preset", m.get("preset", "medium"),
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
    run(cmd)
    script.unlink(missing_ok=True)
    return out
