"""faster-whisper 로 단어 단위 타임스탬프 전사."""
from __future__ import annotations

import json
from pathlib import Path

from .media import extract_audio


def transcribe(video: Path, work: Path, cfg: dict, force: bool = False) -> dict:
    out = work / "transcript.json"
    if out.exists() and not force:
        return json.loads(out.read_text(encoding="utf-8"))
    wav = extract_audio(video, work / "audio.wav")
    from faster_whisper import WhisperModel

    w = cfg["whisper"]
    model = WhisperModel(w.get("model", "small"), device=w.get("device", "cpu"), compute_type=w.get("compute_type", "int8"))
    segments, info = model.transcribe(str(wav), language=w.get("language") or None, word_timestamps=True,
                                      vad_filter=True, vad_parameters={"min_silence_duration_ms": 300})
    words, segs = [], []
    for s in segments:
        segs.append({"start": round(s.start, 3), "end": round(s.end, 3), "text": s.text.strip()})
        for wd in (s.words or []):
            words.append({"word": wd.word.strip(), "start": round(wd.start, 3), "end": round(wd.end, 3),
                          "prob": round(float(wd.probability), 3)})
    data = {"language": info.language, "duration": round(float(info.duration), 3), "segments": segs, "words": words}
    out.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return data


def remap_transcript(tr: dict, keep: list[dict]) -> dict:
    """컷 편집 후 타임라인으로 단어/문장 시각 변환 (잘린 구간의 단어는 제거)."""
    def remap(t: float) -> float | None:
        off = 0.0
        for k in keep:
            if k["start"] <= t <= k["end"]:
                return off + (t - k["start"])
            if t < k["start"]:
                return None
            off += k["end"] - k["start"]
        return None

    words = []
    for w in tr["words"]:
        s, e = remap(w["start"]), remap(w["end"])
        if s is None and e is None:
            continue
        words.append({**w, "start": round(s if s is not None else e, 3), "end": round(e if e is not None else s, 3)})
    segs = []
    for sg in tr["segments"]:
        s, e = remap(sg["start"]), remap(sg["end"])
        if s is None and e is None:
            continue
        segs.append({**sg, "start": round(s if s is not None else e, 3), "end": round(e if e is not None else s, 3)})
    total = sum(k["end"] - k["start"] for k in keep)
    return {**tr, "words": words, "segments": segs, "duration": round(total, 3)}
