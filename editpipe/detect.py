"""개인정보/상표 검출: OCR(macOS Vision, 그 외 OS 는 tesseract 폴백) + 얼굴 검출 + 정규식 + (선택) Claude 판별/로고 검출."""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from . import llm_client as llm
from .media import extract_frames, probe

Box = list[float]  # [x0, y0, x1, y1] normalized 0~1, origin top-left


# ---------- OCR 백엔드 ----------
def _has_vision() -> bool:
    """macOS 의 Apple Vision 사용 가능 여부 (Linux/Windows 에서는 False → tesseract 폴백)."""
    try:
        import Vision, Quartz  # noqa: F401
        return True
    except ImportError:
        return False


def _ci_image(path: Path):
    import Quartz
    url = Quartz.CFURLCreateWithFileSystemPath(None, str(path), Quartz.kCFURLPOSIXPathStyle, False)
    return Quartz.CIImage.imageWithContentsOfURL_(url)


def _ocr_vision(path: Path) -> list[dict]:
    import Vision
    req = Vision.VNRecognizeTextRequest.alloc().init()
    req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    req.setUsesLanguageCorrection_(False)
    try:
        req.setRecognitionLanguages_(["ko-KR", "en-US"])
    except Exception:
        pass
    handler = Vision.VNImageRequestHandler.alloc().initWithCIImage_options_(_ci_image(path), None)
    ok, err = handler.performRequests_error_([req], None)
    out = []
    for obs in (req.results() or []):
        cand = obs.topCandidates_(1)
        if not cand:
            continue
        bb = obs.boundingBox()
        x, y, w, h = bb.origin.x, bb.origin.y, bb.size.width, bb.size.height
        out.append({"text": str(cand[0].string()), "conf": float(cand[0].confidence()),
                    "box": [x, 1 - y - h, x + w, 1 - y]})
    return out


def _ocr_tesseract(path: Path, langs: str = "kor+eng", min_conf: float = 40.0) -> list[dict]:
    """macOS 가 아닐 때의 폴백: tesseract(pytesseract). 단어를 줄(line) 단위로 묶어 Vision 과 같은 형식으로 반환."""
    import pytesseract
    img = Image.open(path).convert("RGB")
    W, H = img.size
    data = pytesseract.image_to_data(img, lang=langs, config="--psm 11", output_type=pytesseract.Output.DICT)
    lines: dict[tuple, dict] = {}
    for i, txt in enumerate(data["text"]):
        txt = (txt or "").strip()
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = -1.0
        if not txt or conf < min_conf:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        x0, y0 = data["left"][i], data["top"][i]
        x1, y1 = x0 + data["width"][i], y0 + data["height"][i]
        ln = lines.setdefault(key, {"words": [], "confs": [], "box": [x0, y0, x1, y1]})
        ln["words"].append(txt)
        ln["confs"].append(conf)
        b = ln["box"]
        ln["box"] = [min(b[0], x0), min(b[1], y0), max(b[2], x1), max(b[3], y1)]
    out = []
    for ln in lines.values():
        b = ln["box"]
        out.append({"text": " ".join(ln["words"]), "conf": sum(ln["confs"]) / len(ln["confs"]) / 100.0,
                    "box": [b[0] / W, b[1] / H, b[2] / W, b[3] / H]})
    return out


def ocr_image(path: Path) -> list[dict]:
    """[{text, conf, box}] (box normalized 0~1, top-left origin). macOS→Vision, 그 외→tesseract."""
    if _has_vision():
        return _ocr_vision(path)
    return _ocr_tesseract(path)


def detect_faces(path: Path) -> list[Box]:
    if not _has_vision():
        return []   # Linux 폴백 없음 (config detect.faces 는 기본 off)
    import Vision
    req = Vision.VNDetectFaceRectanglesRequest.alloc().init()
    handler = Vision.VNImageRequestHandler.alloc().initWithCIImage_options_(_ci_image(path), None)
    handler.performRequests_error_([req], None)
    out = []
    for obs in (req.results() or []):
        bb = obs.boundingBox()
        x, y, w, h = bb.origin.x, bb.origin.y, bb.size.width, bb.size.height
        out.append([x, 1 - y - h, x + w, 1 - y])
    return out


# ---------- 규칙 기반 판별 ----------
def classify_text_rules(text: str, pii_regex: dict[str, str], brand_words: list[str]) -> tuple[str, str] | None:
    for label, pat in pii_regex.items():
        if re.search(pat, text):
            return "pii", label
    low = text.lower()
    for b in brand_words:
        if re.search(rf"(?<![a-z0-9가-힣]){re.escape(b.lower())}(?![a-z0-9가-힣])", low):
            return "brand", b
    return None


CLASSIFY_PROMPT = """아래는 영상 화면에서 OCR로 읽은 텍스트 목록입니다. 각 항목이 다음 중 하나면 표시하세요.
- "pii": 개인정보 (사람 이름, 주소, 전화번호, 이메일, 생년월일, 계좌/카드/주민번호, 차량번호, 아이디/닉네임 등 특정 개인을 식별할 수 있는 정보)
- "brand": 상표·브랜드명·회사명·제품명·저작물 제목 등 지식재산권/상표권 관련 텍스트
- null: 해당 없음 (일반 단어, 자막, 숫자 등)

주의: 과도하게 잡지 말 것. 일반 명사, 흔한 자막 문장, 날짜/시간, 메뉴명은 null.
JSON 배열로만 답하세요: [{"i": 인덱스, "type": "pii"|"brand"|null, "label": "짧은 사유"}]

텍스트 목록:
{items}
"""

LOGO_PROMPT = """이 영상 프레임에서 상표권/지식재산권 보호 대상인 **그래픽 로고, 브랜드 마크, 캐릭터, 포스터/앨범 커버, 제품 패키지의 브랜드 표시**를 모두 찾으세요.
텍스트만 있는 자막은 제외합니다 (별도 처리됨). 사람 얼굴도 제외.
이미지 위에 10×10 격자와 좌표 눈금(0.0~1.0, 좌상단 원점)이 그려져 있습니다. 눈금을 기준으로 각 항목의 위치를 정규화 좌표로 주세요.
박스는 대상을 완전히 덮도록 약간 넉넉하게. 확실히 식별되는 것만, 배경에 흐릿하게 보이는 것은 제외.
JSON 배열로만 답하세요. 없으면 []:
[{"label": "브랜드/대상 이름", "box": [x0, y0, x1, y1]}]
"""


def grid_frame(frame: Path, n: int = 10) -> Path:
    """LLM 로고 검출용: 프레임 위에 n×n 격자와 좌표 눈금을 그려 저장 (좌표 정확도 향상)."""
    from PIL import ImageDraw, ImageFont
    im = Image.open(frame).convert("RGB")
    W, H = im.size
    dr = ImageDraw.Draw(im, "RGBA")
    try:
        f = ImageFont.load_default(max(12, W // 70))
    except TypeError:
        f = ImageFont.load_default()
    for i in range(1, n):
        x, y = W * i / n, H * i / n
        dr.line([(x, 0), (x, H)], fill=(255, 255, 0, 110), width=1)
        dr.line([(0, y), (W, y)], fill=(255, 255, 0, 110), width=1)
        lab = f"{i / n:.1f}"
        dr.text((x + 2, 2), lab, fill=(255, 255, 0, 255), font=f)
        dr.text((2, y + 2), lab, fill=(255, 255, 0, 255), font=f)
    out = frame.with_name(frame.stem + "_grid.jpg")
    im.save(out, quality=85)
    return out


def classify_text_claude(texts: list[str], cfg: dict) -> dict[str, tuple[str, str]]:
    if not texts:
        return {}
    items = "\n".join(f"{i}: {t}" for i, t in enumerate(texts))
    res = llm.ask_json(cfg, CLASSIFY_PROMPT.replace("{items}", items), effort="low")
    out = {}
    for r in res:
        if r.get("type") in ("pii", "brand"):
            try:
                out[texts[int(r["i"])]] = (r["type"], str(r.get("label", "")))
            except (ValueError, IndexError):
                continue
    return out


def detect_logos_claude(frame: Path, cfg: dict) -> list[dict]:
    res = llm.ask_json(cfg, LOGO_PROMPT, images=[grid_frame(frame)], effort="low")
    out = []
    for r in res or []:
        b = r.get("box")
        if isinstance(b, list) and len(b) == 4:
            x0, y0, x1, y1 = [min(1.0, max(0.0, float(v))) for v in b]
            if x1 > x0 and y1 > y0:
                out.append({"label": str(r.get("label", "logo")), "box": [x0, y0, x1, y1]})
    return out


# ---------- 장면 변화 ----------
def _thumb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L").resize((32, 18)), dtype=np.float32) / 255.0


def scene_changed(a: np.ndarray | None, b: np.ndarray, thr: float) -> bool:
    return a is None or float(np.abs(a - b).mean()) > thr * 0.5


# ---------- 트랙 병합 ----------
def _iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _expand(b: Box, m: float) -> Box:
    w, h = b[2] - b[0], b[3] - b[1]
    return [max(0.0, b[0] - w * m), max(0.0, b[1] - h * m), min(1.0, b[2] + w * m), min(1.0, b[3] + h * m)]


def build_tracks(hits: list[dict], interval: float, hold: float, margin: float) -> list[dict]:
    """hits: [{t, type, label, box, (t_end)}] → 시간 연속 트랙 [{start,end,type,label,box}]
    t_end 가 있으면(로고처럼 다음 검사 시점까지 유효한 검출) 그때까지 트랙을 유지한다."""
    tracks: list[dict] = []
    for h in sorted(hits, key=lambda x: x["t"]):
        box = _expand(h["box"], margin)
        end = max(h["t"] + interval, float(h.get("t_end", 0.0))) + hold
        best = None
        for tr in tracks:
            if tr["type"] == h["type"] and h["t"] - tr["end"] <= interval + hold + 1e-6 and _iou(tr["box"], box) > 0.3:
                best = tr
                break
        if best:
            best["end"] = max(best["end"], end)
            best["box"] = [min(best["box"][0], box[0]), min(best["box"][1], box[1]), max(best["box"][2], box[2]), max(best["box"][3], box[3])]
            best["count"] += 1
        else:
            tracks.append({"start": max(0.0, h["t"] - hold), "end": end, "type": h["type"],
                           "label": h["label"], "box": box, "count": 1})
    for t in tracks:
        t["start"], t["end"] = round(t["start"], 3), round(t["end"], 3)
        t["box"] = [round(v, 4) for v in t["box"]]
    return tracks


# ---------- 메인 ----------
def detect_video(video: Path, work: Path, cfg: dict, force: bool = False, use_claude: bool = True) -> dict:
    out = work / "detections.json"
    if out.exists() and not force:
        return json.loads(out.read_text(encoding="utf-8"))
    d = cfg["detect"]
    interval = float(d["sample_interval"])
    frames = extract_frames(video, work / "frames", interval, int(d["frame_width"]))
    claude_ok = use_claude and llm.available(cfg)
    hits: list[dict] = []
    ocr_cache: dict[str, dict] = {}       # 같은 텍스트 재판별 방지
    unknown_texts: set[str] = set()
    prev_thumb, last_logo_t = None, -1e9
    logo_frames = 0
    logo_jobs: list[tuple[float, Path]] = []   # LLM 로고 검출 대상 (병렬 처리)
    for t, fp in frames:
        for o in ocr_image(fp):
            txt = o["text"].strip()
            if len(txt) < 2:
                continue
            r = classify_text_rules(txt, d["pii_regex"], d["brand_words"])
            if r:
                hits.append({"t": t, "type": r[0], "label": f"{r[1]}: {txt}", "box": o["box"]})
            else:
                unknown_texts.add(txt)
                ocr_cache.setdefault(txt, []).append({"t": t, "box": o["box"]})
        if d.get("faces"):
            for b in detect_faces(fp):
                hits.append({"t": t, "type": "face", "label": "얼굴", "box": b})
        if claude_ok and d.get("claude_logo_detect"):
            th = _thumb(fp)
            if scene_changed(prev_thumb, th, float(d["logo_scene_threshold"])) or t - last_logo_t >= float(d["logo_max_interval"]):
                prev_thumb, last_logo_t = th, t
                logo_jobs.append((t, fp))
    workers = int(d.get("parallel", 4))
    # LLM 로고 검출 (프레임별 병렬)
    if logo_jobs:
        logo_frames = len(logo_jobs)
        video_end = frames[-1][0] + interval if frames else 0.0
        next_check = {t: (logo_jobs[i + 1][0] if i + 1 < len(logo_jobs) else video_end) for i, (t, _) in enumerate(logo_jobs)}

        def _logo(job):
            t, fp = job
            try:
                # 장면이 바뀌기 전(다음 로고 검사 시점)까지는 같은 로고가 계속 보인다고 보고 트랙을 유지
                # LLM 좌표는 근사치이고 손/카메라가 움직이므로 OCR 박스보다 여백을 더 준다 (detect.logo_box_margin)
                lm = float(d.get("logo_box_margin", 0.3))
                return [{"t": t, "t_end": next_check[t], "type": "brand", "label": f"로고: {lg['label']}", "box": _expand(lg["box"], lm)}
                        for lg in detect_logos_claude(fp, cfg)]
            except RuntimeError as e:
                print(f"  [로고검출 건너뜀 {t:.1f}s] {e}")
                return []

        with ThreadPoolExecutor(max_workers=workers) as ex:
            for res in ex.map(_logo, logo_jobs):
                hits.extend(res)
    # LLM 텍스트 판별 (이름/주소/상표명 등 정규식이 못 잡는 것) — 배치 병렬
    claude_hits = {}
    if claude_ok and d.get("claude_text_classify") and unknown_texts:
        texts = sorted(unknown_texts)
        batches = [texts[i:i + 100] for i in range(0, len(texts), 100)]

        def _cls(batch):
            try:
                return classify_text_claude(batch, cfg)
            except RuntimeError as e:
                print(f"  [텍스트 판별 건너뜀] {e}")
                return {}

        with ThreadPoolExecutor(max_workers=workers) as ex:
            for res in ex.map(_cls, batches):
                claude_hits.update(res)
        for txt, (typ, label) in claude_hits.items():
            for occ in ocr_cache.get(txt, []):
                hits.append({"t": occ["t"], "type": typ, "label": f"{label}: {txt}", "box": occ["box"]})
    tracks = build_tracks(hits, interval, float(d["hold_seconds"]), float(d["box_margin"]))
    data = {"video": str(video), "sample_interval": interval, "llm_used": llm.provider(cfg) if claude_ok else None, "logo_frames_checked": logo_frames,
            "ocr_unique_texts": len(unknown_texts) + len({h["label"] for h in hits if h["type"] != "face"}),
            "tracks": tracks}
    out.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return data


def detect_image(image: Path, cfg: dict, use_claude: bool = True) -> list[dict]:
    """정지 이미지용: [{type,label,box}]"""
    d = cfg["detect"]
    claude_ok = use_claude and llm.available(cfg)
    hits, unknown = [], []
    for o in ocr_image(image):
        txt = o["text"].strip()
        if len(txt) < 2:
            continue
        r = classify_text_rules(txt, d["pii_regex"], d["brand_words"])
        if r:
            hits.append({"type": r[0], "label": f"{r[1]}: {txt}", "box": _expand(o["box"], float(d["box_margin"]))})
        else:
            unknown.append((txt, o["box"]))
    if d.get("faces"):
        hits += [{"type": "face", "label": "얼굴", "box": _expand(b, float(d["box_margin"]))} for b in detect_faces(image)]
    if claude_ok:
        try:
            cl = classify_text_claude([t for t, _ in unknown], cfg) if d.get("claude_text_classify") else {}
            for txt, box in unknown:
                if txt in cl:
                    hits.append({"type": cl[txt][0], "label": f"{cl[txt][1]}: {txt}", "box": _expand(box, float(d["box_margin"]))})
            if d.get("claude_logo_detect"):
                hits += [{"type": "brand", "label": f"로고: {l['label']}", "box": _expand(l["box"], float(d["box_margin"]) + float(d.get("logo_box_margin", 0.3)))}
                         for l in detect_logos_claude(image, cfg)]
        except RuntimeError as e:
            print(f"  [Claude 판별 건너뜀] {e}")
    return hits
