"""시각 보조 요소: 자막(전사 기반), 키워드 강조(lower-third), 챕터 타이틀 — PIL 로 PNG 렌더 후 compose 로 합성."""
from __future__ import annotations

import json
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from . import llm_client as llm
from .compose import compose
from .media import fmt_time, probe

FONT_TTC = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
FONT_IDX = {"regular": 0, "medium": 2, "semibold": 4, "bold": 6, "extrabold": 14}
# macOS 가 아닐 때 쓰는 한글 폰트 후보 (apt: fonts-noto-cjk, fonts-nanum). 앞에 있는 것 우선.
FONT_FALLBACK = {
    "regular": ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/truetype/nanum/NanumSquareRoundR.ttf",
                "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"],
    "bold": ["/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", "/usr/share/fonts/truetype/nanum/NanumSquareRoundB.ttf",
             "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"],
}
_FONT_WARNED = False


def font(size: int, weight: str = "bold") -> ImageFont.FreeTypeFont:
    global _FONT_WARNED
    try:
        return ImageFont.truetype(FONT_TTC, size, index=FONT_IDX.get(weight, 6))
    except OSError:
        pass
    for path in FONT_FALLBACK["regular" if weight in ("regular", "medium") else "bold"]:
        try:
            return ImageFont.truetype(path, size, index=0)   # Noto CJK ttc 의 index 0 = JP 이지만 한글 글리프 포함
        except OSError:
            continue
    if not _FONT_WARNED:
        _FONT_WARNED = True
        print("  [visuals] 한글 폰트를 찾지 못해 기본 폰트로 렌더합니다 (한글이 □ 로 보일 수 있음). fonts-noto-cjk 또는 fonts-nanum 설치 권장.")
    return ImageFont.load_default(size)


# ───────────── 자막 ─────────────
_END = re.compile(r"[.?!。…]$")


def mask_pii(text: str, pii_regex: dict[str, str] | None) -> str:
    """자막 속 전화번호/이메일/계좌 등은 ***로 가림 (화면 모자이크와 동일 기준 + 붙여 읽은 숫자열)."""
    if not pii_regex:
        return text
    for pat in pii_regex.values():
        text = re.sub(pat, "***", text)
    text = re.sub(r"(?<!\d)\d{9,}(?!\d)", "***", text)   # "01012345678" 처럼 붙여 말한 번호
    return text


def build_subtitles(transcript: dict, max_chars: int = 28, max_gap: float = 0.8, min_dur: float = 0.8,
                    pii_regex: dict[str, str] | None = None) -> list[dict]:
    """단어 타임스탬프 → 자막 줄 [{start, end, text}]"""
    words = transcript.get("words") or []
    lines, cur, cur_start, last_end = [], [], None, None
    for w in words:
        txt = w["word"].strip()
        if not txt:
            continue
        cand = (" ".join(cur + [txt])).strip()
        split = cur and (len(cand) > max_chars or (last_end is not None and w["start"] - last_end > max_gap))
        if split:
            lines.append({"start": cur_start, "end": last_end, "text": " ".join(cur)})
            cur, cur_start = [], None
        if cur_start is None:
            cur_start = w["start"]
        cur.append(txt)
        last_end = w["end"]
        if _END.search(txt):
            lines.append({"start": cur_start, "end": last_end, "text": " ".join(cur)})
            cur, cur_start = [], None
    if cur:
        lines.append({"start": cur_start, "end": last_end, "text": " ".join(cur)})
    # 최소 노출 + 다음 줄과 겹침 방지
    for i, l in enumerate(lines):
        l["text"] = mask_pii(l["text"], pii_regex)
        l["end"] = max(l["end"] + 0.15, l["start"] + min_dur)
        if i + 1 < len(lines):
            l["end"] = min(l["end"], lines[i + 1]["start"] - 0.02)
        l["start"], l["end"] = round(l["start"], 3), round(l["end"], 3)
    return [l for l in lines if l["end"] > l["start"]]


def render_subtitle(text: str, W: int, H: int, st: dict) -> Image.Image:
    s = H / 1080
    f = font(int(st.get("size", 44) * s), "bold")
    pad_x, pad_y = int(28 * s), int(14 * s)
    tmp = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    x0, y0, x1, y1 = tmp.textbbox((0, 0), text, font=f, stroke_width=int(2 * s))
    tw, th = x1 - x0, y1 - y0
    im = Image.new("RGBA", (tw + pad_x * 2, th + pad_y * 2), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((0, 0, im.width - 1, im.height - 1), radius=int(14 * s), fill=(20, 20, 20, int(255 * st.get("box_alpha", 0.6))))
    d.text((pad_x - x0, pad_y - y0), text, font=f, fill=(255, 255, 255, 255), stroke_width=int(2 * s), stroke_fill=(0, 0, 0, 255))
    return im


# ───────────── 키워드 강조 (lower-third) ─────────────
def render_keyword(text: str, sub: str, W: int, H: int, accent=(255, 180, 162)) -> Image.Image:
    s = H / 1080
    f1, f2 = font(int(56 * s), "extrabold"), font(int(30 * s), "medium")
    tmp = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    b1 = tmp.textbbox((0, 0), text, font=f1)
    b2 = tmp.textbbox((0, 0), sub, font=f2) if sub else (0, 0, 0, 0)
    tw = max(b1[2] - b1[0], b2[2] - b2[0])
    bar, pad = int(14 * s), int(26 * s)
    h = (b1[3] - b1[1]) + ((b2[3] - b2[1]) + int(10 * s) if sub else 0) + pad * 2
    im = Image.new("RGBA", (bar + tw + pad * 2 + int(10 * s), h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((0, 0, im.width - 1, im.height - 1), radius=int(18 * s), fill=(255, 255, 255, 235))
    d.rounded_rectangle((0, 0, bar, im.height - 1), radius=int(8 * s), fill=accent + (255,))
    x = bar + pad
    d.text((x - b1[0], pad - b1[1]), text, font=f1, fill=(43, 43, 43, 255))
    if sub:
        d.text((x - b2[0], pad + (b1[3] - b1[1]) + int(10 * s) - b2[1]), sub, font=f2, fill=(90, 90, 90, 255))
    return im


# ───────────── 챕터 타이틀 ─────────────
def render_chapter(idx: int, title: str, W: int, H: int, accent=(181, 234, 215)) -> Image.Image:
    s = H / 1080
    f1, f2 = font(int(34 * s), "extrabold"), font(int(40 * s), "bold")
    num = f"{idx:02d}"
    tmp = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    b1, b2 = tmp.textbbox((0, 0), num, font=f1), tmp.textbbox((0, 0), title, font=f2)
    pad, gap = int(22 * s), int(18 * s)
    h = max(b1[3] - b1[1], b2[3] - b2[1]) + pad * 2
    nw = (b1[2] - b1[0]) + pad * 2
    im = Image.new("RGBA", (nw + gap + (b2[2] - b2[0]) + pad * 2, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((0, 0, im.width - 1, h - 1), radius=int(h / 2), fill=(43, 43, 43, 230))
    d.rounded_rectangle((0, 0, nw, h - 1), radius=int(h / 2), fill=accent + (255,))
    d.text((pad - b1[0], (h - (b1[3] - b1[1])) // 2 - b1[1]), num, font=f1, fill=(43, 43, 43, 255))
    d.text((nw + gap - b2[0], (h - (b2[3] - b2[1])) // 2 - b2[1]), title, font=f2, fill=(255, 255, 255, 255))
    return im


# ───────────── LLM 계획 ─────────────
VIS_PROMPT = """당신은 설명 영상 편집자입니다. 아래 발화 타임라인(컷편집 후 기준){script_note}을 보고 두 가지를 뽑으세요.

1. chapters: 주제가 바뀌는 지점의 챕터 타이틀. 최대 {chapter_max}개, 제목 10자 이내, 첫 챕터는 0초.
2. keywords: 시청자가 꼭 기억해야 할 핵심 용어/숫자/결론을 화면에 강조할 지점. 최대 {keyword_max}개, 간격 20초 이상.
   text 는 2~8자 핵심어, sub 는 15자 이내 보충 설명(없으면 빈 문자열). start 는 그 말이 나오는 시각(초), duration 3~5초.

JSON 으로만 답하세요:
{{"chapters": [{{"start": 0, "title": "복리란"}}], "keywords": [{{"start": 12.5, "duration": 4, "text": "복리", "sub": "이자에 이자가 붙는다"}}]}}

=== 발화 타임라인 ===
{timeline}
{script_block}"""


def plan_visuals(script: str, transcript: dict, work: Path, cfg: dict, force: bool = False) -> dict:
    out = work / "visuals.json"
    if out.exists() and not force:
        return json.loads(out.read_text(encoding="utf-8"))
    v = cfg["visuals"]
    data = {"subtitles": build_subtitles(transcript, int(v["subtitle"]["max_chars"]), pii_regex=cfg["detect"]["pii_regex"]),
            "chapters": [], "keywords": []}
    if (v.get("keywords") or v.get("chapters")) and llm.available(cfg):
        timeline = "\n".join(f"[{fmt_time(s['start'])}] {s['text']}" for s in transcript["segments"])
        prompt = VIS_PROMPT.format(chapter_max=v["chapter_max"], keyword_max=v["keyword_max"], timeline=timeline,
                                   script_note=" 과 대본" if script else "",
                                   script_block=f"\n=== 대본 ===\n{script}" if script else "")
        try:
            res = llm.ask_json(cfg, prompt, effort="low")
            dur = float(transcript.get("duration") or 1e9)
            if v.get("chapters"):
                data["chapters"] = [{"start": round(float(c["start"]), 2), "title": str(c["title"])[:14]}
                                    for c in res.get("chapters", []) if float(c.get("start", 0)) < dur][: int(v["chapter_max"])]
            if v.get("keywords"):
                last = -1e9
                for k in sorted(res.get("keywords", []), key=lambda k: float(k.get("start", 0))):
                    s = float(k.get("start", 0))
                    if s - last < 15 or s >= dur or not k.get("text"):
                        continue
                    data["keywords"].append({"start": round(s, 2), "duration": min(6.0, max(2.5, float(k.get("duration", 4)))),
                                             "text": str(k["text"])[:12], "sub": str(k.get("sub", ""))[:20]})
                    last = s
                data["keywords"] = data["keywords"][: int(v["keyword_max"])]
        except RuntimeError as e:
            data["llm_error"] = str(e)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return data


def save_visuals(data: dict, work: Path) -> None:
    (work / "visuals.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


# ───────────── 적용 ─────────────
def apply_visuals(video: Path, data: dict, work: Path, cfg: dict, out: Path) -> Path:
    v = cfg["visuals"]
    info = probe(video)
    W, H = info["width"], info["height"]
    d = work / "visuals_png"
    d.mkdir(parents=True, exist_ok=True)
    for p in d.glob("*.png"):
        p.unlink()
    items = []
    s = H / 1080
    if v.get("subtitles"):
        st = v["subtitle"]
        for i, l in enumerate(data.get("subtitles", [])):
            im = render_subtitle(l["text"], W, H, st)
            p = d / f"sub_{i:04d}.png"
            im.save(p)
            items.append({"png": str(p), "start": l["start"], "end": l["end"], "x": (W - im.width) // 2,
                          "y": H - im.height - int(st.get("margin_bottom", 60) * s), "w": im.width, "h": im.height, "fade": 0.0})
    if v.get("keywords"):
        for i, k in enumerate(data.get("keywords", [])):
            im = render_keyword(k["text"], k.get("sub", ""), W, H)
            p = d / f"kw_{i:03d}.png"
            im.save(p)
            items.append({"png": str(p), "start": k["start"], "end": k["start"] + k["duration"], "x": int(60 * s),
                          "y": H - im.height - int(170 * s), "w": im.width, "h": im.height, "fade": 0.25})
    if v.get("chapters"):
        for i, c in enumerate(data.get("chapters", []), 1):
            im = render_chapter(i, c["title"], W, H)
            p = d / f"ch_{i:02d}.png"
            im.save(p)
            items.append({"png": str(p), "start": c["start"], "end": c["start"] + float(v.get("chapter_duration", 4)),
                          "x": int(60 * s), "y": int(60 * s), "w": im.width, "h": im.height, "fade": 0.3})
    return compose(video, items, out, cfg)
