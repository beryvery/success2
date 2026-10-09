"""설명 일러스트: LLM이 삽입 지점 계획 → SVG 생성 → PNG 렌더 → 영상에 오버레이.

kind = "object" : 아이콘/사물/심볼 하나. 투명 배경으로 화면 한쪽(기본 우측 중앙)에 얹음
kind = "card"   : 관계/흐름/비교 등 장면. 중앙 카드로 표시
"""
from __future__ import annotations

import base64
import json
import os
import re
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import llm_client as llm
from .compose import compose
from .media import fmt_time, probe, run

PLAN_PROMPT = """당신은 교육/설명 영상 편집자입니다. 아래 **대본**과 **실제 발화 타임라인(컷편집 후 기준)**을 보고,
시청자의 이해를 돕기 위해 설명 그림을 끼워 넣을 지점을 고르세요.

규칙:
- 최대 {max_count}개, 지점 간 최소 {min_gap}초 간격. 영상 길이 {duration:.0f}초.
- 추상적 개념, 구조/흐름, 비교, 숫자 관계, 과정 설명처럼 **그림이 있어야 이해가 쉬운 순간**만 고르세요. 인사말/잡담은 제외.
- kind 선택:
  - "object": 아이콘/사물/심볼 **하나**로 뜻이 통할 때 (예: 돈주머니, 시계, 돋보기, 자물쇠). 투명 배경으로 화면 한쪽에 얹음.
  - "card": 요소 2개 이상의 관계·흐름·비교·과정을 보여줘야 할 때. 중앙 카드로 표시.
- 글자는 최소(최대 2단어). 가능하면 글자 없이 그림만으로.
- start 는 해당 설명이 시작되는 발화 시각(초). duration 은 {duration_default}초 전후(3~8초).
- visual: 그릴 내용을 구체적으로 묘사 (등장 요소, 배치, 화살표/관계, 색 포인트).

JSON 배열로만 답하세요:
[{{"start": 12.5, "duration": 5, "kind": "object", "concept": "핵심 개념 한 줄", "text": "최대 2단어 또는 빈문자열", "visual": "그림 묘사"}}]

=== 대본 ===
{script}

=== 발화 타임라인 ===
{timeline}
"""

SVG_COMMON = """다음 묘사를 SVG 일러스트로 그리세요.

스타일 (반드시 준수):
{style}

기술 요구사항:
- 루트: <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">
- 손그림 느낌: <defs> 안에 feTurbulence(type="fractalNoise" baseFrequency="0.015" numOctaves="2") + feDisplacementMap(scale 2~3) 필터를 정의하고
  도형 그룹에만 filter 적용 (텍스트에는 적용 금지). 선은 stroke-linecap="round" stroke-linejoin="round".
- 외부 이미지/폰트/스크립트 금지. 그라데이션·그림자 금지.
- 텍스트가 있다면 font-family="Apple SD Gothic Neo, Noto Sans KR, sans-serif" font-weight="700", 글자 크기 {font_px}px 이상, 색 #2B2B2B.
- 요소는 크고 단순하게. SVG 코드만 출력. 설명/마크다운 금지.
{kind_rules}

개념: {concept}
텍스트(있다면): {text}
묘사: {visual}
"""

OBJECT_RULES = """- 종류: **오브젝트**. 배경 없음(완전 투명). 배경 사각형을 그리지 마세요.
- 캔버스 중앙에 오브젝트 **하나**를 캔버스의 75~85% 크기로 크게. 글자는 가급적 없음(있다면 오브젝트 아래 1~2단어)."""

CARD_RULES = """- 종류: **카드**. 맨 아래 레이어로 둥근 모서리(rx=48) 흰색(#FFFFFF) 카드를 캔버스 전체에 그리고 외곽선 #2B2B2B 4px.
- 카드 안쪽 여백 60px 이상. 요소 간 관계는 두꺼운 화살표(#2B2B2B)로. 글자는 최대 2단어."""


def _timeline_text(tr: dict) -> str:
    return "\n".join(f"[{fmt_time(s['start'])}] {s['text']}" for s in tr["segments"])


def plan_illustrations(script: str, transcript: dict, work: Path, cfg: dict, force: bool = False) -> list[dict]:
    out = work / "illustrations" / "plan.json"
    if out.exists() and not force:
        return json.loads(out.read_text(encoding="utf-8"))
    out.parent.mkdir(parents=True, exist_ok=True)
    il = cfg["illustrate"]
    duration = float(transcript.get("duration") or 0)
    max_count = max(1, min(int(il["max_count"]), int(duration // int(il["min_gap"])) + 1))
    prompt = PLAN_PROMPT.format(max_count=max_count, min_gap=il["min_gap"], duration=duration,
                                duration_default=il["duration"], script=script or "(대본 없음 — 타임라인만 참고)",
                                timeline=_timeline_text(transcript))
    plan = llm.ask_json(cfg, prompt, effort="medium")
    cleaned, last = [], -1e9
    for p in sorted(plan, key=lambda x: float(x.get("start", 0))):
        s = float(p.get("start", 0))
        d = min(8.0, max(3.0, float(p.get("duration", il["duration"]))))
        if s - last < float(il["min_gap"]) or s + d > duration + 0.5:
            continue
        kind = "card" if str(p.get("kind", "")).lower() == "card" else "object"
        cleaned.append({"start": round(s, 2), "duration": d, "kind": kind, "concept": str(p.get("concept", "")),
                        "text": str(p.get("text", ""))[:20], "visual": str(p.get("visual", ""))})
        last = s
        if len(cleaned) >= max_count:
            break
    save_plan(cleaned, work)
    return cleaned


def save_plan(plan: list[dict], work: Path) -> None:
    out = work / "illustrations" / "plan.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan, ensure_ascii=False, indent=1), encoding="utf-8")


def _sanitize_svg(svg: str) -> str:
    svg = svg.strip()
    m = re.search(r"<svg[\s\S]*</svg>", svg)
    if not m:
        raise ValueError("SVG 태그를 찾지 못했습니다.")
    svg = m.group(0)
    svg = re.sub(r"<script[\s\S]*?</script>", "", svg, flags=re.I)
    svg = re.sub(r"<image[^>]*>", "", svg, flags=re.I)
    ET.fromstring(svg)  # XML 유효성
    return svg


def render_svg(svg: str, png: Path, w: int, h: int) -> Path:
    """투명 배경 PNG 렌더 (SVG 안에 배경 도형이 없으면 그대로 투명)."""
    import resvg_py
    data = resvg_py.svg_to_bytes(svg_string=svg, width=w, height=h, font_family="Apple SD Gothic Neo")
    png.write_bytes(data)
    return png


def canvas_for(kind: str, cfg: dict) -> tuple[int, int]:
    il = cfg["illustrate"]
    w, h = il["card_canvas"] if kind == "card" else il["object_canvas"]
    return int(w), int(h)


IMG_OBJECT_PROMPT = """{style}
Single {concept} icon illustration: {visual}
Transparent background, one object centered filling ~80% of the canvas, flat vector + subtle hand-drawn cartoon feel,
uniform rounded dark outline (#2B2B2B), pastel palette (#FFB4A2, #B5EAD7, #C7CEEA, #FFDAC1, #E2F0CB), no shadows, no gradients, no texture.
{text_rule}"""

IMG_CARD_PROMPT = """{style}
Explanatory scene illustration: {visual}
Draw it inside a white rounded-corner card (#FFFFFF, thin dark outline) that fills the canvas; outside the card is transparent.
Flat vector + subtle hand-drawn cartoon feel, uniform rounded dark outline (#2B2B2B), pastel palette (#FFB4A2, #B5EAD7, #C7CEEA, #FFDAC1, #E2F0CB),
simple geometric shapes, generous margins, thick arrows for relationships, no shadows, no gradients, no texture.
{text_rule}"""


def engine(cfg: dict) -> str:
    """svg | gpt-image"""
    e = (cfg["illustrate"].get("engine") or "auto").lower()
    if e == "auto":
        llm.load_env_file()
        return "gpt-image" if os.environ.get("OPENAI_API_KEY") else "svg"
    return "gpt-image" if e in ("gpt-image", "openai", "gpt") else "svg"


def _gen_gpt_image(item: dict, png_path: Path, cfg: dict) -> Path:
    import openai

    il = cfg["illustrate"]
    kind = item.get("kind", "object")
    text = (item.get("text") or "").strip()
    text_rule = (f'Include only the text "{text}" in large bold handwritten-style Korean lettering (Apple SD Gothic Neo look), nothing else.'
                 if text else "No text, no letters, no numbers anywhere in the image.")
    tpl = IMG_CARD_PROMPT if kind == "card" else IMG_OBJECT_PROMPT
    prompt = tpl.format(style=il["style"].strip(), concept=item.get("concept", ""), visual=item.get("visual", ""), text_rule=text_rule)
    client = openai.OpenAI()
    try:
        resp = client.images.generate(model=il.get("openai_image_model", "gpt-image-1"), prompt=prompt, n=1,
                                      size="1536x1024" if kind == "card" else "1024x1024",
                                      background="transparent", output_format="png",
                                      quality=il.get("openai_image_quality", "medium"))
    except openai.AuthenticationError as e:
        raise RuntimeError("OpenAI API 인증 실패: api_keys.txt 의 OPENAI_API_KEY 확인") from e
    except openai.APIStatusError as e:
        raise RuntimeError("이미지 생성: " + llm.openai_error(e)) from e
    except openai.APIConnectionError as e:
        raise RuntimeError("OpenAI 네트워크 오류") from e
    data = resp.data[0].b64_json
    if not data:
        raise RuntimeError("OpenAI 이미지 응답이 비어 있습니다.")
    png_path.write_bytes(base64.b64decode(data))
    return png_path


def _gen_svg(item: dict, svg_path: Path, png_path: Path, cfg: dict) -> Path:
    il = cfg["illustrate"]
    kind = item.get("kind", "object")
    w, h = canvas_for(kind, cfg)
    prompt = SVG_COMMON.format(style=il["style"], w=w, h=h, font_px=int(h / 10),
                               kind_rules=CARD_RULES if kind == "card" else OBJECT_RULES,
                               concept=item["concept"], text=item.get("text") or "(없음)", visual=item["visual"])
    last_err = None
    for _ in range(2):
        try:
            svg = _sanitize_svg(llm.ask(cfg, prompt, max_tokens=32000, effort="medium", stream=True))
            render_svg(svg, png_path, w, h)
            svg_path.write_text(svg, encoding="utf-8")
            return png_path
        except (ValueError, ET.ParseError) as e:
            last_err = e
    raise RuntimeError(f"SVG 생성 실패: {last_err}")


def generate_one(item: dict, idx: int, work: Path, cfg: dict) -> Path:
    d = work / "illustrations"
    d.mkdir(parents=True, exist_ok=True)
    svg_path, png_path = d / f"ill_{idx:02d}.svg", d / f"ill_{idx:02d}.png"
    try:
        if engine(cfg) == "gpt-image":
            return _gen_gpt_image(item, png_path, cfg)
        return _gen_svg(item, svg_path, png_path, cfg)
    except RuntimeError as e:
        raise RuntimeError(f"일러스트 {idx} 생성 실패: {e}") from e


def generate_illustrations(plan: list[dict], work: Path, cfg: dict, force: bool = False,
                           progress=None) -> list[dict]:
    """계획 항목을 병렬로 생성. progress(done, total, msg) 콜백 선택."""
    jobs = []
    for i, p in enumerate(plan, 1):
        png_path = work / "illustrations" / f"ill_{i:02d}.png"
        if force or not png_path.exists():
            jobs.append((i, p))
    errors: dict[int, str] = {}
    if jobs:
        workers = max(1, int(cfg["illustrate"].get("parallel", 4)))
        done = 0

        def _run(job):
            i, p = job
            try:
                generate_one(p, i, work, cfg)
                return i, None
            except RuntimeError as e:
                return i, str(e)

        with ThreadPoolExecutor(max_workers=workers) as ex:
            for i, err in ex.map(_run, jobs):
                done += 1
                if err:
                    errors[i] = err
                    print(f"  [건너뜀] {err}")
                if progress:
                    progress(done, len(jobs), f"일러스트 {done}/{len(jobs)}")
    out = []
    for i, p in enumerate(plan, 1):
        png_path = work / "illustrations" / f"ill_{i:02d}.png"
        if png_path.exists() and i not in errors:
            out.append({**p, "png": str(png_path)})
    return out


def _placement(kind: str, cw: int, ch: int, W: int, H: int, il: dict) -> tuple[int, int, int, int]:
    """(x, y, w, h) 픽셀 위치."""
    if kind == "card":
        tw = int(W * float(il["card_width"]))
        th = int(tw * ch / cw)
        if th > H * 0.85:
            th = int(H * 0.85)
            tw = int(th * cw / ch)
        return (W - tw) // 2, (H - th) // 2, tw, th
    tw = int(W * float(il["object_width"]))
    th = int(tw * ch / cw)
    margin = int(W * 0.04)
    x = margin if il.get("object_position") == "left" else W - tw - margin
    return x, (H - th) // 2, tw, th


def overlay_illustrations(video: Path, items: list[dict], out: Path, cfg: dict) -> Path:
    il = cfg["illustrate"]
    info = probe(video)
    W, H = info["width"], info["height"]
    layers = []
    for it in items:
        kind = it.get("kind", "object")
        from PIL import Image
        with Image.open(it["png"]) as im:
            cw, ch = im.size
        x, y, tw, th = _placement(kind, cw, ch, W, H, il)
        s = float(it["start"])
        layers.append({"png": it["png"], "start": s, "end": s + float(it["duration"]), "x": x, "y": y, "w": tw, "h": th,
                       "fade": float(il["fade"])})
    return compose(video, layers, out, cfg)
