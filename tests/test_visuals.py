import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from editpipe.compose import compose, blend, _Layer
from editpipe.visuals import mask_pii, build_subtitles, render_subtitle, render_keyword, render_chapter, apply_visuals

CFG = yaml.safe_load((Path(__file__).resolve().parent.parent / "config.yaml").read_text(encoding="utf-8"))


def test_build_subtitles_splits_on_length_gap_and_sentence():
    words = [{"word": w, "start": i * 0.5, "end": i * 0.5 + 0.4} for i, w in enumerate(
        ["안녕하세요.", "오늘은", "복리의", "원리를", "설명해", "드리겠습니다.", "아주", "긴", "문장을", "계속", "이어서", "말하면", "자막이", "길어집니다"])]
    words[7]["start"] += 2.0  # "긴" 앞에 2초 공백
    for w in words[7:]:
        w["start"] += 2.0; w["end"] += 2.0
    lines = build_subtitles({"words": words}, max_chars=30)
    assert lines[0]["text"] == "안녕하세요."                 # 문장 끝에서 분리
    assert lines[1]["text"] == "오늘은 복리의 원리를 설명해 드리겠습니다."
    assert lines[2]["text"] == "아주"                       # 공백 2초로 분리
    assert all(l["end"] > l["start"] for l in lines)
    for a, b in zip(lines, lines[1:]):
        assert a["end"] <= b["start"]


def test_renderers_transparent():
    for im in (render_subtitle("테스트 자막", 1920, 1080, CFG["visuals"]["subtitle"]),
               render_keyword("복리", "이자에 이자", 1920, 1080), render_chapter(1, "복리란", 1920, 1080)):
        assert im.mode == "RGBA" and im.width > 10 and im.height > 10
        assert im.getpixel((0, 0))[3] == 0   # 모서리는 투명


def test_blend_and_compose(tmp_path):
    png = tmp_path / "o.png"
    Image.new("RGBA", (40, 40), (255, 0, 0, 255)).save(png)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    layer = _Layer({"png": str(png), "start": 0, "end": 1, "x": 10, "y": 10, "w": 40, "h": 40, "fade": 0.5})
    blend(frame, layer, 0.25)                     # fade-in 중간 → 50%
    assert 120 <= frame[20, 20, 0] <= 135 and frame[5, 5, 0] == 0
    # 실제 영상 합성
    src = tmp_path / "src.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=160x90:r=10:d=2",
                    "-f", "lavfi", "-i", "sine=f=440:d=2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(src)], check=True)
    out = tmp_path / "out.mp4"
    compose(src, [{"png": str(png), "start": 0.5, "end": 1.5, "x": 0, "y": 0, "w": 40, "h": 40, "fade": 0}], out, CFG)
    r = subprocess.run(["ffmpeg", "-v", "error", "-ss", "1.0", "-i", str(out), "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                       capture_output=True, check=True)
    f = np.frombuffer(r.stdout, dtype=np.uint8).reshape(90, 160, 3)
    assert f[10, 10, 0] > 200 and f[80, 150, 0] < 30


def test_apply_visuals(tmp_path):
    src = tmp_path / "src.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=gray:s=320x180:r=10:d=3",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(src)], check=True)
    data = {"subtitles": [{"start": 0.2, "end": 1.5, "text": "자막 테스트"}],
            "keywords": [{"start": 0.5, "duration": 2, "text": "핵심", "sub": "설명"}],
            "chapters": [{"start": 0, "title": "시작"}]}
    out = apply_visuals(src, data, tmp_path, CFG, tmp_path / "v.mp4")
    assert out.exists() and out.stat().st_size > 1000
    assert len(list((tmp_path / "visuals_png").glob("*.png"))) == 3


def test_mask_pii_in_subtitles():
    r = CFG["detect"]["pii_regex"]
    assert mask_pii("문의는 01012345678로 연락 주세요.", r) == "문의는 ***로 연락 주세요."
    assert mask_pii("010-1234-5678 또는 hong@example.com", r) == "*** 또는 ***"
    assert mask_pii("100만 원을 연 10 %로", r) == "100만 원을 연 10 %로"
