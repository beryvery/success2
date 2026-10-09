import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from editpipe.cuts import build_cut_list, find_fillers
from editpipe.detect import build_tracks, classify_text_rules
from editpipe.mosaic import pixelate
from editpipe.transcribe import remap_transcript
from editpipe.illustrate import _sanitize_svg, render_svg

CFG = yaml.safe_load((Path(__file__).resolve().parent.parent / "config.yaml").read_text(encoding="utf-8"))


def test_fillers_and_cuts():
    words = [{"word": "안녕하세요", "start": 0.5, "end": 1.2}, {"word": "어", "start": 1.4, "end": 1.6},
             {"word": "오늘은", "start": 1.8, "end": 2.3}, {"word": "음...", "start": 2.5, "end": 2.9},
             {"word": "어머니", "start": 3.0, "end": 3.5}]
    f = find_fillers(words, CFG["cut"]["fillers"], 1.2, 0.04)
    assert [x["reason"] for x in f] == ["필러 '어'", "필러 '음...'"]
    plan = build_cut_list(10.0, [(4.0, 6.0), (9.0, None)], f, CFG["cut"])
    keep = plan["keep"]
    assert keep[0]["start"] == 0.0 and abs(keep[0]["end"] - 1.36) < 1e-6
    assert abs(keep[-1]["end"] - 9.18) < 1e-6            # 끝 무음은 keep_silence 만큼 남김
    assert any(abs(k["start"] - 5.82) < 1e-6 for k in keep)
    assert abs(plan["result_duration"] - (10 - plan["removed_seconds"])) < 1e-6


def test_remap():
    tr = {"words": [{"word": "a", "start": 0.5, "end": 1.0}, {"word": "b", "start": 2.5, "end": 3.0}],
          "segments": [{"start": 0.5, "end": 3.0, "text": "a b"}], "duration": 4}
    keep = [{"start": 0.0, "end": 1.2}, {"start": 2.0, "end": 4.0}]
    r = remap_transcript(tr, keep)
    assert r["words"][1]["start"] == 1.7 and r["duration"] == 3.2


def test_pii_rules():
    d = CFG["detect"]
    assert classify_text_rules("연락처 010-1234-5678", d["pii_regex"], d["brand_words"]) == ("pii", "전화번호")
    assert classify_text_rules("mail: hong@example.com", d["pii_regex"], d["brand_words"])[1] == "이메일"
    assert classify_text_rules("Samsung Galaxy", d["pii_regex"], d["brand_words"]) == ("brand", "samsung")
    assert classify_text_rules("오늘의 날씨 2026", d["pii_regex"], d["brand_words"]) is None


def test_tracks_merge():
    hits = [{"t": t, "type": "pii", "label": "x", "box": [0.1, 0.1, 0.3, 0.2]} for t in (0.0, 0.5, 1.0)]
    hits.append({"t": 5.0, "type": "pii", "label": "x", "box": [0.1, 0.1, 0.3, 0.2]})
    tr = build_tracks(hits, 0.5, 0.6, 0.1)
    assert len(tr) == 2 and tr[0]["count"] == 3 and tr[0]["end"] > 1.5


def test_pixelate():
    f = np.random.randint(0, 255, (120, 160, 3), dtype=np.uint8)
    orig = f.copy()
    pixelate(f, [0.25, 0.25, 0.75, 0.75], 10)
    assert not np.array_equal(f[30:90, 40:120], orig[30:90, 40:120])
    assert np.array_equal(f[:30], orig[:30])        # 바깥은 그대로
    assert len(np.unique(f[30:40, 40:50].reshape(-1, 3), axis=0)) == 1  # 블록 내 단색


def test_svg_render(tmp_path):
    svg = _sanitize_svg('```svg\n<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><script>x</script>'
                        '<rect width="100" height="100" fill="#FFF8E7"/><text x="10" y="50" font-size="20">안녕</text></svg>\n```')
    assert "<script" not in svg
    p = render_svg(svg, tmp_path / "a.png", 200, 200)
    assert p.stat().st_size > 100
