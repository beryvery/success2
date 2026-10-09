#!/usr/bin/env python
"""영상 자동 편집 파이프라인.

사용법:
  python pipeline.py 입력영상.mp4 [--script 대본.txt] [--out out/] [--steps cut,detect,mosaic,illustrate,visuals]
  python pipeline.py 사진.jpg                      # 이미지 모자이크만
옵션:
  --force        기존 중간 결과 무시하고 다시 계산
  --no-claude    LLM(Claude/GPT) 사용 안 함 (정규식/OCR만)
  --faces        얼굴도 모자이크
  --dry-run      계획(컷 목록/검출 목록)만 만들고 영상 인코딩은 생략
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import yaml

from editpipe import llm_client as llm
from editpipe.cuts import apply_cuts, plan_cuts
from editpipe.detect import detect_image, detect_video
from editpipe.illustrate import generate_illustrations, overlay_illustrations, plan_illustrations
from editpipe.media import fmt_time, is_image, probe
from editpipe.mosaic import apply_mosaic_image, apply_mosaic_video
from editpipe.transcribe import remap_transcript, transcribe
from editpipe.visuals import apply_visuals, plan_visuals

ROOT = Path(__file__).resolve().parent


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def write_report(work: Path, cuts: dict | None, det: dict | None, ills: list[dict] | None, final: Path | None, vis: dict | None = None):
    lines = ["# 편집 리포트", ""]
    if cuts:
        lines += [f"## 컷 편집", f"- 원본 {cuts['source_duration']:.1f}초 → 결과 {cuts['result_duration']:.1f}초 (제거 {cuts['removed_seconds']:.1f}초, {len(cuts['removed'])}곳)",
                  f"- 상세: `cuts_review.csv` / 수정하려면 `cuts.json` 의 keep 을 고치고 `--steps cut --force` 없이 재실행", ""]
        fill = [r for r in cuts["removed"] if r["reason"].startswith("필러")]
        if fill:
            lines.append("필러워드 제거: " + ", ".join(f"{fmt_time(r['start'])} {r['reason']}" for r in fill[:40]) + (" ..." if len(fill) > 40 else ""))
            lines.append("")
    if det is not None:
        lines += ["## 모자이크", f"- LLM 사용: {det.get('llm_used') or '아니오(정규식/OCR만)'}",
                  f"- 검출 {len(det['tracks'])}건 (컷편집 후 타임라인 기준). 오검출은 `detections.json` 에서 삭제 후 재실행", ""]
        for t in det["tracks"]:
            lines.append(f"- {fmt_time(t['start'])}~{fmt_time(t['end'])} [{t['type']}] {t['label']}")
        lines.append("")
    if ills is not None:
        lines += ["## 설명 일러스트", f"- {len(ills)}개 삽입. 파일: `illustrations/`, 계획: `illustrations/plan.json`", ""]
        for i in ills:
            lines.append(f"- {fmt_time(i['start'])} ({i['duration']}s) [{i.get('kind', 'object')}] {i['concept']} — \"{i.get('text', '')}\"")
        lines.append("")
    if vis is not None:
        lines += ["## 자막·시각 요소", f"- 자막 {len(vis.get('subtitles', []))}줄, 키워드 강조 {len(vis.get('keywords', []))}개, 챕터 {len(vis.get('chapters', []))}개 (`visuals.json`)", ""]
        for c in vis.get("chapters", []):
            lines.append(f"- 챕터 {fmt_time(c['start'])} {c['title']}")
        for k in vis.get("keywords", []):
            lines.append(f"- 키워드 {fmt_time(k['start'])} {k['text']} ({k.get('sub', '')})")
        lines.append("")
    if final:
        lines += ["## 결과", f"- 최종 파일: `{final}`"]
    (work / "report.md").write_text("\n".join(lines), encoding="utf-8")


def process_image(src: Path, out_dir: Path, cfg: dict, use_claude: bool):
    out_dir.mkdir(parents=True, exist_ok=True)
    hits = detect_image(src, cfg, use_claude)
    out = out_dir / f"{src.stem}_mosaic{src.suffix}"
    apply_mosaic_image(src, hits, out, cfg)
    (out_dir / f"{src.stem}_detections.json").write_text(json.dumps(hits, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"이미지 검출 {len(hits)}건 → {out}")
    for h in hits:
        log(f"  [{h['type']}] {h['label']}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="영상 또는 이미지 파일")
    ap.add_argument("--script", help="대본 텍스트 파일")
    ap.add_argument("--out", default="out", help="출력 폴더")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--steps", default="cut,detect,mosaic,illustrate,visuals")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-claude", action="store_true")
    ap.add_argument("--faces", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.faces:
        cfg["detect"]["faces"] = True
    use_claude = not args.no_claude
    if use_claude and not llm.available(cfg):
        log("LLM API 키가 없어 규칙 기반으로만 동작합니다 (일러스트 생략, 이름/로고 판별 생략). .env 에 ANTHROPIC_API_KEY 또는 OPENAI_API_KEY 입력 시 활성화.")
        use_claude = False
    elif use_claude:
        log(f"LLM: {llm.provider(cfg)} ({llm.model_name(cfg)})")
    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    script = Path(args.script).read_text(encoding="utf-8") if args.script else ""
    out_root = Path(args.out)

    for src in args.inputs:
        src = Path(src)
        if not src.exists():
            log(f"파일 없음: {src}")
            continue
        if is_image(src):
            process_image(src, out_root, cfg, use_claude)
            continue

        work = out_root / src.stem
        work.mkdir(parents=True, exist_ok=True)
        info = probe(src)
        log(f"=== {src.name}  {info['width']}x{info['height']} {info['fps']:.2f}fps {info['duration']:.1f}s ===")
        current = src
        cuts = det = ills = vis = None

        # 1. 전사
        log("전사(Whisper) 중...")
        tr = transcribe(src, work, cfg, args.force)
        log(f"  단어 {len(tr['words'])}개, 문장 {len(tr['segments'])}개")

        # 2. 컷 편집
        if "cut" in steps:
            cuts = plan_cuts(src, tr, work, cfg, args.force)
            log(f"컷 편집: {cuts['source_duration']:.1f}s → {cuts['result_duration']:.1f}s (제거 {len(cuts['removed'])}곳, {cuts['removed_seconds']:.1f}s)")
            tr = remap_transcript(tr, cuts["keep"])
            (work / "transcript_cut.json").write_text(json.dumps(tr, ensure_ascii=False, indent=1), encoding="utf-8")
            if not args.dry_run:
                cut_file = work / "1_cut.mp4"
                if not cut_file.exists() or args.force:
                    apply_cuts(src, cuts, cut_file, cfg)
                current = cut_file

        # 3. 검출 + 모자이크
        if "detect" in steps or "mosaic" in steps:
            log("개인정보/상표 검출 중 (OCR" + (f", {llm.provider(cfg)}" if use_claude else "") + ")...")
            det = detect_video(current, work, cfg, args.force, use_claude)
            log(f"  검출 {len(det['tracks'])}건")
            for t in det["tracks"][:30]:
                log(f"    {fmt_time(t['start'])}~{fmt_time(t['end'])} [{t['type']}] {t['label']}")
            if "mosaic" in steps and not args.dry_run:
                mos_file = work / "2_mosaic.mp4"
                if not mos_file.exists() or args.force:
                    apply_mosaic_video(current, det["tracks"], mos_file, cfg)
                current = mos_file

        # 4. 설명 일러스트
        if "illustrate" in steps and cfg["illustrate"].get("enabled", True):
            if not use_claude:
                log("일러스트: LLM 키 없음 → 건너뜀")
            else:
                log("일러스트 계획/생성 중...")
                try:
                    plan = plan_illustrations(script, tr, work, cfg, args.force)
                    from editpipe.illustrate import engine
                    log(f"  계획 {len(plan)}개 → 생성 엔진 {engine(cfg)}, 병렬 {cfg['illustrate'].get('parallel', 4)}")
                    ills = generate_illustrations(plan, work, cfg, args.force)
                    log(f"  일러스트 {len(ills)}개")
                    if not args.dry_run:
                        ill_file = work / "3_illustrated.mp4"
                        overlay_illustrations(current, ills, ill_file, cfg)
                        current = ill_file
                except RuntimeError as e:
                    log(f"  일러스트 실패: {e}")

        # 5. 자막·키워드·챕터
        if "visuals" in steps:
            log("자막·시각 요소 생성 중" + (" (키워드/챕터는 LLM)" if use_claude else " (자막만)") + "...")
            vis = plan_visuals(script, tr, work, cfg, args.force)
            log(f"  자막 {len(vis['subtitles'])}줄, 키워드 {len(vis['keywords'])}개, 챕터 {len(vis['chapters'])}개")
            if not args.dry_run:
                vis_file = work / "4_visuals.mp4"
                apply_visuals(current, vis, work, cfg, vis_file)
                current = vis_file

        final = None
        if not args.dry_run and current != src:
            final = out_root / f"{src.stem}_edited.mp4"
            shutil.copyfile(current, final)
            log(f"완료 → {final}")
        write_report(work, cuts, det, ills, final, vis)
        log(f"리포트: {work / 'report.md'}")


if __name__ == "__main__":
    main()
