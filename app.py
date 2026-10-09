"""영상 자동 편집 — 로컬 웹 앱 (Streamlit)
실행: .venv/bin/streamlit run app.py
"""
from __future__ import annotations

import json
import shutil
import time
from functools import partial
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

from editpipe import llm_client as llm
from editpipe.cuts import apply_cuts, plan_cuts
from editpipe.detect import detect_image, detect_video
from editpipe.illustrate import engine as ill_engine, generate_illustrations, generate_one, overlay_illustrations, plan_illustrations, save_plan
from editpipe.media import fmt_time, is_image, probe
from editpipe.mosaic import apply_mosaic_image, apply_mosaic_video
from editpipe.transcribe import remap_transcript, transcribe
from editpipe.visuals import apply_visuals, plan_visuals, save_visuals

ROOT = Path(__file__).resolve().parent
UPLOAD = ROOT / "uploads"
OUT = ROOT / "out"
UPLOAD.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

st.set_page_config(page_title="영상 자동 편집", page_icon="🎬", layout="wide")


@st.cache_data
def load_cfg():
    return yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


cfg = load_cfg()
ss = st.session_state
ss.setdefault("src", None)       # 업로드된 원본 경로
ss.setdefault("log", [])


def log(msg):
    ss.log.append(f"[{time.strftime('%H:%M:%S')}] {msg}")


def work_dir() -> Path:
    return OUT / Path(ss.src).stem


def rj(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def wj(p: Path, data):
    p.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def current_video() -> Path:
    """가장 마지막 단계까지 완료된 영상."""
    w = work_dir()
    for name in ("4_visuals.mp4", "3_illustrated.mp4", "2_mosaic.mp4", "1_cut.mp4"):
        if (w / name).exists():
            return w / name
    return Path(ss.src)


def cut_transcript():
    w = work_dir()
    return rj(w / "transcript_cut.json") or rj(w / "transcript.json")


# ───────────────────────── 사이드바 ─────────────────────────
with st.sidebar:
    st.header("⚙️ 설정")
    prov = llm.provider(cfg)
    if prov:
        st.success(f"LLM: {prov} ({llm.model_name(cfg)})\n\n키 출처: {llm.key_source(cfg)}")
    else:
        st.warning("LLM 키 없음 → 일러스트·이름/로고 판별 비활성.\n`api_keys.txt` 에 키 입력 후 새로고침")
    use_llm = st.checkbox("LLM 사용 (판별·일러스트)", value=bool(prov), disabled=not prov)
    st.divider()
    st.caption("컷 편집")
    cfg["cut"]["min_silence"] = st.slider("무음 최소 길이(초)", 0.3, 2.0, float(cfg["cut"]["min_silence"]), 0.1)
    cfg["cut"]["silence_db"] = st.slider("무음 기준(dB)", -60, -20, int(cfg["cut"]["silence_db"]), 1)
    fillers = st.text_input("필러워드 (쉼표 구분)", ", ".join(cfg["cut"]["fillers"]))
    cfg["cut"]["fillers"] = [f.strip() for f in fillers.split(",") if f.strip()]
    st.caption("모자이크")
    cfg["detect"]["faces"] = st.checkbox("얼굴도 모자이크", value=bool(cfg["detect"]["faces"]))
    cfg["mosaic"]["block"] = st.slider("모자이크 블록 크기", 8, 40, int(cfg["mosaic"]["block"]), 2)
    st.caption("일러스트")
    cfg["illustrate"]["object_position"] = st.radio("오브젝트 위치", ["right", "left"], horizontal=True,
                                                    format_func=lambda v: "우측" if v == "right" else "좌측")
    cfg["illustrate"]["object_width"] = st.slider("오브젝트 크기(화면 폭 대비)", 0.15, 0.4, float(cfg["illustrate"]["object_width"]), 0.01)
    cfg["illustrate"]["card_width"] = st.slider("카드 크기(화면 폭 대비)", 0.4, 0.8, float(cfg["illustrate"]["card_width"]), 0.02)
    cfg["illustrate"]["max_count"] = st.number_input("최대 삽입 수", 1, 20, int(cfg["illustrate"]["max_count"]))
    cfg["illustrate"]["engine"] = st.selectbox("그림 생성 엔진", ["auto", "gpt-image", "svg"],
                                               index=["auto", "gpt-image", "svg"].index(cfg["illustrate"].get("engine", "auto")),
                                               help="gpt-image: OpenAI 이미지 생성(OPENAI_API_KEY 필요) / svg: LLM이 SVG 작성 / auto: OpenAI 키 있으면 gpt-image")
    cfg["illustrate"]["parallel"] = st.slider("동시 생성 수(병렬)", 1, 8, int(cfg["illustrate"].get("parallel", 4)))
    st.caption(f"현재 엔진: {ill_engine(cfg)}")
    st.caption("자막·시각 요소")
    cfg["visuals"]["subtitles"] = st.checkbox("자막", value=bool(cfg["visuals"]["subtitles"]))
    cfg["visuals"]["keywords"] = st.checkbox("키워드 강조 (LLM)", value=bool(cfg["visuals"]["keywords"]), disabled=not prov)
    cfg["visuals"]["chapters"] = st.checkbox("챕터 타이틀 (LLM)", value=bool(cfg["visuals"]["chapters"]), disabled=not prov)
    st.divider()
    with st.expander("로그"):
        st.code("\n".join(ss.log[-40:]) or "-")

st.title("🎬 영상 자동 편집")

# ───────────────────────── 입력 ─────────────────────────
c1, c2 = st.columns([1, 1])
with c1:
    up = st.file_uploader("영상 또는 이미지 업로드", type=["mp4", "mov", "m4v", "mkv", "webm", "jpg", "jpeg", "png", "webp"])
    if up is not None:
        dst = UPLOAD / up.name
        if ss.src != str(dst) or not dst.exists():
            dst.write_bytes(up.getbuffer())
            ss.src = str(dst)
            ss.log = []
            log(f"업로드: {up.name}")
    local = st.text_input("또는 로컬 파일 경로 (큰 영상은 이쪽이 빠름)", placeholder="/Users/rok/Movies/강의.mp4", key="local_path")
    if st.button("이 경로 사용") and local.strip():
        lp = Path(local.strip().strip("'\""))
        if lp.exists():
            ss.src = str(lp)
            ss.log = []
            log(f"선택: {lp}")
            st.rerun()
        else:
            st.error("파일이 없습니다.")
with c2:
    use_script = st.checkbox("대본 사용 (일러스트·키워드·챕터 선정에 참고. 없어도 발화 내용으로 동작)", value=False)
    script_text = ""
    if use_script:
        script_file = st.file_uploader("대본 (txt)", type=["txt", "md"])
        script_text = st.text_area("대본 내용", script_file.getvalue().decode("utf-8", "ignore") if script_file else "", height=150,
                                   placeholder="대본을 붙여 넣거나 파일을 올리세요.")

if not ss.src:
    st.info("파일을 올리면 버튼이 나타납니다.")
    st.stop()

src = Path(ss.src)
work = work_dir()
work.mkdir(parents=True, exist_ok=True)

# ═══════════════════════ 이미지 모드 ═══════════════════════
if is_image(src):
    st.subheader(f"🖼 이미지: {src.name}")
    a, b = st.columns(2)
    a.image(str(src), caption="원본", use_container_width=True)
    if st.button("🔍 개인정보/상표 검출 → 모자이크", type="primary"):
        with st.spinner("검출 중..."):
            hits = detect_image(src, cfg, use_llm)
            out = OUT / f"{src.stem}_mosaic{src.suffix}"
            apply_mosaic_image(src, hits, out, cfg)
            wj(OUT / f"{src.stem}_detections.json", hits)
            ss.img_hits, ss.img_out = hits, str(out)
            log(f"이미지 검출 {len(hits)}건")
    if ss.get("img_out") and Path(ss.img_out).exists():
        b.image(ss.img_out, caption="모자이크 결과", use_container_width=True)
        if ss.get("img_hits"):
            st.dataframe(pd.DataFrame([{"종류": h["type"], "내용": h["label"]} for h in ss.img_hits]), hide_index=True, use_container_width=True)
        else:
            st.caption("검출된 항목 없음")
        with open(ss.img_out, "rb") as f:
            st.download_button("📥 결과 이미지 다운로드", f.read(), Path(ss.img_out).name)
    st.stop()

# ═══════════════════════ 영상 모드 ═══════════════════════
info = probe(src)
st.subheader(f"🎞 {src.name}  ·  {info['width']}×{info['height']}  {info['fps']:.0f}fps  {fmt_time(info['duration'])}")

tr_json = work / "transcript.json"
cuts_json = work / "cuts.json"
det_json = work / "detections.json"
plan_json = work / "illustrations" / "plan.json"
cut_mp4, mos_mp4, ill_mp4, vis_mp4 = work / "1_cut.mp4", work / "2_mosaic.mp4", work / "3_illustrated.mp4", work / "4_visuals.mp4"
vis_json = work / "visuals.json"

b1, b2, b3, b4, b5, b6, b7 = st.columns(7)
run_all = b7.button("▶ 전체 실행", type="primary", use_container_width=True)


def step_transcribe(force=False):
    with st.spinner("① 전사 중 (Whisper)... 처음엔 모델 다운로드로 수 분 걸릴 수 있습니다"):
        tr = transcribe(src, work, cfg, force)
        log(f"전사 완료: 단어 {len(tr['words'])}개")


def step_cut(force=False):
    tr = rj(tr_json)
    if tr is None:
        step_transcribe()
        tr = rj(tr_json)
    with st.spinner("② 컷 편집 중..."):
        plan = plan_cuts(src, tr, work, cfg, force)
        wj(work / "transcript_cut.json", remap_transcript(tr, plan["keep"]))
        apply_cuts(src, plan, cut_mp4, cfg)
        for f in (mos_mp4, ill_mp4, vis_mp4, det_json):     # 이후 단계 무효화
            f.unlink(missing_ok=True)
        log(f"컷 편집: {plan['source_duration']:.1f}s → {plan['result_duration']:.1f}s (제거 {len(plan['removed'])}곳)")


def step_detect(force=False):
    base = cut_mp4 if cut_mp4.exists() else src
    with st.spinner("③ 개인정보/상표 검출 중 (OCR" + (", LLM" if use_llm else "") + ")..."):
        det = detect_video(base, work, cfg, force, use_llm)
        for f in (mos_mp4, ill_mp4, vis_mp4):
            f.unlink(missing_ok=True)
        log(f"검출 {len(det['tracks'])}건")


def step_mosaic():
    det = rj(det_json)
    if det is None:
        step_detect()
        det = rj(det_json)
    base = cut_mp4 if cut_mp4.exists() else src
    with st.spinner("④ 모자이크 적용 중..."):
        apply_mosaic_video(base, det["tracks"], mos_mp4, cfg)
        ill_mp4.unlink(missing_ok=True)
        vis_mp4.unlink(missing_ok=True)
        log("모자이크 완료")


def step_plan_ill(force=False):
    tr = cut_transcript()
    if tr is None:
        step_transcribe()
        tr = cut_transcript()
    with st.spinner("⑤ 일러스트 지점 계획 중 (LLM)..."):
        plan = plan_illustrations(script_text, tr, work, cfg, force)
        log(f"일러스트 계획 {len(plan)}개")


def step_render_ill(force=False):
    plan = rj(plan_json) or []
    prog = st.progress(0, f"일러스트 생성 중... (엔진 {ill_engine(cfg)}, 병렬 {cfg['illustrate']['parallel']})")
    done = generate_illustrations(plan, work, cfg, force, progress=lambda d, n, m: prog.progress(d / n, m))
    prog.empty()
    base = mos_mp4 if mos_mp4.exists() else (cut_mp4 if cut_mp4.exists() else src)
    with st.spinner("일러스트 오버레이 중..."):
        overlay_illustrations(base, done, ill_mp4, cfg)
    vis_mp4.unlink(missing_ok=True)
    log(f"일러스트 {len(done)}개 삽입 완료 ({ill_engine(cfg)})")


def step_visuals(force=False):
    tr = cut_transcript()
    if tr is None:
        step_transcribe()
        tr = cut_transcript()
    with st.spinner("⑥ 자막·키워드·챕터 생성 중..."):
        data = plan_visuals(script_text, tr, work, cfg, force)
        base = ill_mp4 if ill_mp4.exists() else (mos_mp4 if mos_mp4.exists() else (cut_mp4 if cut_mp4.exists() else src))
        apply_visuals(base, data, work, cfg, vis_mp4)
        log(f"시각 요소: 자막 {len(data['subtitles'])}줄, 키워드 {len(data['keywords'])}개, 챕터 {len(data['chapters'])}개")


def safe(*fns) -> bool:
    """단계 함수들을 순서대로 실행. 오류는 화면에 표시하고 중단."""
    for fn in fns:
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            name = getattr(getattr(fn, "func", fn), "__name__", "처리")
            names = {"step_transcribe": "① 전사", "step_cut": "② 컷 편집", "step_detect": "③ 검출",
                     "step_mosaic": "④ 모자이크", "step_plan_ill": "⑤ 일러스트 계획", "step_render_ill": "⑤ 일러스트 생성"}
            msg = f"{names.get(name, name)} 단계 오류: {e}"
            log(msg)
            st.error(msg)
            return False
    return True


if b1.button("① 전사", use_container_width=True):
    safe(partial(step_transcribe, force=True))
if b2.button("② 컷 편집", use_container_width=True):
    safe(partial(step_cut, force=True))
if b3.button("③ 검출", use_container_width=True):
    safe(partial(step_detect, force=True))
if b4.button("④ 모자이크", use_container_width=True):
    safe(step_mosaic)
if b5.button("⑤ 일러스트", use_container_width=True, disabled=not use_llm):
    safe(partial(step_plan_ill, force=True), partial(step_render_ill, force=True))
if b6.button("⑥ 자막·강조", use_container_width=True):
    safe(partial(step_visuals, force=True))
if run_all:
    ok = safe(step_transcribe, step_cut, step_detect, step_mosaic)
    if ok and use_llm:
        if not safe(step_plan_ill, step_render_ill):
            st.warning("일러스트 단계는 실패했지만 나머지 결과는 사용할 수 있습니다.")
    if ok:
        safe(step_visuals)
    if ok:
        final = OUT / f"{src.stem}_edited.mp4"
        shutil.copyfile(current_video(), final)
        log(f"전체 완료 → {final}")
        st.success(f"완료: {final}")

st.divider()
tab_cut, tab_det, tab_ill, tab_vis, tab_out = st.tabs(["✂️ 컷 편집 검수", "🟦 모자이크 검수", "🎨 일러스트 검수", "📝 자막·강조 검수", "📦 결과"])

# ── 컷 편집 검수 ──
with tab_cut:
    plan = rj(cuts_json)
    if plan is None:
        st.caption("② 컷 편집을 먼저 실행하세요.")
    else:
        st.markdown(f"원본 **{plan['source_duration']:.1f}s** → 결과 **{plan['result_duration']:.1f}s** · 제거 {len(plan['removed'])}곳")
        df = pd.DataFrame([{"적용": True, "시작": fmt_time(r["start"]), "끝": fmt_time(r["end"]),
                            "길이": round(r["end"] - r["start"], 2), "사유": r["reason"], "_s": r["start"], "_e": r["end"]}
                           for r in plan["removed"]])
        edited = st.data_editor(df, hide_index=True, use_container_width=True, disabled=["시작", "끝", "길이", "사유"],
                                column_config={"_s": None, "_e": None}, key="cut_editor")
        if st.button("체크 해제한 구간은 살리고 다시 컷 편집"):
            keep_removed = [r for r, ok in zip(plan["removed"], edited["적용"]) if ok]
            from editpipe.cuts import build_cut_list
            new_plan = build_cut_list(plan["source_duration"], [], [{"start": r["start"], "end": r["end"], "reason": r["reason"]} for r in keep_removed], cfg["cut"])
            new_plan["removed"] = [{**r} for r in keep_removed]
            wj(cuts_json, new_plan)
            tr = rj(tr_json)
            wj(work / "transcript_cut.json", remap_transcript(tr, new_plan["keep"]))
            with st.spinner("재인코딩..."):
                apply_cuts(src, new_plan, cut_mp4, cfg)
            for f in (mos_mp4, ill_mp4, vis_mp4, det_json):
                f.unlink(missing_ok=True)
            log("컷 편집 수정 적용")
            st.rerun()
        if cut_mp4.exists():
            st.video(str(cut_mp4))

# ── 모자이크 검수 ──
with tab_det:
    det = rj(det_json)
    if det is None:
        st.caption("③ 검출을 먼저 실행하세요.")
    else:
        st.markdown(f"검출 **{len(det['tracks'])}건** · LLM: {det.get('llm_used') or '미사용'}")
        if det["tracks"]:
            df = pd.DataFrame([{"적용": True, "시작": fmt_time(t["start"]), "끝": fmt_time(t["end"]), "종류": t["type"], "내용": t["label"]}
                               for t in det["tracks"]])
            edited = st.data_editor(df, hide_index=True, use_container_width=True, disabled=["시작", "끝", "종류", "내용"], key="det_editor")
            if st.button("체크 해제한 항목 제외하고 모자이크 적용"):
                det["tracks"] = [t for t, ok in zip(det["tracks"], edited["적용"]) if ok]
                wj(det_json, det)
                if safe(step_mosaic):
                    st.rerun()
        else:
            st.caption("검출된 항목이 없습니다.")
        if mos_mp4.exists():
            st.video(str(mos_mp4))

# ── 일러스트 검수 ──
with tab_ill:
    plan = rj(plan_json)
    if plan is None:
        st.caption("⑤ 일러스트를 먼저 실행하세요. (LLM 키 필요)")
    else:
        st.markdown(f"계획 **{len(plan)}개** · 엔진 **{ill_engine(cfg)}** · object = 투명 배경 오브젝트(화면 한쪽), card = 중앙 카드")
        new_plan = []
        for i, p in enumerate(plan, 1):
            png = work / "illustrations" / f"ill_{i:02d}.png"
            with st.container(border=True):
                a, b = st.columns([1, 2])
                if png.exists():
                    a.image(str(png), use_container_width=True)
                else:
                    a.caption("(아직 생성 안 됨)")
                r1, r2, r3 = b.columns(3)
                start = r1.number_input("시작(초)", 0.0, 10000.0, float(p["start"]), 0.5, key=f"il_s{i}")
                dur = r2.number_input("길이(초)", 1.0, 15.0, float(p["duration"]), 0.5, key=f"il_d{i}")
                kind = r3.selectbox("종류", ["object", "card"], index=0 if p.get("kind") == "object" else 1, key=f"il_k{i}")
                concept = b.text_input("개념", p["concept"], key=f"il_c{i}")
                text = b.text_input("글자(최대 2단어)", p.get("text", ""), key=f"il_t{i}")
                visual = b.text_area("그림 묘사", p["visual"], height=80, key=f"il_v{i}")
                x1, x2 = b.columns(2)
                regen = x1.button("🔄 이 그림 다시 생성", key=f"il_r{i}", disabled=not use_llm)
                drop = x2.checkbox("삭제", key=f"il_x{i}")
                item = {"start": start, "duration": dur, "kind": kind, "concept": concept, "text": text, "visual": visual}
                if regen:
                    with st.spinner("생성 중..."):
                        try:
                            generate_one(item, i, work, cfg)
                            log(f"일러스트 {i} 재생성")
                        except RuntimeError as e:
                            st.error(str(e))
                    st.rerun()
                if not drop:
                    new_plan.append(item)
        y1, y2 = st.columns(2)
        if y1.button("💾 계획 저장"):
            save_plan(new_plan, work)
            log("일러스트 계획 저장")
            st.rerun()
        if y2.button("🎬 저장된 계획으로 오버레이 적용", type="primary"):
            save_plan(new_plan, work)
            if safe(partial(step_render_ill, force=False)):
                st.rerun()
        if ill_mp4.exists():
            st.video(str(ill_mp4))

# ── 자막·강조 검수 ──
with tab_vis:
    vis = rj(vis_json)
    if vis is None:
        st.caption("⑥ 자막·강조를 먼저 실행하세요.")
    else:
        st.markdown(f"자막 **{len(vis.get('subtitles', []))}줄** · 키워드 **{len(vis.get('keywords', []))}개** · 챕터 **{len(vis.get('chapters', []))}개**"
                    + (f" · LLM 오류: {vis['llm_error']}" if vis.get("llm_error") else ""))
        st.caption("챕터 (시작 초, 제목)")
        ch_df = st.data_editor(pd.DataFrame(vis.get("chapters", []) or [{"start": 0.0, "title": ""}]), num_rows="dynamic",
                               hide_index=True, use_container_width=True, key="ch_editor")
        st.caption("키워드 강조 (시작 초, 길이, 핵심어, 보충 설명)")
        kw_df = st.data_editor(pd.DataFrame(vis.get("keywords", []) or [{"start": 0.0, "duration": 4.0, "text": "", "sub": ""}]),
                               num_rows="dynamic", hide_index=True, use_container_width=True, key="kw_editor")
        with st.expander(f"자막 {len(vis.get('subtitles', []))}줄 보기/수정"):
            sub_df = st.data_editor(pd.DataFrame(vis.get("subtitles", [])), num_rows="dynamic", hide_index=True,
                                    use_container_width=True, key="sub_editor")
        if st.button("💾 저장하고 다시 적용", type="primary"):
            vis["chapters"] = [r for r in ch_df.fillna("").to_dict("records") if str(r.get("title", "")).strip()]
            vis["keywords"] = [r for r in kw_df.fillna("").to_dict("records") if str(r.get("text", "")).strip()]
            vis["subtitles"] = [r for r in sub_df.fillna("").to_dict("records") if str(r.get("text", "")).strip()]
            save_visuals(vis, work)
            base = ill_mp4 if ill_mp4.exists() else (mos_mp4 if mos_mp4.exists() else (cut_mp4 if cut_mp4.exists() else src))
            if safe(lambda: apply_visuals(base, vis, work, cfg, vis_mp4)):
                log("시각 요소 수정 적용")
                st.rerun()
        if vis_mp4.exists():
            st.video(str(vis_mp4))

# ── 결과 ──
with tab_out:
    cur = current_video()
    st.markdown(f"현재 최종 단계 파일: `{cur.name}`")
    if cur != src:
        st.video(str(cur))
        with open(cur, "rb") as f:
            st.download_button("📥 최종 영상 다운로드", f.read(), f"{src.stem}_edited.mp4", type="primary")
    else:
        st.caption("아직 처리된 결과가 없습니다.")
    if st.button("🗑 이 영상의 작업 폴더 초기화"):
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True, exist_ok=True)
        log("작업 폴더 초기화")
        st.rerun()
