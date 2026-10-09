# agent.md — AI 코딩 에이전트용 안내

이 문서는 Claude Code, Codex, Cursor 등 AI 에이전트가 이 저장소에서 작업할 때 따라야 할 규칙입니다. 상세 구조는 `CLAUDE.md`, 사용법은 `README.md` 를 먼저 읽으세요.

## 목적
영상 제작자가 로컬에서 버튼 몇 번으로 **컷 편집 → 개인정보/상표 모자이크 → 설명 일러스트 삽입**을 끝내는 도구. 사용자는 비개발자이므로 결과물은 항상 "실행 가능한 상태"여야 합니다.

## 작업 순서
1. 변경 전 `.venv/bin/python -m pytest -q tests` 로 현재 상태 확인.
2. 코드 수정 → 단위 테스트 추가/수정 → `samples/test_video.mp4` 로 통합 실행(`--no-claude`) 해서 끝까지 도는지 확인.
3. UI 를 바꿨으면 `streamlit run app.py` 로 실제 화면을 열어 버튼 동작을 확인.
4. 설정값을 추가했으면 `config.yaml`(주석 포함) 과 `README.md` 를 함께 갱신.

## 절대 하지 말 것
- `api_keys.txt`, `.env` 의 내용을 읽어 출력하거나 커밋하지 않는다.
- 사용자 영상/작업 폴더(`out/`, `uploads/`)를 커밋하지 않는다.
- 외부 서비스에 영상 프레임을 보내는 코드를 추가할 때는 어떤 데이터가 전송되는지 README 에 명시한다.
- 기존 산출물 JSON 스키마(`cuts.json`, `detections.json`, `plan.json`)를 깨뜨리지 않는다. 필드 추가는 가능, 삭제·이름 변경은 금지.

## 자주 쓰는 명령
```bash
./run.command                                   # 웹 앱
.venv/bin/python pipeline.py 영상.mp4 --script 대본.txt
.venv/bin/python -m pytest -q tests
ffprobe -v error -show_entries stream=codec_type,duration -of csv=p=0 out/<영상>/1_cut.mp4
```

## 환경 특이사항
- macOS 전용(Apple Vision). Linux 지원이 필요하면 `detect.py` 의 OCR/얼굴 함수를 교체하고 나머지는 유지.
- ffmpeg 9: `-/filter_complex 파일` 문법, `drawtext` 없음.
- Python 3.11 고정(3.14 는 일부 휠 미지원).
