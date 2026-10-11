# CLAUDE.md — 영상 자동 편집 파이프라인

이 저장소는 영상(또는 이미지) + 대본을 입력받아 **① 컷 편집(무음·필러 제거) → ② 개인정보·상표 모자이크 → ③ 설명 일러스트 삽입**을 자동으로 수행하는 로컬 도구입니다. macOS 전용(Apple Vision OCR 사용).

## 빠른 명령
```bash
.venv/bin/streamlit run app.py                 # 웹 앱 (포트 8502) — 또는 ./run.command
.venv/bin/python pipeline.py 영상.mp4 --script 대본.txt --out out   # CLI 전체 실행
.venv/bin/python pipeline.py 영상.mp4 --no-claude --dry-run         # LLM 없이 계획만
.venv/bin/python -m pytest -q tests            # 단위 테스트
```
환경: Python 3.11 venv(`.venv`), `brew install ffmpeg` 필수. 패키지는 `requirements.txt`.

## 구조
| 경로 | 역할 |
|---|---|
| `app.py` | Streamlit 웹 UI. 단계별 버튼 + 검수 탭. 단계 함수 `step_*` 를 `safe()` 로 감싸 오류를 화면에 표시 |
| `pipeline.py` | CLI. 같은 단계를 순서대로 실행하고 `report.md` 작성 |
| `editpipe/media.py` | ffprobe/ffmpeg 헬퍼 (`probe`, `extract_audio`, `extract_frames`) |
| `editpipe/transcribe.py` | faster-whisper 단어 타임스탬프, 컷 후 타임라인 재매핑 `remap_transcript` |
| `editpipe/cuts.py` | 무음(ffmpeg silencedetect) + 필러 검출 → `cuts.json` → trim/atrim+concat 으로 재인코딩 |
| `editpipe/detect.py` | Vision OCR(Linux 는 tesseract 폴백)·얼굴 검출, 정규식/브랜드 사전, LLM 텍스트 판별·로고 비전 검출, 트랙 병합 |
| `editpipe/mosaic.py` | ffmpeg rawvideo 파이프 ↔ numpy 픽셀화 (OpenCV 미사용) |
| `editpipe/illustrate.py` | LLM 계획 → 그림 생성(`gpt-image`: OpenAI images API 투명 PNG / `svg`: LLM SVG → resvg) → compose 오버레이. `object`(투명, 화면 한쪽) / `card`(중앙 카드). ThreadPool 병렬 |
| `editpipe/compose.py` | 공통 PNG 오버레이 합성기 (rawvideo 파이프 + numpy 알파 블렌딩, 페이드). 모든 오버레이는 이걸 통해서만 |
| `editpipe/visuals.py` | 자막(전사 기반, PII 마스킹)·키워드 lower-third·챕터 칩을 PIL 로 렌더 후 compose |
| `editpipe/llm_client.py` | Claude(Anthropic SDK)·GPT(OpenAI SDK) 공통 호출. 키는 `api_keys.txt` → `.env` → 환경변수 순 |
| `editpipe/drive.py` | 구글 드라이브 업로드(서비스 계정, 키는 환경변수 `GDRIVE_SERVICE_ACCOUNT_JSON`). 개인 드라이브는 같은 이름 파일 내용 교체 방식 |
| `config.yaml` | 모든 기본값(무음 기준, 필러 목록, 정규식, 모자이크 강도, 일러스트 스타일, 모델명) |

## 작업 산출물 규칙
- 영상별 작업 폴더 `out/<영상이름>/` 에 `transcript.json → cuts.json → 1_cut.mp4 → detections.json → 2_mosaic.mp4 → illustrations/plan.json, ill_XX.png → 3_illustrated.mp4 → visuals.json → 4_visuals.mp4` 순서로 생성.
- 각 단계는 산출물이 있으면 재계산하지 않음(`force` 로 강제). 앞 단계를 다시 만들면 뒤 단계 파일은 삭제해 무효화한다.
- 타임스탬프는 **컷 편집 이후 타임라인** 기준(`transcript_cut.json`). 검출·일러스트는 모두 `1_cut.mp4` 기준으로 계산.

## 코드 작성 시 지킬 것
- ffmpeg 9 기준: 필터 그래프가 길면 `-/filter_complex <파일>` 사용 (`-filter_complex_script` 는 제거됨). 옵션은 `media.filter_script_args()` 로 만들면 ffmpeg 6 이하에서 자동으로 구형 옵션을 쓴다. `drawtext`·`subtitles`(libass) 는 이 빌드에 없음 → 텍스트/자막은 PIL 로 PNG 렌더 후 `compose.py` 로 합성.
- `faster-whisper`(av) 와 OpenCV 를 한 프로세스에 함께 import 하면 libavdevice 충돌 → OpenCV 를 쓰지 않는다.
- LLM 호출은 반드시 `llm_client.ask / ask_json` 을 통해서만. Claude 는 `client.beta.messages.create(... fallbacks="default", betas=["server-side-fallback-2026-07-01"])`, 응답의 `stop_reason` 확인. 모델명은 `config.yaml llm.*` 에서만 읽는다.
- LLM 이 반환한 SVG 는 `_sanitize_svg` 로 `<script>`/`<image>` 제거 + XML 검증 후 렌더.
- 비밀 키 파일(`api_keys.txt`, `.env`)은 절대 커밋·출력하지 않는다. `.env.example` 만 커밋.
- 한국어 UI/메시지 유지. 새 설정값은 `config.yaml` 에 주석과 함께 추가하고 README 에 반영.
- `editpipe/` 수정 후 Streamlit 은 재시작해야 반영된다.

## 테스트
- `tests/test_units.py` : 컷 목록, 타임라인 재매핑, 정규식, 트랙 병합, 픽셀화, SVG 렌더.
- `tests/test_visuals.py` : 자막 줄 나누기, PII 마스킹, 렌더 투명도, compose 합성, visuals 적용.
- 통합 확인: `samples/test_video.mp4`(한국어 TTS + 화면 속 전화번호/이메일) 로 `pipeline.py ... --no-claude` 실행 후 `out/test_video/report.md` 확인.
- 새 테스트 영상이 필요하면 `say -v Yuna` 로 음성을 만들고 PNG 오버레이로 텍스트를 넣는다(README 참고).
