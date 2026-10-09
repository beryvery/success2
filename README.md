# 영상 자동 편집 파이프라인

영상 + 대본을 넣으면 아래 순서로 자동 처리합니다.

1. **컷 편집 1차** : 무음 구간 제거 + 필러워드("어", "음" …) 제거
2. **개인정보·상표 모자이크** : 화면 속 전화번호/이메일/주민번호/계좌/차량번호/주소, 이름·브랜드명(Claude), 그래픽 로고(Claude 비전), (옵션) 얼굴
3. **설명 일러스트 삽입** : 대본·발화 내용을 보고 이해를 돕는 지점에 플랫 벡터 + 손그림 카툰 스타일 카드(글자 최소)를 오버레이

이미지 파일을 넣으면 2번(모자이크)만 적용한 이미지를 만듭니다.

## 요구 사항
- macOS (OCR·얼굴검출에 Apple Vision 프레임워크 사용), Python 3.11, ffmpeg (`brew install ffmpeg`)
- LLM API 키: **Claude 또는 GPT(OpenAI) 중 하나 이상** (일러스트 생성, 이름/상표 판별, 로고 검출에 필요. 없으면 정규식·OCR만으로 동작)
  - 프로젝트 폴더의 **`api_keys.txt`** 파일을 열어 키를 넣습니다. (숨김 파일 `.env` 도 같은 역할, 둘 중 아무거나)
    ```
    ANTHROPIC_API_KEY=sk-ant-...     # Claude  (https://console.anthropic.com → API Keys)
    OPENAI_API_KEY=sk-proj-...       # GPT     (https://platform.openai.com/api-keys)
    ```
  - 둘 다 넣었을 때 어느 것을 쓸지는 `config.yaml` 의 `llm.provider` (auto / claude / openai) 로 정합니다. auto 는 Claude 우선.
  - 모델명은 `llm.claude_model`, `llm.openai_model` 에서 변경.

## 실행 (웹 앱)
```bash
./run.command
```
더블클릭하면 브라우저에 http://localhost:8502 가 열립니다.

화면 구성
- **입력**: 영상/이미지 업로드 또는 로컬 파일 경로 입력(큰 파일은 이쪽), 대본 붙여넣기
- **버튼**: ① 전사 → ② 컷 편집 → ③ 검출 → ④ 모자이크 → ⑤ 일러스트, 또는 **▶ 전체 실행**
- **검수 탭**
  - ✂️ 컷 편집 검수: 제거 구간 목록에서 체크 해제 → "다시 컷 편집" 으로 살리기
  - 🟦 모자이크 검수: 오검출 항목 체크 해제 → 제외하고 재적용
  - 🎨 일러스트 검수: 시작/길이/종류(object·card)/묘사 수정, 개별 재생성, 삭제 후 오버레이 재적용
  - 📦 결과: 최종 영상 재생·다운로드
- **사이드바**: LLM 상태, 무음·필러 기준, 얼굴 모자이크, 블록 크기, 오브젝트 위치/크기, 카드 크기

터미널(CLI)로도 같은 처리를 할 수 있습니다.
```bash
.venv/bin/python pipeline.py 영상.mp4 --script 대본.txt --out out
.venv/bin/python pipeline.py 사진.jpg                 # 이미지 모자이크
.venv/bin/python pipeline.py 영상.mp4 --dry-run       # 계획만
.venv/bin/python pipeline.py 영상.mp4 --faces         # 얼굴도 모자이크
```

## 일러스트 종류
| kind | 용도 | 표시 방식 |
|---|---|---|
| `object` | 아이콘/사물/심볼 하나로 설명될 때 | **투명 배경**으로 화면 우측(또는 좌측) 중앙에 얹음 |
| `card` | 관계·흐름·비교 등 장면이 필요할 때 | 흰 바탕 둥근 **중앙 카드** |

그림체는 깔끔한 플랫 벡터에 손그림 느낌을 살짝만 얹은 톤(파스텔 3~4색, 일정한 외곽선, 글자 최대 2단어)이며 `config.yaml` 의 `illustrate.style` 에서 바꿀 수 있습니다.

## 결과물 (`out/<영상이름>/`)
| 파일 | 내용 |
|---|---|
| `transcript.json` | Whisper 단어 단위 전사 |
| `cuts.json`, `cuts_review.csv` | 제거 구간 목록(사유 포함) 과 유지 구간 |
| `1_cut.mp4` | 컷 편집 결과 |
| `detections.json` | 모자이크 대상 (시간, 위치, 종류, 사유) |
| `2_mosaic.mp4` | 모자이크 적용 결과 |
| `illustrations/plan.json`, `ill_XX.svg/png` | 일러스트 계획과 그림 |
| `3_illustrated.mp4` | 일러스트 삽입 결과 |
| `report.md` | 전체 요약 |
| `out/<영상이름>_edited.mp4` | **최종본** |

## 검수 후 재실행 (중간 결과 재사용)
각 단계 결과는 JSON으로 남고, 파일이 있으면 다시 계산하지 않습니다.
- 잘못 잘린 구간 복구 → `cuts.json` 의 `keep` 수정 후 `1_cut.mp4` 삭제하고 재실행
- 오검출 모자이크 제거 → `detections.json` 의 `tracks` 에서 항목 삭제 후 `2_mosaic.mp4` 삭제하고 재실행
- 일러스트 위치/내용 변경 → `illustrations/plan.json` 수정 후 해당 `ill_XX.png` 삭제하고 재실행
- 전부 다시 → `--force`

## 설정 (`config.yaml`)
- `whisper.model` : tiny/base/small/medium/large-v3 (기본 small, 정확도 필요하면 medium)
- `cut.*` : 무음 기준 dB, 최소 무음 길이, 필러워드 목록
- `detect.*` : 검사 간격, 얼굴 모자이크 여부, 정규식, 브랜드 단어 목록, Claude 사용 여부
- `mosaic.block` : 모자이크 블록 크기
- `illustrate.*` : 삽입 개수/간격/길이, pip(우측 카드)·full 레이아웃, 스타일 지시문

## 동작 방식 요약
- 무음: ffmpeg `silencedetect` → 기준 이상 무음을 앞뒤 여백만 남기고 제거
- 필러: Whisper 단어 타임스탬프 중 필러 목록과 일치하는 단어 구간 제거 ("그", "아" 처럼 실제 단어와 겹치는 것은 기본 제외)
- 모자이크: 0.5초 간격 프레임 → Apple Vision OCR → 정규식/브랜드 사전 → (Claude) 이름·주소·상표 판별, 장면 전환마다 로고 비전 검출 → 시간 연속 트랙으로 묶어 픽셀화
- 일러스트: Claude가 대본+타임라인에서 삽입 지점 선정 → SVG 생성(손그림 필터 포함) → PNG 렌더 → 페이드 인/아웃 오버레이

## 테스트
```bash
.venv/bin/python -m pytest tests
.venv/bin/python pipeline.py samples/test_video.mp4 --script samples/script.txt --out out --no-claude
```
`samples/test_video.mp4` 는 한국어 TTS(필러·무음 포함)와 화면 속 전화번호/이메일이 들어간 합성 테스트 영상입니다.

## 한계
- 로고 검출 좌표는 Claude 비전의 추정값이라 여백을 넉넉히 잡습니다. 정밀 추적이 필요하면 `detections.json` 을 손봐 주세요.
- 이름은 OCR 텍스트 기준으로만 판별합니다(음성 속 이름은 처리 안 함).
- 음악·효과음만 있는 구간은 "무음"이 아니므로 유지됩니다.
