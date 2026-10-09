#!/bin/zsh
# 더블클릭 실행 → 브라우저에 웹 앱이 열립니다.
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  python3.11 -m venv .venv && .venv/bin/pip install -q -r requirements.txt
fi
.venv/bin/streamlit run app.py
