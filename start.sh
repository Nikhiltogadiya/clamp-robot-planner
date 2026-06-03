#!/usr/bin/env bash
# CLAMP Console launcher. Run:  ./start.sh   (or: bash start.sh)
set -u
cd "$(dirname "$0")"
PY=.venv/bin/python

if [ ! -x "$PY" ]; then
  echo "!! No virtualenv found. First run:"
  echo "     uv venv -p 3.11 .venv && uv pip install -r requirements.txt"
  exit 1
fi

have_key() {  # true if a live OpenRouter key is set
  [ -n "${OPENROUTER_API_KEY:-}" ]
}

cat <<'MENU'

============================================================
  CLAMP Console - closed-loop robot planner in your browser
============================================================
   1) Open the operator console  (browser, live 3D, LLM+VLM)
   2) Run the test suite         (offline, no API key)
   0) Quit
============================================================
MENU
printf "Choose: "
read -r choice
case "$choice" in
  1) if have_key; then echo ">> Live brain ON - real LLM plans, real VLM verifies.";
     else echo ">> No OPENROUTER_API_KEY - starting in MOCK mode (still fully usable)."; fi
     echo ">> Opening http://localhost:8080 - type an instruction, click Run, then try Sabotage."
     $PY -m console.app ;;
  2) $PY -m pytest -q ;;
  0|q|Q) echo "bye." ;;
  *) echo "?? unknown choice: $choice" ;;
esac
