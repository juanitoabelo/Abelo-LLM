#!/bin/bash
set +e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

cleanup() {
  echo ""
  echo "Shutting down..."
  kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
  wait $BACKEND_PID $FRONTEND_PID 2>/dev/null
  echo "Stopped."
}
trap cleanup EXIT INT TERM

# Check Ollama
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
  echo "  Ollama: running"
else
  echo "  Ollama: NOT running — start it with: ollama serve"
fi

echo ""

# Activate virtual env
if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
fi

echo "Starting backend (FastAPI :8000)..."
python -m src.server.app &
BACKEND_PID=$!

sleep 2

echo "Starting frontend (Next.js :3000)..."
cd "$ROOT_DIR/frontend"
npm run dev &
FRONTEND_PID=$!

cd "$ROOT_DIR"
echo ""
echo "=========================================="
echo "  Backend:  http://localhost:8000"
echo "  Frontend: http://localhost:3000"
echo "  API Docs: http://localhost:8000/docs"
echo "=========================================="
echo ""

wait
