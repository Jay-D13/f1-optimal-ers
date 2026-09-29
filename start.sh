#!/bin/bash

# Defaults
BACKEND_PORT=8000
FRONTEND_PORT=5173

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --backend-port) BACKEND_PORT="$2"; shift ;;
        --frontend-port) FRONTEND_PORT="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Kill all background jobs on exit
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

echo "🏎️  Starting F1 ERS Tool..."
echo "   Backend Port: $BACKEND_PORT"
echo "   Frontend Port: $FRONTEND_PORT"

# Start Backend
echo "📡 Starting Backend (FastAPI)..."
uv run uvicorn backend.server:app --reload --reload-dir backend --host 127.0.0.1 --port $BACKEND_PORT &
BACKEND_PID=$!

# Start Frontend
echo "🎨 Starting Frontend (Vite)..."
cd frontend
# Pass port to Vite 
VITE_API_URL="http://localhost:$BACKEND_PORT" npm run dev -- --port $FRONTEND_PORT &
FRONTEND_PID=$!

echo "✅ System Online."
echo "   Backend: http://localhost:$BACKEND_PORT"
echo "   Frontend: http://localhost:$FRONTEND_PORT"
echo "   Press Ctrl+C to stop."

wait
