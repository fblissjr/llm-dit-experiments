#!/bin/bash
#
# Stop the Z-Image web server.
#
# Usage:
#   ./scripts/stop-server.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="${PROJECT_DIR}/logs"
PID_FILE="${LOG_DIR}/server.pid"

if [ ! -f "$PID_FILE" ]; then
    echo "No server PID file found. Server may not be running."
    exit 0
fi

PID=$(cat "$PID_FILE")

if kill -0 "$PID" 2>/dev/null; then
    echo "Stopping server (PID $PID)..."
    kill "$PID"

    # Wait for graceful shutdown
    for i in {1..10}; do
        if ! kill -0 "$PID" 2>/dev/null; then
            break
        fi
        sleep 0.5
    done

    # Force kill if still running
    if kill -0 "$PID" 2>/dev/null; then
        echo "Force killing..."
        kill -9 "$PID" 2>/dev/null || true
    fi

    rm -f "$PID_FILE"
    echo "Server stopped."
else
    echo "Server not running (stale PID file removed)."
    rm -f "$PID_FILE"
fi
