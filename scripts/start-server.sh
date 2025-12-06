#!/bin/bash
#
# Start Z-Image web server in the background.
# The server will continue running after SSH disconnect.
#
# Usage:
#   ./scripts/start-server.sh [args...]
#
# Examples:
#   # Start with default settings
#   ./scripts/start-server.sh --model-path /path/to/z-image
#
#   # Start with config file
#   ./scripts/start-server.sh --config config.toml
#
#   # Stop the server
#   ./scripts/stop-server.sh
#
# Log output: logs/server.log
# PID file: logs/server.pid

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="${PROJECT_DIR}/logs"
LOG_FILE="${LOG_DIR}/server.log"
PID_FILE="${LOG_DIR}/server.pid"

# Create log directory
mkdir -p "$LOG_DIR"

# Check if already running
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "Server already running with PID $OLD_PID"
        echo "Use ./scripts/stop-server.sh to stop it first"
        exit 1
    else
        rm -f "$PID_FILE"
    fi
fi

echo "Starting Z-Image web server..."
echo "  Log file: $LOG_FILE"
echo "  PID file: $PID_FILE"

# Start server with nohup
cd "$PROJECT_DIR"
nohup uv run web/server.py "$@" > "$LOG_FILE" 2>&1 &
SERVER_PID=$!

# Save PID
echo "$SERVER_PID" > "$PID_FILE"

# Wait briefly and check if started
sleep 2
if kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Server started with PID $SERVER_PID"
    echo ""
    echo "To view logs: tail -f $LOG_FILE"
    echo "To stop: ./scripts/stop-server.sh"
else
    echo "Failed to start server. Check logs:"
    tail -20 "$LOG_FILE"
    exit 1
fi
