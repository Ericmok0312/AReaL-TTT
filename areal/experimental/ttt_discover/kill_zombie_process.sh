#!/bin/bash
USER=$(whoami)
echo "Finding processes for user: $USER"

PIDS=$(pgrep -u "$USER" -f "python.*areal|vllm|EngineCore|rpc_server")

if [ -z "$PIDS" ]; then
    echo "No AReaL processes found."
    exit 0
fi

echo "Killing PIDs: $PIDS"
echo "$PIDS" | xargs kill -9

sleep 2
echo "Done. Remaining GPU usage:"
nvidia-smi