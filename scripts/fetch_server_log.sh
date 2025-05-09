#!/bin/bash

SERVER="paratera"
REMOTE_PATH="/root/projects/permumark/logs/quantization.log"
LOCAL_PATH="logs/quantization.log"
CHECK_INTERVAL=60
FINISHED_MARKER="FINISHED"

echo "Starting to monitor server log..."

while true; do
    scp "$SERVER:$REMOTE_PATH" "$LOCAL_PATH" 2>/dev/null

    if [ $? -ne 0 ]; then
        echo "Error: Unable to connect to server. Assuming server is down."
        exit 1
    fi

    if tail -n 1 "$LOCAL_PATH" | grep -q "$FINISHED_MARKER"; then
        echo "Task finished. Log file downloaded successfully."
        exit 0
    fi

    echo "Task not finished yet. Retrying in $CHECK_INTERVAL seconds..."
    sleep $CHECK_INTERVAL
done

