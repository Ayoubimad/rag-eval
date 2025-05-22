#!/bin/bash
# This script runs the grid search in the background with nohup,
# discarding all output so no nohup.out file is created

cd "$(dirname "$0")"  # Change to the script's directory
nohup python run_grid_search.py > /dev/null 2>&1 &

# Print the process ID
echo "Started grid search in background (PID: $!)"
echo "You can check the progress in $(find . -name "*_progress.log" -type f | sort -t_ -k2 | tail -n1)"
echo "To monitor: tail -f $(find . -name "*_progress.log" -type f | sort -t_ -k2 | tail -n1)"
