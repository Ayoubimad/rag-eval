#!/bin/bash
# This script runs the grid search with YAML config in the background with nohup,
# discarding all output so no nohup.out file is created

cd "$(dirname "$0")"  # Change to the script's directory

# Default config file is grid_search_config.yaml if not specified
CONFIG_FILE=${1:-grid_search_config.yaml}

echo "Starting grid search with configuration file: $CONFIG_FILE"
nohup python run_grid_search_from_yaml.py "$CONFIG_FILE" > /dev/null 2>&1 &

# Print the process ID
echo "Started grid search in background (PID: $!)"
echo "You can check the progress in $(find . -name "*_progress.log" -type f | sort -t_ -k2 | tail -n1)"
echo "To monitor: tail -f $(find . -name "*_progress.log" -type f | sort -t_ -k2 | tail -n1)"
# Note: The script assumes that the Python script `run_grid_search_from_yaml.py` is in the same directory.