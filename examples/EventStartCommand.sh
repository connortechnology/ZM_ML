#!/usr/bin/env bash
trap 'cleanup' SIGINT SIGTERM
cleanup() {
  exit 1
}

MID=$2
EID=$1

[[ -z $ML_CONFIG_FILE ]] && ML_CONFIG_FILE='/var/lib/zm_ml/scripts/eventstart.py'

event_start_command=(
  python3
  "${ML_CONFIG_FILE}"
  --eid "${EID}"
   --mid "${MID}"
   --live
   --start
)

echo -e "\n\nRunning eventstart.py script: ${event_start_command[*]}\n\n"
es_output=$("${event_start_command[@]}")
echo -e "\n\neventstart.py output: ${es_output}\n\n"

echo 0
exit 0
