#!/usr/bin/env bash
trap 'cleanup' SIGINT SIGTERM
cleanup() {
  exit 1
}

MID=$2
EID=$1

[[ -z $ML_CLIENT_CONF_FILE ]] && ML_CLIENT_CONF_FILE='/var/lib/zm_ml/scripts/eventstart.py'

event_start_command=(
  python3
  "${ML_CLIENT_CONF_FILE}"
  --eid "${EID}"
   --mid "${MID}"
   --live
   --start
)

echo -e "\n\nRunning ${0} script: ${event_start_command[*]}\n\n"
es_output=$("${event_start_command[@]}")
echo -e "\n\n${0} output: ${es_output}\n\n"

echo 0
exit 0
