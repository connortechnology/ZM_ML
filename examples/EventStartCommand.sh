#!/usr/bin/env bash
trap 'cleanup' SIGINT SIGTERM
cleanup() {
  exit 1
}

MID=$2
EID=$1
config="${ML_CLIENT_CONF_FILE:-/etc/zm/ml/client.yml}"
detect_script="${ML_CLIENT_EVENT_START:-/home/zmadmin/zm_ml/examples/eventstart.py}"

event_start_command=(
  python3
  "${detect_script}"
  --config "${config}"
  --eid "${EID}"
   --mid "${MID}"
   --live
)

echo -e "\n\nRunning ${0} script: ${event_start_command[*]}\n\n"
es_output=$("${event_start_command[@]}")
echo -e "\n\n${0} output: ${es_output}\n\n"

echo 0
exit 0
