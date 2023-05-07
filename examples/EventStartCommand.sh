#!/usr/bin/env bash
trap 'cleanup' SIGINT SIGTERM
cleanup() {
  exit 1
}
MID=$2
EID=$1

config="${ML_CLIENT_CONF_FILE:-/etc/zm/ml/client.yml}"
detect_script="${ML_CLIENT_EVENT_START:-zmml-eventstart}"

event_start_command=(
  python3
  "${detect_script}"
  --config "${config}"
  --eid "${EID}"
   --mid "${MID}"
   --live
)

eval "${event_start_command[@]}"
exit 0
