#!/usr/bin/env bash
trap 'cleanup' SIGINT SIGTERM
cleanup() {
  exit 1
}
MID=$2
EID=$1

config="${ML_CLIENT_CONF_FILE:-/etc/zm/client.yml}"
detect_script="${ML_CLIENT_EVENT_START:-/usr/local/bin/zmml_eventproc}"

event_start_command=(
  python3
  "${detect_script}"
  --config "${config}"
  --eid "${EID}"
   --mid "${MID}"
   --live
   --event-start
)

eval "${event_start_command[@]}"
exit 0
