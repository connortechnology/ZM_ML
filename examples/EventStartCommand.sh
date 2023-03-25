#!/usr/bin/env bash
trap 'cleanup' SIGINT SIGTERM
cleanup() {
  exit 1
}

METHOD="${ML_CLIENT_EVENT_START_MODE:-parallel}"  # consecutive, new, legacy
echo "Starting ML Client in $METHOD mode"
LEGACY_OUT=""
NEW_OUT=""
MID=$2
EID=$1
# NEW
config="${ML_CLIENT_CONF_FILE:-/etc/zm/ml/client.yml}"
detect_script="${ML_CLIENT_EVENT_START:-/home/zmadmin/zm_ml/examples/eventstart.py}"
# LEGACY
ZMES_HOOK_CONFIG_FILE='/etc/zm/objectconfig.yml'
ZMES_DIR="/var/lib/zmeventnotification"

event_start_command=(
  python3
  "${detect_script}"
  --config "${config}"
  --eid "${EID}"
   --mid "${MID}"
   --live
)
legacy_event_start_command=(
  python3.9
  "${ZMES_DIR}/bin/zm_detect.py"
  --event-id "${EID}"
  --monitor-id "${MID}"
  --config "${ZMES_HOOK_CONFIG_FILE}"
  --live
)
run_parallel() {
  echo -e "\n\nRunning ${0} scripts: ${event_start_command[*]} ||| ${legacy_event_start_command[*]}\n\n"
  "${legacy_event_start_command[@]}" &
  "${event_start_command[@]}" &
  wait
}
run_consecutive() {
  echo -e "\n\nRunning ${0} LEGACY script: ${legacy_event_start_command[*]}\n\n"
  LEGACY_OUT=$("${legacy_event_start_command[@]}")

  echo -e "\n\nRunning ${0} NEW script: ${event_start_command[*]}\n\n"
  NEW_OUT=$("${event_start_command[@]}")
}
run_new() {
  echo -e "\n\nRunning ${0} NEW script: ${event_start_command[*]}\n\n"
  NEW_OUT=$("${event_start_command[@]}")
}
run_legacy() {
  echo -e "\n\nRunning ${0} LEGACY script: ${legacy_event_start_command[*]}\n\n"
  LEGACY_OUT=$("${legacy_event_start_command[@]}")
}
if [ "${METHOD}" == "parallel" ]; then
  run_parallel
elif [ "${METHOD}" == "consecutive" ]; then
  run_consecutive
elif [ "${METHOD}" == "new" ]; then
  run_new
elif [ "${METHOD}" == "legacy" ]; then
  run_legacy
else
  echo "Unknown method: ${METHOD}"
  exit 1
fi

exit 0
