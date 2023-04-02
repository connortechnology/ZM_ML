#!/command/with-contenv bash
# shellcheck shell=bash

program_name="log-config"

echo "[${program_name}] Configuring s6 log rotation with a maximum of ${MAX_LOG_NUMBER} logs and a max log size of ${MAX_LOG_SIZE_BYTES} bytes"
echo -n "1 n${MAX_LOG_NUMBER} s${MAX_LOG_SIZE_BYTES}" > /run/s6/container_environment/S6_LOGGING_SCRIPT
