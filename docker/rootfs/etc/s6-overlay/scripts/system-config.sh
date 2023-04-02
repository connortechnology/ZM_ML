#!/command/with-contenv bash
# shellcheck shell=bash
program_name="system-config"

## Configure Timezone
echo "[${program_name}] Setting system timezone to ${TZ}"
ln -sf "/usr/share/zoneinfo/$TZ" /etc/localtime