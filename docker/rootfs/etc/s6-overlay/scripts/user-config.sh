#!/command/with-contenv bash
# shellcheck shell=bash
program_name="configure-user"

PUID=${PUID:-911}
PGID=${PGID:-911}

if [ "${PUID}" -ne 911 ] || [ "${PGID}" -ne 911 ]; then
  echo "[${program_name}] Reconfiguring GID and UID from 911:911 to ${PGID}:${PUID}"
  groupmod -o -g "$PGID" www-data
  usermod -o -u "$PUID" www-data

  echo "[${program_name}] User uid:    $(id -u www-data)"
  echo "[${program_name}] User gid:    $(id -g www-data)"

  echo "[${program_name}] Setting \"/zm_ml\" permissions for user www-data"
  chown -R www-data:www-data \
    /zm_ml
  chmod -R 755 \
    /zm_ml
else
  echo "[${program_name}] Setting \"/zm_ml/conf\" permissions for user www-data"
  chown -R www-data:www-data \
    /zm_ml/conf
  chmod -R 755 \
    /zm_ml/conf
fi

echo "[${program_name}] Setting \"/log\" permissions for user: nobody"
chown -R nobody:nogroup \
  /log
chmod -R 755 \
  /log