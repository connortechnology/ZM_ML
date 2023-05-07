#!/command/with-contenv bash
# shellcheck shell=bash
program_name="mlapi-config"

if [[ ! -f /zm_ml/conf/server.yml ]]; then
  echo "[${program_name}] Creating default ZM ML Server CONFIGURATION file"
  eval "python3.9 /opt/zm_ml/src/examples/install.py --dir-config /zm_ml/conf --dir-data /zm_ml/data --dir-log /zm_ml/logs --config-only --install-type server --debug --user www-data --group www-data"
fi
if [[ ! -f /zm_ml/conf/secrets.yml ]]; then
  echo "[${program_name}] Creating default ZM ML Server SECRETS file"
  eval "python3.9 /opt/zm_ml/src/examples/install.py --dir-config /zm_ml/conf --dir-data /zm_ml/data --dir-log /zm_ml/logs --secrets-only --install-type server --debug --user www-data --group www-data"
fi
