#!/usr/bin/with-contenv bash
# shellcheck shell=bash
program_name="mlapi-config"

create(){
  # $1 = 'file' or 'dir'
  # $2 = 'SOURCE'
  # $3 = 'DESTINATION' - checks if the DESTINATION exists, if not it copies SOURCE to DESTINATION

  if [ "${1}" == 'file' ]; then
    if [ ! -f "${3}" ]; then
      echo "[${program_name}] Creating File: ${3} using: ${2}"
      s6-setuidgid www-data cp "${2}" "${3}"
    fi
  elif [ "${1}" == 'dir' ]; then
    if [ ! -d "${3}" ]; then
      echo "[${program_name}] Creating Directory: ${3} using: ${2}"
      s6-setuidgid www-data cp -r "${2}" "${3}"
    fi
  else
    echo "[${program_name}] create(${1}): Unknown type"
  fi
}

[[ ! -f /config/server.yml ]] && echo "[${program_name}] Creating default ZM ML Server configuration file"
create 'file' '/opt/zm_ml/src/configs/example_server.yml' '/zm_ml/conf/server.yml'
create 'file' '/opt/zm_ml/src/example_secrets.yml' '/zm_ml/conf/secrets.yml'
# Directory structure
chown -R www-data:www-data /zm_ml