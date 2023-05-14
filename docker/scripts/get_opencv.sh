#!/usr/bin/env bash
# Build from source tarball or git repository
#$ Args
# 1: Method: release, branch, sha
# 2: Version: 3.4.0, 3.4.0-dev, 3.4.0-rc1, master, 3.4, 3.4.0-rc1-1-g5d6c9a3
# 3: Git URL: https://github.com/<owner>/<repo>  # NOTICE NO TRAILING SLASH, also do not append .git
# 4: Destination directory

grab_source() {
  echo "About to call grab_source with args: $@"
  local _METHOD=${1:-<unset>}
  local _VERSION=${2}
  local GIT_URL=${3}
  local DEST_DIR=${4}
  if [[ "$_METHOD" == "release" || "$_METHOD" == "branch" || "$_METHOD" == "sha" ]]; then
    echo "\$_METHOD is $_METHOD | it is one of release, branch or sha"
    if [[ "$_METHOD" == "release" ]]; then
      local _start=$(pwd)
      local TMP_DIR=$(mktemp -d)
      echo "\"release method\" -> Created temp directory: ${TMP_DIR}"
      cd "$TMP_DIR" || exit 1
      wget -q -O temp.tgz "${GIT_URL}/archive/refs/tags/${_VERSION}.tar.gz"
      tar -xzf temp.tgz
      rm temp.tgz
      mv ./* "$DEST_DIR"
      cd "$_start" || exit 1
      rm -rf "$TMP_DIR"
      else
        git clone --branch "${_VERSION}" "${GIT_URL}.git" "$DEST_DIR"
    fi
  else
    echo "Unknown method: ${_METHOD}. Allowed: release, branch, sha"
    exit 55
  fi
}