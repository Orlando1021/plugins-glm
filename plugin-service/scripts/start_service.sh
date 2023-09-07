#!/usr/bin/env bash

set -euo pipefail
trap "echo 'error: Script failed: see failed command above'" ERR

PROJ_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd ".." && pwd)"

uvicorn \
  --host 0.0.0.0 \
  --port 30016 \
  --log-config "${PROJ_HOME}/configs/logging.yaml" \
  plugin_service:app
