#!/usr/bin/env bash
set -euo pipefail

FLOW_ROOT=/home/ubuntu/yzy/FlowDAgger
OPENPI_ROOT=/home/ubuntu/yzy/openpi
export PYTHONPATH="${OPENPI_ROOT}/src:${OPENPI_ROOT}/packages/openpi-client/src:${FLOW_ROOT}/flowdagger_pi05${PYTHONPATH:+:${PYTHONPATH}}"

exec /home/ubuntu/openpi/.venv/bin/python \
  "${FLOW_ROOT}/flowdagger_pi05/arx_offline_validate.py" "$@"
