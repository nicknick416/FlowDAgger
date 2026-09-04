#!/usr/bin/env bash
set -euo pipefail

exec /home/ubuntu/openpi/.venv/bin/python \
  /home/ubuntu/yzy/FlowDAgger/flowdagger_pi05/arx_evaluate_runs.py "$@"
