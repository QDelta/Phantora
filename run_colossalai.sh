#!/usr/bin/env bash
set -euo pipefail

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$WORKDIR"

docker build -t phantora .
cd tests/docker/ColossalAI
python3 config_gen.py --nhost 4 --ngpu 4 --vram_mib 143771
./run.sh
./stop.sh
cd ../../..
