#!/usr/bin/env bash
# Phantora preset: Llama3 8B (TorchTitan + FSDP2).
#
# TorchTitan reads its config from a .toml file; this script points run.sh at
# the canonical Llama3 8B config that ships with Phantora. Extra args are
# forwarded — useful for TorchTitan-style overrides like
# `--parallelism.tensor_parallel_degree=2`.

WORKDIR=$(dirname "$(realpath "$0")")

exec "$WORKDIR/../run.sh" \
    --job.config_file=/phantora/tests/test_torchtitan_llama3_8b.toml \
    "$@"
