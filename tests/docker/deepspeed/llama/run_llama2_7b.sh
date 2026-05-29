#!/usr/bin/env bash
# Phantora preset: Llama2 7B (ZeRO-3 via DeepSpeed).

WORKDIR=$(dirname "$(realpath "$0")")

MODEL_ARGS=(
    --num_layers 32
    --hidden_size 4096
    --ffn_hidden_size 11008
    --num_attention_heads 32
    --vocab_size 32000
    --rope_theta 10000
)

exec "$WORKDIR/../run.sh" \
    "${MODEL_ARGS[@]}" \
    "$@"
