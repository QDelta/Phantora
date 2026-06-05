#!/usr/bin/env bash
# Phantora preset: Llama2 13B (ZeRO-3 via DeepSpeed).

WORKDIR=$(dirname "$(realpath "$0")")

MODEL_ARGS=(
    --num_layers 40
    --hidden_size 5120
    --ffn_hidden_size 13824
    --num_attention_heads 40
    --vocab_size 32000
    --rope_theta 10000
)

exec "$WORKDIR/../run.sh" \
    "${MODEL_ARGS[@]}" \
    "$@"
