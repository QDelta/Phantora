#!/usr/bin/env bash
# Phantora preset: Llama2 70B (ZeRO-3 via DeepSpeed).
#
# Note: 70B uses GQA (8 KV groups / 64 Q heads).

WORKDIR=$(dirname "$(realpath "$0")")

MODEL_ARGS=(
    --num_layers 80
    --hidden_size 8192
    --ffn_hidden_size 28672
    --num_attention_heads 64
    --num_key_value_heads 8
    --vocab_size 32000
    --rope_theta 10000
)

exec "$WORKDIR/../run.sh" \
    "${MODEL_ARGS[@]}" \
    "$@"
