#!/usr/bin/env bash
# Phantora preset: Llama2 13B.

WORKDIR=$(dirname "$(realpath "$0")")

MODEL_ARGS=(
    --num_layers 40
    --hidden_size 5120
    --ffn_hidden_size 13824
    --num_attention_heads 40
    --vocab_size 32000
    --swiglu
    --position_embedding_type rope
    --rotary_base 10000
    --normalization RMSNorm
    --disable_bias_linear
)

exec "$WORKDIR/../run.sh" \
    "${MODEL_ARGS[@]}" \
    "$@"
