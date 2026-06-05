#!/usr/bin/env bash
# Phantora preset: Llama2 70B.
#
# Note: 70B uses GQA (8 KV groups / 64 Q heads), unlike the 7B/13B variants.

WORKDIR=$(dirname "$(realpath "$0")")

MODEL_ARGS=(
    --num_layers 80
    --hidden_size 8192
    --ffn_hidden_size 28672
    --num_attention_heads 64
    --num_query_groups 8
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
