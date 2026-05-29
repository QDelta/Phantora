#!/usr/bin/env bash
# Phantora preset: Llama3 8B.
#
# Differences from Llama2: GQA across all sizes, larger vocab (128K),
# and a much larger RoPE base (500K).

WORKDIR=$(dirname "$(realpath "$0")")

MODEL_ARGS=(
    --num_layers 32
    --hidden_size 4096
    --ffn_hidden_size 14336
    --num_attention_heads 32
    --num_query_groups 8
    --vocab_size 128256
    --swiglu
    --position_embedding_type rope
    --rotary_base 500000
    --normalization RMSNorm
    --disable_bias_linear
)

exec "$WORKDIR/../run.sh" \
    "${MODEL_ARGS[@]}" \
    "$@"
