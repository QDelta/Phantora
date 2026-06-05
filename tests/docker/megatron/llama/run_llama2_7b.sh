#!/usr/bin/env bash
# Phantora preset: Llama2 7B.
#
# Mirrors upstream Megatron-LM's examples/llama/ shell scripts. Numbers are
# the published Llama2 7B config; extra args are forwarded to ../run.sh.

WORKDIR=$(dirname "$(realpath "$0")")

MODEL_ARGS=(
    --num_layers 32
    --hidden_size 4096
    --ffn_hidden_size 11008
    --num_attention_heads 32
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
