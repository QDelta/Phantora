#!/usr/bin/env bash
# Phantora preset: tiny gpt-oss (real HF MoE architecture).
#
# Real-model coverage of the gpt-oss MoE arch under DeepSpeed DP/ZeRO. Experts run
# locally per rank (HF gpt-oss has no expert all-to-all), so this exercises the
# model + DP/ZeRO collectives, not expert parallelism. Sized small to run on a
# modest box; the real gpt-oss-20b is 24 layers / hidden 2880 / 32 experts / top-4
# (set those via extra args, with --zero_stage 3, on a machine with enough memory).
# Requires transformers >= 4.55.

WORKDIR=$(dirname "$(realpath "$0")")

MODEL_ARGS=(
    --model gpt_oss
    --num_layers 4
    --hidden_size 1024
    --ffn_hidden_size 1024
    --num_attention_heads 16
    --num_key_value_heads 4
    --head_dim 64
    --num_experts 8
    --experts_per_tok 2
    --vocab_size 2048
    --sequence_length 1024
)

exec "$WORKDIR/../run.sh" \
    "${MODEL_ARGS[@]}" \
    "$@"
