#!/usr/bin/env bash
# Phantora preset: gpt-oss-20b on DeepSpeed (real HF model).
#
# Builds the real openai/gpt-oss-20b architecture (transformers GptOssForCausalLM)
# at its actual dimensions: 24 layers, hidden 2880, 64 heads / 8 KV heads,
# head_dim 64, 32 experts, top-4, expert ffn 2880, vocab 201088. gpt-oss's experts
# run *locally* per rank (HF gpt-oss has no expert all-to-all), so this exercises
# the model + DeepSpeed DP/ZeRO collectives, not expert parallelism. Requires
# transformers >= 4.55.
#
# ~21B total / ~3.6B active. Needs ZeRO-3 (and a machine with enough memory) to
# fit; for a quick check, scale down with --num_layers / --hidden_size overrides.
#   ./gpt_oss/run_gpt_oss_20b.sh --zero_stage 3 --sequence_length 4096

WORKDIR=$(dirname "$(realpath "$0")")

MODEL_ARGS=(
    --model gpt_oss
    --num_layers 24
    --hidden_size 2880
    --ffn_hidden_size 2880
    --num_attention_heads 64
    --num_key_value_heads 8
    --head_dim 64
    --num_experts 32
    --experts_per_tok 4
    --vocab_size 201088
)

exec "$WORKDIR/../run.sh" \
    "${MODEL_ARGS[@]}" \
    "$@"
