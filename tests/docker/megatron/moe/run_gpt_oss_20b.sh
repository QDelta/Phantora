#!/usr/bin/env bash
# Phantora preset: gpt-oss-20b-shaped MoE on Megatron.
#
# Megatron builds from its own GPTModel, not HF, so this is NOT the exact gpt-oss
# architecture (no attention sinks, standard SwiGLU experts rather than gpt-oss's
# clamped gating). It matches the gpt-oss-20b *dimensions* (24 layers, hidden 2880,
# 64 heads / 8 KV heads, head_dim 64, 32 experts, top-4, expert ffn 2880, vocab
# 201088) so the simulated throughput/MFU is a faithful performance proxy for
# gpt-oss-20b -- the cost is dominated by these dims, the GEMMs, and the expert
# all-to-all, which Megatron's MoE simulates.
#
# ~21B total / ~3.6B active. Needs expert (and likely tensor/pipeline) parallelism
# and recompute to fit; pass parallelism via $@, e.g.:
#   python3 config_gen.py --nhost 4 --ngpu 8 --vram_mib 143771
#   ./moe/run_gpt_oss_20b.sh --expert_model_parallel_size 8 --tensor_parallel_size 4 \
#       --pipeline_model_parallel_size 1 --recompute_activations --sequence_length 8192

WORKDIR=$(dirname "$(realpath "$0")")

MODEL_ARGS=(
    --num_layers 24
    --hidden_size 2880
    --num_attention_heads 64
    --num_query_groups 8
    --kv_channels 64
    --ffn_hidden_size 2880
    --num_moe_experts 32
    --moe_router_topk 4
    --moe_token_dispatcher_type alltoall
    --vocab_size 201088
    --swiglu
    --position_embedding_type rope
    --rotary_base 150000
    --normalization RMSNorm
    --disable_bias_linear
)

exec "$WORKDIR/../run.sh" \
    "${MODEL_ARGS[@]}" \
    "$@"
