#!/usr/bin/env bash
# Phantora preset: Qwen3 30B-A3B on Megatron.
#
# The real Qwen3-MoE architecture: GQA + RoPE + QK-norm (RMSNorm on Q/K) + top-8-of-128
# SwiGLU experts, no shared experts. Megatron-core's GPTModel expresses all of this
# (qk_layernorm + kv_channels for the head_dim != hidden/heads case). Dimensions match
# Qwen/Qwen3-30B-A3B: hidden 2048, 48 layers, 32 heads / 4 KV heads, head_dim 128,
# 128 experts top-8, expert ffn (moe_inter_dim) 768, vocab 151936.
#
# ~30.5B total / ~3.3B active. Needs expert (and typically tensor/pipeline)
# parallelism plus recompute to fit; pass parallelism via $@, e.g.:
#   python3 config_gen.py --nhost 8 --ngpu 8 --vram_mib 81920
#   ./qwen3/run_qwen3_30b_a3b.sh --expert_model_parallel_size 8 --tensor_parallel_size 4 \
#       --pipeline_model_parallel_size 2 --recompute_activations --sequence_length 4096

WORKDIR=$(dirname "$(realpath "$0")")

MODEL_ARGS=(
    --num_layers 48
    --hidden_size 2048
    --ffn_hidden_size 768
    --num_attention_heads 32
    --num_query_groups 4
    --kv_channels 128
    --num_moe_experts 128
    --moe_router_topk 8
    --moe_token_dispatcher_type alltoall
    --qk_layernorm
    --vocab_size 151936
    --swiglu
    --position_embedding_type rope
    --rotary_base 1000000
    --normalization RMSNorm
    --disable_bias_linear
)

exec "$WORKDIR/../run.sh" \
    "${MODEL_ARGS[@]}" \
    "$@"
