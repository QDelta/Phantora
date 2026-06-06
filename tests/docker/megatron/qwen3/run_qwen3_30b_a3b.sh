#!/usr/bin/env bash
# Phantora preset: Qwen3 30B-A3B on Megatron.
#
# The real Qwen3-MoE architecture: GQA + RoPE + QK-norm (RMSNorm on Q/K) + top-8-of-128
# SwiGLU experts, no shared experts. Megatron-core's GPTModel expresses all of this
# (qk_layernorm + kv_channels for the head_dim != hidden/heads case). Dimensions match
# Qwen/Qwen3-30B-A3B: hidden 2048, 48 layers, 32 heads / 4 KV heads, head_dim 128,
# 128 experts top-8, expert ffn (moe_inter_dim) 768, vocab 151936.
#
# ~30.5B total / ~3.3B active. The 128 experts are sharded by expert parallelism;
# tensor parallelism defaults to 1 here (override with --tensor_parallel_size).
# Needs enough EP (and/or a larger --vram_mib) plus recompute to fit. On an 8-GPU
# node (world = nhost*ngpu must equal TP*EP*PP):
#   python3 config_gen.py --nhost 1 --ngpu 8 --vram_mib 81920
#   ./qwen3/run_qwen3_30b_a3b.sh --expert_model_parallel_size 8 --recompute_activations
# For larger clusters add --tensor_parallel_size / --pipeline_model_parallel_size
# and size config_gen so nhost*ngpu == TP*EP*PP.

WORKDIR=$(dirname "$(realpath "$0")")

MODEL_ARGS=(
    # TP=1 by default (test_megatron.py's default is 4, which would force
    # world == 4*EP); override via --tensor_parallel_size.
    --tensor_parallel_size 1
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
