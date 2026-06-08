#!/usr/bin/env bash
# Phantora preset: Mixtral 8x7B on Megatron.
#
# Unlike the gpt-oss preset (a dimension-only proxy), this is the *actual*
# Mixtral architecture: Megatron-core's MoE GPTModel is exactly GQA attention +
# RoPE + SwiGLU experts with top-2 routing over 8 experts, which is what Mixtral
# is. Dimensions match mistralai/Mixtral-8x7B (hidden 4096, ffn 14336, 32 layers,
# 32 heads / 8 KV heads, head_dim 128, 8 experts, top-2, vocab 32000).
#
# ~46.7B total / ~12.9B active. The 8 experts are sharded by expert parallelism;
# tensor parallelism defaults to 1 here (override with --tensor_parallel_size).
# Needs enough EP (and/or a larger --vram_mib) plus recompute to fit. On an 8-GPU
# node (world = nhost*ngpu must equal TP*EP*PP):
#   python3 config_gen.py --nhost 1 --ngpu 8 --vram_mib 81920
#   ./mixtral/run_mixtral_8x7b.sh --expert_model_parallel_size 8 --recompute_activations
# For larger clusters add --tensor_parallel_size / --pipeline_model_parallel_size
# and size config_gen so nhost*ngpu == TP*EP*PP.

WORKDIR=$(dirname "$(realpath "$0")")

MODEL_ARGS=(
    # TP=1 by default (test_megatron.py's default is 4, which would force
    # world == 4*EP); override via --tensor_parallel_size.
    --tensor_parallel_size 1
    --num_layers 32
    --hidden_size 4096
    --ffn_hidden_size 14336
    --num_attention_heads 32
    --num_query_groups 8
    --vocab_size 32000
    --swiglu
    --position_embedding_type rope
    --rotary_base 1000000
    --normalization RMSNorm
    --disable_bias_linear
    --num_moe_experts 8
    --moe_router_topk 2
    --moe_token_dispatcher_type alltoall
)

exec "$WORKDIR/../run.sh" \
    "${MODEL_ARGS[@]}" \
    "$@"
