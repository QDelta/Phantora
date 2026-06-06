#!/usr/bin/env bash
# Phantora preset: Mixtral 8x7B on Megatron.
#
# Unlike the gpt-oss preset (a dimension-only proxy), this is the *actual*
# Mixtral architecture: Megatron-core's MoE GPTModel is exactly GQA attention +
# RoPE + SwiGLU experts with top-2 routing over 8 experts, which is what Mixtral
# is. Dimensions match mistralai/Mixtral-8x7B (hidden 4096, ffn 14336, 32 layers,
# 32 heads / 8 KV heads, head_dim 128, 8 experts, top-2, vocab 32000).
#
# ~46.7B total / ~12.9B active. Needs expert (and typically tensor/pipeline)
# parallelism plus recompute to fit; pass parallelism via $@, e.g.:
#   python3 config_gen.py --nhost 4 --ngpu 8 --vram_mib 81920
#   ./mixtral/run_mixtral_8x7b.sh --expert_model_parallel_size 8 --tensor_parallel_size 4 \
#       --pipeline_model_parallel_size 1 --recompute_activations --sequence_length 4096

WORKDIR=$(dirname "$(realpath "$0")")

MODEL_ARGS=(
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
