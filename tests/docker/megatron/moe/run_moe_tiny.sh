#!/usr/bin/env bash
# Phantora preset: tiny Mixtral-style MoE (8 experts, top-2).
#
# Deliberately small so it builds fast; the point is to exercise Megatron's MoE
# expert all-to-all (dispatch/combine), which it emits as grouped ncclSend/
# ncclRecv across the expert-parallel (EP) ranks when
# moe_token_dispatcher_type=alltoall. Run with --expert_model_parallel_size >= 2
# (and a matching cluster from config_gen) to actually trigger the all-to-all.
#
# Example:
#   python3 config_gen.py --nhost 1 --ngpu 2 --vram_mib 81920
#   ./moe/run_moe_tiny.sh --tensor_parallel_size 1 --expert_model_parallel_size 2 --iterations 2

WORKDIR=$(dirname "$(realpath "$0")")

MODEL_ARGS=(
    --num_layers 4
    --hidden_size 1024
    --ffn_hidden_size 1024
    --num_attention_heads 16
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
