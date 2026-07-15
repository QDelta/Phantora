#!/usr/bin/env bash
# Record a Phantora performance database for the model presets on the local GPU.
#
# Each preset is run once in --record-perf-db mode (the simulator profiles every
# kernel on the GPU and dumps the timing tables as CSV under tests/perfdb/$DB).
# Recording merges into the existing DB, so all presets accumulate into one
# directory. Recording uses a small layer count on purpose: the set of unique
# (op, shape) keys is independent of num_layers, so the DB also serves the
# full-depth presets at the SAME parallelism / sequence-length / micro-batch.
#
# After recording, the DB replays on a GPU-less machine:
#   cd tests/docker/<fw> && python3 config_gen.py ... --perf-db $DB && ./run.sh ...
#
# Usage:  tests/perfdb/record_all.sh [DB_NAME]   (default: l40s)
set -euo pipefail

DB="${1:-l40s}"
ROOT="$(cd "$(dirname "$0")/../docker" && pwd)"
DC="docker compose"

run() {  # $1=framework dir  $2..=launch argv (inside host-1)
  local fw="$1"; shift
  local cf="$ROOT/$fw/compose.yaml"
  echo "=== recording: $fw :: $* ==="
  $DC -f "$cf" down --remove-orphans >/dev/null 2>&1 || true
  $DC -f "$cf" up -d >/dev/null 2>&1
  sleep 6
  $DC -f "$cf" exec -T "$@"
  $DC -f "$cf" down --remove-orphans >/dev/null 2>&1 || true
}

# ---- Megatron (ngpu 2) ----
( cd "$ROOT/megatron" && python3 config_gen.py --nhost 1 --ngpu 2 --vram_mib 81920 --record-perf-db "$DB" >/dev/null )
MEGA="host-1 /phantora/dist/phantora_run torchrun --nproc_per_node 2 --nnodes 1 --rdzv_backend c10d --rdzv_endpoint=host-1:12345 /phantora/tests/test_megatron.py --num_layers 4 --sequence_length 1024 --iterations 3"
# llama3-8B (dense)
run megatron $MEGA --tensor_parallel_size 2 --hidden_size 4096 --ffn_hidden_size 14336 --num_attention_heads 32 --num_query_groups 8 --vocab_size 128256 --swiglu --position_embedding_type rope --rotary_base 500000 --normalization RMSNorm --disable_bias_linear
# Mixtral 8x7B (MoE)
run megatron $MEGA --tensor_parallel_size 1 --expert_model_parallel_size 2 --hidden_size 4096 --ffn_hidden_size 14336 --num_attention_heads 32 --num_query_groups 8 --vocab_size 32000 --swiglu --position_embedding_type rope --rotary_base 1000000 --normalization RMSNorm --disable_bias_linear --num_moe_experts 8 --moe_router_topk 2 --moe_token_dispatcher_type alltoall
# Qwen3 30B-A3B (MoE)
run megatron $MEGA --tensor_parallel_size 1 --expert_model_parallel_size 2 --hidden_size 2048 --ffn_hidden_size 768 --num_attention_heads 32 --num_query_groups 4 --kv_channels 128 --num_moe_experts 128 --moe_router_topk 8 --moe_token_dispatcher_type alltoall --qk_layernorm --vocab_size 151936 --swiglu --position_embedding_type rope --rotary_base 1000000 --normalization RMSNorm --disable_bias_linear

# ---- DeepSpeed (ngpu 2, larger sim VRAM for gpt-oss activations) ----
( cd "$ROOT/deepspeed" && python3 config_gen.py --nhost 1 --ngpu 2 --vram_mib 200000 --record-perf-db "$DB" >/dev/null )
DS="host-1 /phantora/dist/phantora_run deepspeed -H /hostfile /phantora/tests/test_deepspeed.py --num_layers 4 --sequence_length 1024 --iterations 3"
# llama3-8B (dense, PipelineModule)
run deepspeed $DS --hidden_size 4096 --ffn_hidden_size 14336 --num_attention_heads 32 --num_key_value_heads 8 --vocab_size 128256 --rope_theta 500000
# gpt-oss-20B (MoE, ZeRO-3)
run deepspeed $DS --model gpt_oss --hidden_size 2880 --ffn_hidden_size 2880 --num_attention_heads 64 --num_key_value_heads 8 --head_dim 64 --num_experts 32 --experts_per_tok 4 --vocab_size 201088 --zero_stage 3

# ---- TorchTitan (ngpu 2) ----
( cd "$ROOT/torchtitan" && python3 config_gen.py --nhost 1 --ngpu 2 --vram_mib 81920 --record-perf-db "$DB" >/dev/null )
TT="--workdir /phantora host-1 /phantora/dist/phantora_run torchrun --nproc_per_node 2 --nnodes 1 --rdzv_backend c10d --rdzv_endpoint=host-1:12345 /phantora/tests/test_torchtitan.py --training.steps=3"
# llama3-8B (dense, FSDP2)
run torchtitan $TT --job.config_file=/phantora/tests/test_torchtitan_llama3_8b.toml --model.flavor=debugmodel
# Qwen3 (MoE)
run torchtitan $TT --job.config_file=/phantora/tests/test_torchtitan_qwen3_moe.toml --model.flavor=debugmodel_moe --training.debug_moe_force_load_balance

echo "=== done; DB at tests/perfdb/$DB ==="
