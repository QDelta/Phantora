#!/usr/bin/env bash
set -euo pipefail

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$WORKDIR"

docker build -t phantora .
cd tests/docker/ColossalAI
python3 config_gen.py --nhost 1 --ngpu 2 --vram_mib 24576
source ./config.sh

compose_file="$(pwd)/compose.yaml"
ITERATIONS="${ITERATIONS:-4}"
SEQ_LEN="${SEQ_LEN:-9092}"
NUM_LAYERS="${NUM_LAYERS:-12}"
HIDDEN_SIZE="${HIDDEN_SIZE:-1024}"
FFN_HIDDEN_SIZE="${FFN_HIDDEN_SIZE:-4096}"
NUM_HEADS="${NUM_HEADS:-16}"

base_cmd="torchrun --nproc_per_node $EVAL_NGPU --nnodes $EVAL_NHOST --rdzv_backend c10d --rdzv_endpoint=\"host-1:12345\" /phantora/tests/test_ColossalAI.py --num_layers $NUM_LAYERS --hidden_size $HIDDEN_SIZE --ffn_hidden_size $FFN_HIDDEN_SIZE --num_attention_heads $NUM_HEADS --sequence_length $SEQ_LEN --micro_batch_size 1 --iterations $ITERATIONS"

run_cluster() {
  local cmd="$1"
  docker compose -f "$compose_file" down --remove-orphans
  docker compose -f "$compose_file" up -d
  sleep 1
  for w in $(seq 2 "$EVAL_NHOST"); do
    docker compose -f "$compose_file" exec -it -d host-"$w" bash -c "$cmd"
  done
  docker compose -f "$compose_file" exec -it host-1 bash -c "$cmd"
}

# With Phantora (simulate)
run_cluster "PHANTORA=1 /phantora/dist/phantora_run $base_cmd"

# Without Phantora (baseline)
run_cluster "env -u PHANTORA $base_cmd"
./stop.sh
cd ../../..
