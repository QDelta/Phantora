#!/usr/bin/env bash

WORKDIR=`dirname $(realpath $0)`
source $WORKDIR/config.sh

compose_file=$WORKDIR/compose.yaml

docker compose -f $compose_file down
docker compose -f $compose_file up -d
sleep 1

cmd="/phantora/dist/phantora_run torchrun --nproc_per_node $EVAL_NGPU --nnodes $EVAL_NHOST --rdzv_backend c10d --rdzv_endpoint=\"host-1:12345\" /phantora/tests/test_torchtitan.py $@"

for w in $(seq 2 $EVAL_NHOST); do
  docker compose -f $compose_file exec -it -d --workdir /phantora host-$w bash -c "$cmd"
done

docker compose -f $compose_file exec -it --workdir /phantora host-1 bash -c "$cmd"
