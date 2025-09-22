#!/usr/bin/env bash

WORKDIR=`dirname $(realpath $0)`

compose_file=$WORKDIR/compose.yaml

docker compose -f $compose_file down --remove-orphans
docker compose -f $compose_file up -d
sleep 1

docker compose -f $compose_file exec -it host-1 /phantora/dist/phantora_run deepspeed -H /hostfile /phantora/tests/test_deepspeed.py "$@"
