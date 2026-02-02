#!/usr/bin/env bash

WORKDIR=`dirname $(realpath $0)`
compose_file=$WORKDIR/compose.yaml
docker compose -f $compose_file down --remove-orphans
sudo rm -f /run/phantora/phantora*
