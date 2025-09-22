#!/usr/bin/env python3

SIMULATOR_TEMPLATE = r"""
  simulator:
    image: "phantora:latest"
    volumes:
      - /run/phantora:/run/phantora
      - ./netconfig.toml:/netconfig.toml:ro
    pid: host
    ipc: host
    environment:
      - PHANTORA_LOG=${{PHANTORA_LOG:-info}}
      - PHANTORA_SOCKET_PREFIX=/run/phantora/phantora
    command: /phantora/dist/phantora_server --netconfig /netconfig.toml
    cpuset: '{cpuset}'
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']
              capabilities: [gpu]
"""

HOST_TEMPLATE = r"""
  host-{host_id}:
    image: "phantora:latest"
    volumes:
      - /run/phantora:/run/phantora
      - ../..:/phantora/tests:ro
    pid: host
    ipc: host
    environment:
      - CUDA_DEVICE_MAX_CONNECTIONS=1
      - PHANTORA_NGPU={ngpu}
      - PHANTORA_VRAM_MIB={vram_mib}
      - PHANTORA_IGNORE_CPU_TIME=1
      - PHANTORA_SOCKET_PREFIX=/run/phantora/phantora
    hostname: host-{host_id}
    command: sleep infinity
    cpuset: '{cpuset}'
    depends_on:
      - simulator
"""

NETCONFIG_TEMPLATE = r"""
host_mapping = {host_list}

[simulator]
loopback_speed = 2880
fairness = "PerFlowMaxMin"

[topology]
type = "TwoLayerMultiPath"

[topology.args]
nspines = 2
nracks = {nracks}
rack_size = 2
host_bw = 800
rack_uplink_port_bw = 800
load_balancer_type = "EcmpEverything"
"""

if __name__ == '__main__':
    import argparse
    from os.path import dirname, realpath, join
    from multiprocessing import cpu_count
    script_dir = dirname(realpath(__file__))

    nproc = cpu_count()
    if nproc <= 2:
        default_sim_core = str(nproc - 1)
        default_host_cpuset = str(nproc - 1)
    else:
        default_sim_core = str(nproc // 2)
        default_host_cpuset = f"{nproc // 2 + 1}-{nproc - 1}"

    parser = argparse.ArgumentParser()
    parser.add_argument("--nhost", type=int, default=4)
    parser.add_argument("--ngpu", type=int, default=4)
    parser.add_argument("--vram_mib", type=int, default=143771)
    parser.add_argument("--cpuset_sim", type=str, default=default_sim_core)
    parser.add_argument("--cpuset_host", type=str, default=default_host_cpuset)
    args = parser.parse_args()

    nhosts = args.nhost
    ngpu = args.ngpu

    with open(join(script_dir, "compose.yaml"), "w") as f:
      f.write("services:")
      f.write(SIMULATOR_TEMPLATE.format(cpuset=args.cpuset_sim))
      for i in range(1, nhosts + 1):
          f.write(HOST_TEMPLATE.format(
              host_id=i, ngpu=ngpu, vram_mib=args.vram_mib, cpuset=args.cpuset_host
          ))

    with open(join(script_dir, "netconfig.toml"), "w") as f:
        host_list = str([f"host-{i}" for i in range(1, nhosts + 1)])
        f.write(NETCONFIG_TEMPLATE.format(host_list=host_list, nracks=(nhosts + 1) // 2))

    with open(join(script_dir, "config.sh"), "w") as f:
        f.write(f"EVAL_NHOST={nhosts}\n")
        f.write(f"EVAL_NGPU={ngpu}\n")
