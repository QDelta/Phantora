#!/usr/bin/env python3

# GPU presets: `--gpu <name>` bundles the simulated per-GPU VRAM, the GPU name
# presented to the framework (PHANTORA_GPU_NAME), and the perf-db directory
# (tests/perfdb/<name>) of kernel timings recorded for that GPU. Add new GPUs
# here; record their DB once with `--gpu <name> --record` on a real GPU.
GPU_PRESETS = {
    "l40s": {"vram_mib": 46068, "name": "NVIDIA L40S"},
    "a100": {"vram_mib": 81920, "name": "NVIDIA A100-SXM4-80GB"},
    "h100": {"vram_mib": 81559, "name": "NVIDIA H100 80GB HBM3"},
    "h200": {"vram_mib": 143771, "name": "NVIDIA H200"},
}

DEFAULT_VRAM_MIB = 143771

GPU_RESERVATION = """
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']
              capabilities: [gpu]"""

SIMULATOR_TEMPLATE = r"""
  simulator:
    image: "phantora:latest"
    volumes:
      - /run/phantora:/run/phantora
      - ./netconfig.toml:/netconfig.toml:ro
      - ../..:/phantora/tests
    pid: host
    ipc: host
    environment:
      - PHANTORA_LOG=${{PHANTORA_LOG:-info}}
      - PHANTORA_USE_CUPTI=${{PHANTORA_USE_CUPTI:-1}}
      - PHANTORA_SOCKET_PREFIX=/run/phantora/phantora
    command: /phantora/dist/phantora_server --netconfig /netconfig.toml{perfdb_cmd}
    cpuset: '{cpuset}'{gpu_reservation}
"""

HOST_TEMPLATE = r"""
  host-{host_id}:
    image: "phantora:latest"
    volumes:
      - /run/phantora:/run/phantora
      - ../..:/phantora/tests:ro
      - ./hostfile:/hostfile:ro
      - ./deepspeed_env:/root/.deepspeed_env:ro
    pid: host
    ipc: host
    environment:
      - PHANTORA_NGPU={ngpu}
      - PHANTORA_VRAM_MIB={vram_mib}
      - PHANTORA_IGNORE_CPU_TIME=${{PHANTORA_IGNORE_CPU_TIME:-1}}
      - PHANTORA_SOCKET_PREFIX=/run/phantora/phantora{gpu_name_env}
    hostname: host-{host_id}
    command: /usr/sbin/sshd -D
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
    parser.add_argument("--gpu", type=str, default=None,
                        help="GPU preset (e.g. h200): sets simulated VRAM + GPU name and "
                             "replays kernel timings from tests/perfdb/<gpu> with no real "
                             f"GPU. Known: {', '.join(sorted(GPU_PRESETS))}.")
    parser.add_argument("--record", action="store_true",
                        help="With --gpu, record kernel timings into tests/perfdb/<gpu> on "
                             "a real GPU instead of replaying.")
    parser.add_argument("--vram_mib", type=int, default=None,
                        help="Simulated per-GPU VRAM in MiB. Optional; defaults to the "
                             f"--gpu preset's value (or {DEFAULT_VRAM_MIB} if neither is "
                             "given). Overrides the preset when set.")
    parser.add_argument("--cpuset_sim", type=str, default=default_sim_core)
    parser.add_argument("--cpuset_host", type=str, default=default_host_cpuset)
    parser.add_argument("--perf-db", dest="perf_db", type=str, default=None,
                        help="(Advanced) Replay kernel timings from tests/perfdb/<NAME> "
                             "without a GPU preset (no GPU; drops the GPU reservation).")
    parser.add_argument("--record-perf-db", dest="record_perf_db", type=str, default=None,
                        help="(Advanced) Record kernel timings into tests/perfdb/<NAME> "
                             "(requires a GPU).")
    args = parser.parse_args()

    gpu_name = None
    if args.gpu is not None:
        if args.perf_db or args.record_perf_db:
            parser.error("--gpu cannot be combined with --perf-db/--record-perf-db")
        if args.gpu not in GPU_PRESETS:
            parser.error(f"unknown --gpu {args.gpu!r}; known: {', '.join(sorted(GPU_PRESETS))}")
        preset = GPU_PRESETS[args.gpu]
        vram_mib = args.vram_mib if args.vram_mib is not None else preset["vram_mib"]
        gpu_name = preset["name"]
        if args.record:
            perfdb_cmd = f" --record-perf-db /phantora/tests/perfdb/{args.gpu}"
            gpu_reservation = GPU_RESERVATION
        else:
            perfdb_cmd = f" --perf-db /phantora/tests/perfdb/{args.gpu}"
            gpu_reservation = ""  # replay needs no GPU
    else:
        if args.record:
            parser.error("--record requires --gpu")
        if args.perf_db and args.record_perf_db:
            parser.error("--perf-db and --record-perf-db are mutually exclusive")
        vram_mib = args.vram_mib if args.vram_mib is not None else DEFAULT_VRAM_MIB
        if args.perf_db:
            perfdb_cmd = f" --perf-db /phantora/tests/perfdb/{args.perf_db}"
            gpu_reservation = ""  # replay needs no GPU
        elif args.record_perf_db:
            perfdb_cmd = f" --record-perf-db /phantora/tests/perfdb/{args.record_perf_db}"
            gpu_reservation = GPU_RESERVATION
        else:
            perfdb_cmd = ""
            gpu_reservation = GPU_RESERVATION

    gpu_name_env = f"\n      - PHANTORA_GPU_NAME={gpu_name}" if gpu_name else ""

    nhosts = args.nhost
    ngpu = args.ngpu

    with open(join(script_dir, "compose.yaml"), "w") as f:
      f.write("services:")
      f.write(SIMULATOR_TEMPLATE.format(cpuset=args.cpuset_sim, perfdb_cmd=perfdb_cmd, gpu_reservation=gpu_reservation))
      for i in range(1, nhosts + 1):
          f.write(HOST_TEMPLATE.format(
              host_id=i, ngpu=ngpu, vram_mib=vram_mib, cpuset=args.cpuset_host,
              gpu_name_env=gpu_name_env,
          ))

    with open(join(script_dir, "hostfile"), "w") as f:
        for i in range(1, nhosts + 1):
            f.write(f"host-{i} slots={ngpu}\n")

    with open(join(script_dir, "netconfig.toml"), "w") as f:
        host_list = str([f"host-{i}" for i in range(1, nhosts + 1)])
        f.write(NETCONFIG_TEMPLATE.format(host_list=host_list, nracks=(nhosts + 1) // 2))

    with open(join(script_dir, "deepspeed_env"), "w") as f:
        f.write("LD_LIBRARY_PATH=/phantora/dist:/phantora/pytorch/torch/lib:/usr/local/cuda/lib64:/usr/local/python3.11.9/lib\n")
        f.write("LD_PRELOAD=/phantora/dist/libcuda.so.1\n")
        f.write("PHANTORA_SOCKET_PREFIX=/run/phantora/phantora\n")
        f.write("PHANTORA=1\n")
        f.write(f"PHANTORA_NGPU={ngpu}\n")
        f.write(f"PHANTORA_VRAM_MIB={vram_mib}\n")
        f.write("PHANTORA_IGNORE_CPU_TIME=1\n")
        if gpu_name:
            f.write(f"PHANTORA_GPU_NAME={gpu_name}\n")
