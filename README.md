# Phantora

**Estimate the throughput, MFU, and iteration time of distributed LLM training on clusters you don't own — by running a Megatron, DeepSpeed, or TorchTitan script on a single GPU.**

Phantora is a *hybrid* GPU cluster simulator for ML system performance estimation. Instead of asking you to reimplement your workload in a simulator's DSL, Phantora intercepts the GPU and NCCL calls of an ML framework, simulates them, and lets the framework's own performance-logging code (e.g., `mfu`, iteration time) print as if you ran on a real cluster.

Phantora was accepted at **NSDI 2026** 🎉 — see the [paper](https://danyangzhuo.com/papers/NSDI26-Phantora.pdf) for design details.

## How it works

You add a few lines to your training script to enable the Phantora tracer, then run it on a single GPU. A stub `libcuda.so` and `libnccl.so` intercept every GPU and collective call and forward them over a Unix socket to a Rust simulator (`phantora_server`), which:

- estimates each CUDA kernel's execution time from a one-time profile on the local GPU,
- simulates NCCL collectives over a flow-level network simulator with a configurable cluster topology,
- maintains a per-rank virtual clock and event queue, and
- patches `time.perf_counter` (and a few framework hooks) so the framework sees simulated time.

The framework's own logging then emits the metrics it would on a real cluster — produced from a single GPU and a virtual cluster configuration of your choosing.

## Limitations

### Control flow must be data-independent

Phantora simulates GPU computation and communication, but the tensor values produced by simulated kernels are arbitrary. **Your training script's control flow must therefore not depend on the contents of GPU tensors.** Any branch that reads a value out of a tensor — for example, an early-exit on a loss threshold, a gradient-norm check, or a NaN/inf rescue path — will see garbage data and may follow a path that does not match real execution. Loss values printed during simulation are also not meaningful.

Concretely:

- **Megatron**: gradient clipping must be disabled. It copies a norm to CPU and takes a `sqrt`, which can fault on the random GPU memory contents under simulation.
- Avoid early-stopping logic or NaN/inf rescue paths in the iterations being simulated.
- Stick to control flow that depends only on hyperparameters, iteration counts, and configuration. This covers the common case in LLM pre-training.

### Limited NCCL coverage

Phantora ships a stub `libnccl.so` that intercepts NCCL calls and forwards them to the simulator. Only a subset of the NCCL API is currently implemented — calling an unsupported entry point will abort with `NOT_IMPLEMENTED`.

**Collective and point-to-point operations**

| NCCL op | Status |
| --- | --- |
| `ncclAllReduce` | ✅ Supported |
| `ncclAllGather` | ✅ Supported |
| `ncclReduceScatter` | ✅ Supported |
| `ncclBcast` (legacy in-place API) | ✅ Supported |
| `ncclBroadcast` | ❌ Not implemented |
| `ncclReduce` | ❌ Not implemented |
| `ncclSend` (point-to-point) | 🚧 Work in progress |
| `ncclRecv` (point-to-point) | 🚧 Work in progress |

`ncclSend` / `ncclRecv` are actively being worked on — once they land, pipeline parallelism and (via grouped point-to-point) expert parallelism / all-to-all will become simulatable across all three frameworks.

**Communicator, group, and utility calls**

| NCCL op | Status |
| --- | --- |
| `ncclCommInitRank`, `ncclCommInitRankConfig`, `ncclCommInitRankScalable` | ✅ Supported |
| `ncclCommInitAll` | ✅ Supported |
| `ncclCommSplit` | ✅ Supported |
| `ncclCommDestroy`, `ncclCommAbort`, `ncclCommFinalize` | ✅ Supported |
| `ncclCommRegister`, `ncclCommDeregister` | ✅ Supported (no-op) |
| `ncclGroupStart`, `ncclGroupEnd` | ✅ Supported |
| `ncclGetUniqueId`, `ncclGetVersion`, `ncclGetErrorString`, `ncclGetLastError`, `ncclCommGetAsyncError` | ✅ Supported |
| `ncclCommCount`, `ncclCommCuDevice`, `ncclCommUserRank` | ❌ Not implemented |
| `ncclRedOpCreatePreMulSum`, `ncclRedOpDestroy` | ❌ Not implemented |

The full set of stubs lives in [`stub/nccl.c`](stub/nccl.c). Pull requests to expand NCCL coverage are very welcome.

### Framework feature support

The matrix below summarises which features of each supported framework Phantora can simulate today. A 🚧 row maps to a missing underlying collective in the NCCL table above, or to a separate communication library that Phantora does not yet intercept.

| Feature | Megatron | DeepSpeed | TorchTitan | Required collective(s) |
| --- | :---: | :---: | :---: | --- |
| Data parallelism (DP) | ✅ | ✅ | ✅ | AllReduce |
| Tensor parallelism (TP) | ✅ | ✅ (via Megatron-LM) | ✅ | AllReduce, AllGather, ReduceScatter |
| ZeRO-1 / ZeRO-2 / ZeRO-3 | — | ✅ | — | AllReduce, AllGather, ReduceScatter |
| FSDP / FSDP2 | — | — | ✅ | AllGather, ReduceScatter |
| Activation checkpointing | ✅ | ✅ | ✅ | (no extra communication) |
| Pipeline parallelism (PP) | 🚧 | 🚧 | 🚧 | `ncclSend` / `ncclRecv` (point-to-point) — in progress |
| Expert parallelism / MoE | 🚧 | 🚧 | 🚧 | All-to-all via `ncclSend` / `ncclRecv` groups (in progress), and/or [DeepEP](https://github.com/deepseek-ai/DeepEP) |

Two communication paths gate the 🚧 rows, and we are actively closing both:

- **NCCL point-to-point — in progress.** `ncclSend` / `ncclRecv` are being added to the stub. Pipeline parallelism uses them directly, and NCCL's all-to-all is implemented as grouped point-to-point, so MoE / expert parallelism unlocks at the same time.
- **DeepEP — on the roadmap.** Some recent MoE training stacks (e.g., DeepSeek-style models) bypass NCCL entirely and use [DeepEP](https://github.com/deepseek-ai/DeepEP) for expert dispatch/combine. We plan to add a DeepEP interception layer so those stacks can be simulated as well.

Rows marked `—` mean the feature does not exist in that framework. If you'd like to help land any of the in-progress pieces sooner, contributions are very welcome — see [Contributing](#contributing).

## Requirements

- Linux x86_64
- An NVIDIA GPU for kernel profiling. The build targets compute capability 8.0 (e.g., A100, H200) and 9.0 (e.g., H100); other GPUs may work but are untested.
- CUDA 11.8 (the Docker image is based on `nvidia/cuda:11.8.0-devel-ubuntu22.04`)
- Docker with Docker Compose (recommended)
- Python 3.11.9 if building outside Docker
- Tens of GB of free disk for the image and downloaded model assets

## Build Instructions

Clone the repository via `git`.

```bash
git clone https://github.com/QDelta/Phantora
cd Phantora
git submodule update --init --recursive
```

Note: `pytorch/` is a git submodule pointing at a custom PyTorch branch (`2.7.1-phantora`) with the function tracer patched.

Docker (with Docker Compose) is recommended for building and using Phantora. In the repository root, run:

```bash
docker build -t phantora .
```

It might take a while.

If you want to build it locally without Docker, also refer to `Dockerfile` for the detailed commands.

## Try our examples

Once you built the `phantora` docker image, you can try our examples of distributed training using Megatron, DeepSpeed and TorchTitan. The examples will launch multiple containers (using Docker Compose) to simulate a GPU cluster.

For example, to simulate a distributed Llama2 7B training using Megatron:

```bash
cd tests/docker/megatron

# Generate configurations for a 16-GPU cluster with 140GB VRAM per GPU
python3 config_gen.py --nhost 4 --ngpu 4 --vram_mib 143771

# Start training
./run.sh

# ... look at the terminal output

# Cleanup containers and other temporary files
./stop.sh
```

Similar for DeepSpeed and TorchTitan.

For TorchTitan, the `tokenizer.model` of Llama3 is needed, you can get it from its [huggingface repo](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct/blob/main/original/tokenizer.model). Place `tokenizer.model` in `tests/assets` before starting.

`run.sh` will pass its arguments to the corresponding scripts (`tests/test_{megatron,deepspeed,torchtitan}.py`)

## Adapt your training scripts

Scripts and configurations in `tests/` will be good examples.

Generally, edit your script like this:

```python
from phantora_utils import (
    enable_function_tracer,
    disable_function_tracer,
)

# ... Your original script
# Use time.perf_counter or phantora_utils.time for timers

if __name__ == "__main__":
    enable_function_tracer()
    # ... Your original main
    disable_function_tracer()
```

For other configurations, you can refer to generated configurations in `tests/docker/{megatron,deepspeed,torchtitan}`.

## Accuracy: Validated Configurations

The tables below list the configurations we have validated against real-hardware ground truth. **We would love community contributions of additional ground-truth measurements** so that we can better understand Phantora's accuracy across hardware, frameworks, and workloads. See [Contributing ground truth](#contributing-ground-truth) below.

### Hardware

Phantora itself always runs on a single GPU. The three host configurations we used are:

| Phantora host | CPU | GPU |
| --- | --- | --- |
| H200 host | 2× AMD EPYC 9355 | 1× NVIDIA H200 NVL |
| A100 host | 2× Intel Xeon Gold 6348 | 1× NVIDIA A100 40G |
| RTX 3090 host | 2× Intel Xeon Gold 5215 | 1× NVIDIA RTX 3090 |

Ground truth comes from a mix of in-house testbeds and published reports:

- **Megatron Llama2 7B** — in-house ground truth on the same physical box as the H200 host, using all 4× H200 NVL GPUs over NVLink.
- **TorchTitan (FSDP2)** — TorchTitan's published [H100](https://github.com/pytorch/torchtitan/blob/4b3f2e41a084bf79a8540068ed525539d1244edd/docs/performance.md) and [A100-80G](https://github.com/pytorch/torchtitan/blob/217cc94e2abf8472db098c1c0e5e020e62dcfc7d/docs/performance.md) performance reports. H100 targets are simulated on the H200 host; A100-80G targets are simulated on the A100 host. In both cases Phantora's VRAM is configured to 80 GB to match the target.
- **DeepSpeed non-LLM workloads** — in-house 4-server, 8-GPU RTX 3090 cluster (2 GPUs per server over Ethernet).

### Megatron — Llama2 7B

Comparison against on-testbed measurements, with and without optimizer.

| Parallelism | Micro batch |
| --- | --- |
| TP=4 | 1 |
| TP=4 | 2 |
| DP=2, TP=2 | 1 |

Reported accuracy: average error **3.7%**, maximum **5.3%** (TP=4, micro batch 1).

### TorchTitan (FSDP2) — Large-scale public reports

Ground truth from TorchTitan's published H100 / A100-80G performance reports (linked above).

| Model | Cluster | Notes |
| --- | --- | --- |
| Llama3 8B | 8× H100 | batch=2 |
| Llama3 8B | 128× H100 | batch=2 |
| Llama3 8B | 64× A100 | |
| Llama2 13B | 64× A100 | |
| Llama2 13B | 64× A100 | without activation checkpointing |
| Llama2 70B | 64× A100 | |
| Llama2 70B | 64× A100 | batch=2 |
| Llama3 70B | 64× A100 | |

Reported accuracy: average error **2.9%**, maximum **8.5%** (Llama2 13B).

### DeepSpeed — Non-LLM workloads

| Model | GPU counts |
| --- | --- |
| ResNet-50 | 2, 4, 8 |
| Stable Diffusion | 2, 4, 8 |
| GAT (Graph Attention Network) | 2, 4, 8 |

Reported accuracy: average error **6.6%**, maximum **8.1%** (Stable Diffusion, 2 GPUs).

### Contributing ground truth

If you can run any of the supported frameworks (Megatron, DeepSpeed, TorchTitan) on real GPU hardware, we would love your help expanding this list. Please open an issue or pull request including:

- **Hardware**: GPU model and count, CPU, network interconnect, per-server topology
- **Workload**: framework + version, model and size, parallelism strategy (DP / TP / PP / FSDP / ZeRO stage), micro-batch size, global batch size, sequence length, activation checkpointing on/off, optimizer settings
- **Ground-truth numbers**: throughput (e.g., tokens/s/GPU) and/or average iteration time, with confidence interval if possible
- **Phantora simulation result** and the configuration files used (e.g., from `tests/docker/{megatron,deepspeed,torchtitan}`) so we can reproduce
- **Versions**: Phantora commit hash, PyTorch version, framework version

We especially welcome data points that fall outside what is covered above — different GPUs (e.g., MI300X, B200, GB200), interconnects (RoCE, different InfiniBand speeds, multi-rail), parallelism strategies, models, sequence lengths, or training optimizations.

## Contributing

Contributions of any size are welcome:

- **Ground-truth measurements** to expand the validation matrix — see the [Contributing ground truth](#contributing-ground-truth) checklist above.
- **NCCL coverage** — see the matrix in [Limited NCCL coverage](#limited-nccl-coverage). PRs that add `ncclSend`/`ncclRecv` (which would unblock pipeline parallelism), `ncclReduce`, `ncclBroadcast`, or any other unimplemented entry point are especially valuable.
- **New ML frameworks or models** — Phantora's design is framework-agnostic; small runtime patches let new PyTorch-based frameworks run unchanged.
- **Bug reports and feature requests** — please file them on [GitHub Issues](https://github.com/QDelta/Phantora/issues).

For any non-trivial change, please open an issue first to discuss the approach.

## License

Phantora is licensed under the [Apache License 2.0](LICENSE).

## Citation

If you use Phantora for your research, please cite our [paper](https://danyangzhuo.com/papers/NSDI26-Phantora.pdf).

```bibtex
@inproceedings{qin2026phantora,
  title="{Phantora: Maximizing Code Reuse in Simulation-based Machine Learning System Performance Estimation}",
  author={Jianxing Qin and Jingrong Chen and Xinhao Kong and Yongji Wu and Tianjun Yuan and Liang Luo and Zhaodong Wang and Ying Zhang and Tingjun Chen and Alvin R. Lebeck and Danyang Zhuo},
  booktitle={The 23rd USENIX Symposium on Networked Systems Design and Implementation (NSDI)},
  year={2026},
}
```
