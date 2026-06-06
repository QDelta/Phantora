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

## What Phantora can simulate

Phantora runs your **unmodified** Megatron-LM, DeepSpeed, or TorchTitan training script and estimates how it would perform on a virtual cluster you describe — any GPU count, per-GPU VRAM, and network topology — all from a single GPU.

### Available model presets

Ready-to-run presets ship for each framework. `✅` links to the preset — a launcher script for Megatron/DeepSpeed (under `tests/docker/<framework>/`), or a `.toml` config for TorchTitan; `—` means there is no preset for that pair (the model may still be expressible by hand). All MoE presets assume **load-balanced experts** ([why](#control-flow-must-be-data-independent)).

| Model | Megatron | DeepSpeed | TorchTitan |
| --- | :---: | :---: | :---: |
| **Dense** | | | |
| Llama2 7B | [✅](tests/docker/megatron/llama/run_llama2_7b.sh) | [✅](tests/docker/deepspeed/llama/run_llama2_7b.sh) | — |
| Llama2 13B | [✅](tests/docker/megatron/llama/run_llama2_13b.sh) | [✅](tests/docker/deepspeed/llama/run_llama2_13b.sh) | — |
| Llama2 70B | [✅](tests/docker/megatron/llama/run_llama2_70b.sh) | [✅](tests/docker/deepspeed/llama/run_llama2_70b.sh) | — |
| Llama3 8B | [✅](tests/docker/megatron/llama/run_llama3_8b.sh) | [✅](tests/docker/deepspeed/llama/run_llama3_8b.sh) | [✅](tests/docker/torchtitan/llama3/run_llama3_8b.sh) ¹ |
| Llama3 70B | [✅](tests/docker/megatron/llama/run_llama3_70b.sh) | [✅](tests/docker/deepspeed/llama/run_llama3_70b.sh) | — |
| **MoE** | | | |
| Mixtral 8×7B | [✅](tests/docker/megatron/moe/run_mixtral_8x7b.sh) ² | — | — |
| Expert-parallel MoE (synthetic) | — | ✅ ³ | — |
| gpt-oss | [✅](tests/docker/megatron/moe/run_gpt_oss_20b.sh) ⁴ | [✅](tests/docker/deepspeed/gpt_oss/run_gpt_oss_tiny.sh) | — |
| Qwen3 MoE | — | — | [✅](tests/test_torchtitan_qwen3_moe.toml) ⁵ |

1. TorchTitan Llama3 8B also has a pipeline-parallel variant, [`test_torchtitan_llama3_8b_pp.toml`](tests/test_torchtitan_llama3_8b_pp.toml).
2. The real Mixtral architecture — Megatron-core's MoE GPTModel *is* GQA + RoPE + top-2-of-8 SwiGLU experts. For a quick check, scale it down with `--num_layers` / `--hidden_size` overrides.
3. DeepSpeed's expert-parallel all-to-all is exercised by a small synthetic MoE stack via `--model deepspeed_moe` (no dedicated launcher; experts are `Linear-GELU-Linear`).
4. Megatron builds from its own `GPTModel`, so this matches gpt-oss-20b's *dimensions* as a throughput proxy, not the exact architecture (no attention sinks / clamped gating). DeepSpeed's gpt-oss is the real Hugging Face model.
5. Run with `--training.debug_moe_force_load_balance`, e.g. `./run.sh --job.config_file=tests/test_torchtitan_qwen3_moe.toml --training.debug_moe_force_load_balance`.

See [Try our examples](#try-our-examples) for how to launch one.

### Parallelism support

The strategies Phantora models today, which **compose** (e.g. TP+EP with sequence parallelism, or DP+PP):

| Strategy | Megatron | DeepSpeed | TorchTitan | Required collective(s) |
| --- | :---: | :---: | :---: | --- |
| Data parallelism (DP) | ✅ | ✅ | ✅ | AllReduce |
| Tensor parallelism (TP) | ✅ | ✅ | ✅ | AllReduce, AllGather, ReduceScatter |
| ZeRO-1 / ZeRO-2 / ZeRO-3 | — | ✅ | — | AllReduce, AllGather, ReduceScatter |
| FSDP / FSDP2 | — | — | ✅ | AllGather, ReduceScatter |
| Activation checkpointing | ✅ | ✅ | ✅ | (no extra communication) |
| Pipeline parallelism (PP) | ✅ | ✅ | ✅ | `ncclSend` / `ncclRecv` |
| Expert parallelism / MoE | ✅¹ | ✅¹ | ✅¹ | All-to-all via grouped `ncclSend` / `ncclRecv` |

`✅` = simulated end-to-end; `—` = the strategy does not exist in that framework. ¹ MoE is supported under a **load-balanced-experts** assumption (see [Limitations](#limitations)).

Phantora's estimates have been **validated against real-hardware ground truth** to within a few percent — see [Accuracy: Validated Configurations](#accuracy-validated-configurations).

## Requirements

- Linux x86_64
- An NVIDIA GPU for kernel profiling. The build targets compute capability 8.0 (e.g., A100, H200) and 9.0 (e.g., H100); other GPUs may work but are untested.
- CUDA 12.8 (the Docker image is based on `nvidia/cuda:12.8.0-devel-ubuntu22.04`)
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

Note: `pytorch/` is a git submodule pointing at a custom PyTorch branch (`2.9.1-phantora`) with the function tracer patched.

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

TorchTitan (≥ 0.2.0) loads a Hugging Face tokenizer **directory** (`tokenizer.json` + config) via `hf_assets_path`, not a tiktoken `.model` file. Place any HF tokenizer in `tests/assets/hf_tokenizer/` before starting — under payload-free simulation only its vocab size matters (it sets the embedding dimensions). The Llama3 tokenizer works, or any ungated one (e.g. `openai-community/gpt2`).

`run.sh` will pass its arguments to the corresponding scripts (`tests/test_{megatron,deepspeed,torchtitan}.py`). Each entry in the [model-preset table](#available-model-presets) links to a ready-made launcher you can run this way.

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

## Limitations

### Control flow must be data-independent

Phantora simulates GPU computation and communication, but the tensor values produced by simulated kernels are arbitrary. **Your training script's control flow must therefore not depend on the contents of GPU tensors.** Any branch that reads a value out of a tensor — for example, an early-exit on a loss threshold, a gradient-norm check, or a NaN/inf rescue path — will see garbage data and may follow a path that does not match real execution. Loss values printed during simulation are also not meaningful.

Concretely:

- **Megatron**: gradient clipping must be disabled. It copies a norm to CPU and takes a `sqrt`, which can fault on the random GPU memory contents under simulation.
- **MoE routing** is inherently data-dependent: the router picks experts from logits, and the dispatch counts / all-to-all split sizes / permutation indices all follow from that choice. Under simulation those are garbage, so Phantora's MoE support assumes **load-balanced experts** and makes the dispatch *shapes* analytic and uniform instead (see [Payload-free NCCL simulation](#payload-free-nccl-simulation) below). Don't rely on simulated routing decisions or per-expert token counts being realistic.
- Avoid early-stopping logic or NaN/inf rescue paths in the iterations being simulated.
- Stick to control flow that depends only on hyperparameters, iteration counts, and configuration. This covers the common case in LLM pre-training.

### Payload-free NCCL simulation

Phantora models the timing and ordering of point-to-point transfers, but it does not transfer bytes between ranks. Framework paths that inspect activation values or other transferred tensor payloads need to use a CPU backend path for now. Two consequences worth calling out:

- **MoE assumes load-balanced experts.** Because expert routing reads garbage tensor values (see [Control flow must be data-independent](#control-flow-must-be-data-independent)), Phantora's framework shims in [`tests/phantora_utils.py`](tests/phantora_utils.py) replace the data-dependent dispatch sizing with the analytic *uniform* distribution: every expert receives an equal share of tokens. The expert all-to-all itself is still simulated, so this gives a faithful throughput/MFU estimate for a balanced workload, but it does not model routing imbalance, capacity overflow, or token dropping. Covered today: Megatron (EP, and TP+EP with sequence parallelism), DeepSpeed (`deepspeed.moe` expert-parallel all-to-all, and the Hugging Face gpt-oss architecture whose experts run locally per rank), and TorchTitan (qwen3 expert parallelism).
- **DeepEP — on the roadmap.** Some recent MoE training stacks (e.g., DeepSeek-style models) bypass NCCL entirely and use [DeepEP](https://github.com/deepseek-ai/DeepEP) for expert dispatch/combine. We plan to add a DeepEP interception layer so those stacks can be simulated as well.

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
| `ncclSend` (point-to-point) | ✅ Supported |
| `ncclRecv` (point-to-point) | ✅ Supported |


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
| `ncclCommCount`, `ncclCommCuDevice`, `ncclCommUserRank` | ✅ Supported |
| `ncclRedOpCreatePreMulSum`, `ncclRedOpDestroy` | ✅ Supported (PreMulSum modeled as sum) |

The full set of stubs lives in [`stub/nccl.c`](stub/nccl.c). Pull requests to expand NCCL coverage are very welcome.

## Contributing

Contributions of any size are welcome:

- **Ground-truth measurements** to expand the validation matrix — see the [Contributing ground truth](#contributing-ground-truth) checklist above.
- **NCCL coverage** — see the matrix in [Limited NCCL coverage](#limited-nccl-coverage). PRs that add payload-aware support for metadata exchanges, `ncclReduce`, `ncclBroadcast`, or any other unimplemented entry point are especially valuable.
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
