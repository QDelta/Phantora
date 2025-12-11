## What is Phantora?

Phantora is a *hybrid* GPU cluster simulator for ML system performance estimation.

For more details, please refer to our [paper](https://arxiv.org/abs/2505.01616).

Phantora is accepted by **NSDI 2026** 🎉

## Build Instructions

Clone the repository via `git`.

```bash
git clone https://github.com/QDelta/Phantora
cd Phantora
git submodule update --init --recursive
```

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

For TorchTitan, the `tokenizer.model` of Llama3 is needed, you can get it from its [huggingface repo](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct/blob/main/original/tokenizer.model). Place `tokenizer.model` in `tests/assets` before starting

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

## Citation

If you use Phantora for your research, please cite our [paper](https://arxiv.org/abs/2505.01616).

```bibtex
@misc{qin2025phantora,
  title="{Phantora: Maximizing Code Reuse in Simulation-based Machine Learning System Performance Estimation}",
  author={Jianxing Qin and Jingrong Chen and Xinhao Kong and Yongji Wu and Tianjun Yuan and Liang Luo and Zhaodong Wang and Ying Zhang and Tingjun Chen and Alvin R. Lebeck and Danyang Zhuo},
  year={2025},
  eprint={2505.01616},
  archivePrefix={arXiv},
  primaryClass={cs.DC},
  url={https://arxiv.org/abs/2505.01616},
}
```
