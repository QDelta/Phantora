from phantora_utils import (
    RandomTokens,
    disable_function_tracer,
    enable_function_tracer,
    install_phantora_deepspeed_patches,
    install_phantora_gpt_oss_patches,
    time_pair,
)

import argparse
import copy
import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import deepspeed
from deepspeed.pipe import PipelineModule
from deepspeed.runtime.pipe.topology import PipeModelDataParallelTopology
from transformers import LlamaConfig, LlamaForCausalLM


class LlamaDecoderLayerPipe(nn.Module):
    # DeepSpeed PipelineModule runs a sequence of layer callables. Keep the
    # original HF decoder layer and only adapt its PipelineModule I/O shape.
    #
    # transformers >= 4.5x computes the rotary (cos, sin) once in
    # LlamaModel.forward and passes it to every layer as position_embeddings;
    # LlamaDecoderLayer no longer derives it from position_ids itself. The
    # pipeline split bypasses LlamaModel.forward, so each stage carries its own
    # rotary module and recomputes position_embeddings here (a per-layer copy
    # keeps each pipeline stage self-contained across ranks).
    def __init__(self, layer: nn.Module, rotary_emb: nn.Module):
        super().__init__()
        self.layer = layer
        self.rotary_emb = rotary_emb

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        seq_len = hidden_states.shape[1]
        position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        out = self.layer(
            hidden_states,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            use_cache=False,
        )
        # transformers 4.56's LlamaDecoderLayer.forward returns the hidden-states
        # tensor directly; older versions returned a 1-tuple. Indexing [0] on the
        # new tensor would slice off the batch dim, so unpack only when needed.
        return out[0] if isinstance(out, tuple) else out


def causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    # PipelineModule computes loss outside LlamaForCausalLM.forward(labels=...),
    # so reproduce HF causal-LM's shifted-token cross entropy here.
    shift_logits = logits[..., :-1, :].contiguous().float()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )


def build_pipeline_model(args: argparse.Namespace) -> PipelineModule:
    config = LlamaConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        intermediate_size=args.ffn_hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        max_position_embeddings=args.sequence_length,
        rope_theta=args.rope_theta,
        use_cache=False,
        # eager (not flash_attention_2): flash-attn derives seqlens from tensor
        # values (e.g. cu_seqlens via .max()), which are garbage under Phantora's
        # payload-free sim and overflow the kernel-time estimator. Attention is
        # simulated either way, so eager is equivalent here (matches gpt-oss).
        attn_implementation="eager",
    )

    from transformers.modeling_utils import no_init_weights

    dtype_orig = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        # Skip HF weight init (same as build_gpt_oss): under Phantora kernels
        # never execute and tensor values are garbage, so per-parameter init is
        # wasted CPU work. It also lets the empty, identically-shaped params share
        # storage under Phantora's parameter sharing, so each rank doesn't
        # materialize a full model copy in host RAM during 8-way construction.
        with no_init_weights():
            llama = LlamaForCausalLM(config)
    finally:
        torch.set_default_dtype(dtype_orig)
    if args.tensor_parallel_size > 1:
        llama = deepspeed.tp_model_init(
            llama,
            tp_size=args.tensor_parallel_size,
            dtype=torch.bfloat16,
        )
    layers = [
        llama.model.embed_tokens,
        *(
            LlamaDecoderLayerPipe(layer, copy.deepcopy(llama.model.rotary_emb))
            for layer in llama.model.layers
        ),
        llama.model.norm,
        llama.lm_head,
    ]
    topology = None
    if args.tensor_parallel_size > 1:
        topology = PipeModelDataParallelTopology(
            num_pp=args.pipeline_parallel_size,
            num_mp=args.tensor_parallel_size,
            num_dp=args.data_parallel_size,
        )
    return PipelineModule(
        layers=layers,
        num_stages=None if topology is not None else args.pipeline_parallel_size,
        topology=topology,
        loss_fn=causal_lm_loss,
        partition_method="uniform",
        dynamic_shape=False,
    )


def build_gpt_oss(args: argparse.Namespace) -> nn.Module:
    """Build a (real-architecture) gpt-oss MoE causal LM from a config.

    gpt-oss is a real HF MoE model: its experts run *locally* per rank (no expert
    all-to-all), so this gives real-model coverage of the MoE architecture under
    DeepSpeed DP/ZeRO. Dimensions are CLI-driven; the defaults are a small,
    runnable config (the real gpt-oss-20b is 24 layers / hidden 2880 / 32 experts /
    top-4 — set those via args on a machine with enough memory).
    """
    from transformers import GptOssConfig, GptOssForCausalLM
    from transformers.modeling_utils import no_init_weights

    install_phantora_gpt_oss_patches()
    config = GptOssConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        intermediate_size=args.ffn_hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads or args.num_attention_heads,
        head_dim=args.head_dim,
        num_local_experts=args.num_experts,
        num_experts_per_tok=args.experts_per_tok,
        max_position_embeddings=args.sequence_length,
        # eager attention avoids depending on flash-attn's API for gpt-oss's
        # sink/sliding-window attention; kernels are simulated either way.
        attn_implementation="eager",
        use_cache=False,
    )
    dtype_orig = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        # Under Phantora, kernels never execute and tensor values are garbage,
        # so HF's per-parameter weight initialization (_init_weights) is pure
        # wasted CPU work -- for gpt-oss-20b it dominates startup (~10 min for
        # the 21B model, 8 ranks contending). Skip it; the simulated forward is
        # value-independent anyway.
        with no_init_weights():
            model = GptOssForCausalLM(config)
    finally:
        torch.set_default_dtype(dtype_orig)
    return model


def run_module(args: argparse.Namespace, model: nn.Module, rank: int) -> None:
    """Generic (non-pipeline) DeepSpeed train loop for a full HF causal LM.

    Used for real-model presets like gpt-oss: forward(input_ids, labels) -> loss,
    DeepSpeed backward/step, under DP/ZeRO.
    """
    if rank == 0:
        print(f"Model size: {sum(p.numel() for p in model.parameters())}")

    model_engine, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=[p for p in model.parameters() if p.requires_grad],
        config={
            "train_micro_batch_size_per_gpu": args.micro_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation,
            "steps_per_print": 1000000,
            "optimizer": {"type": "AdamW", "params": {"torch_adam": True, "lr": 5e-5}},
            "bf16": {"enabled": True},
            "gradient_clipping": 0.0,
            "zero_optimization": {"stage": args.zero_stage},
            "wall_clock_breakdown": False,
        },
    )

    enable_function_tracer()
    duras, duras_wall = [], []
    try:
        for i in range(args.iterations):
            start, start_wall = time_pair()
            input_ids = torch.randint(
                0, args.vocab_size, (args.micro_batch_size, args.sequence_length),
                device=model_engine.device,
            )
            out = model_engine(input_ids=input_ids, labels=input_ids)
            model_engine.backward(out.loss)
            model_engine.step()
            torch.cuda.synchronize()
            end, end_wall = time_pair()
            if rank == 0:
                print(f"rank {rank} iter {i} time: {end - start:.2f} wall: {end_wall - start_wall:.2f}")
            duras.append(end - start)
            duras_wall.append(end_wall - start_wall)
    finally:
        disable_function_tracer()
        dist.destroy_process_group()

    print(f"Rank {rank} Time: {duras} Avg Time: {sum(duras[1:]) / (len(duras) - 1):.2f}\n", end="")
    print(f"Rank {rank} Wall: {duras_wall} Avg Wall: {sum(duras_wall[1:]) / (len(duras_wall) - 1):.2f}\n", end="")


def main(args: argparse.Namespace) -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank or 0))
    torch.cuda.set_device(local_rank)

    deepspeed.init_distributed()
    install_phantora_deepspeed_patches()

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # Real-model preset (e.g. gpt-oss): full HF model, generic DP/ZeRO loop.
    if args.model != "llama":
        builders = {"gpt_oss": build_gpt_oss}
        run_module(args, builders[args.model](args), rank)
        return

    model_parallel_size = args.pipeline_parallel_size * args.tensor_parallel_size
    if world_size % model_parallel_size != 0:
        raise ValueError(
            "WORLD_SIZE must be divisible by "
            "pipeline_parallel_size * tensor_parallel_size"
        )
    args.data_parallel_size = world_size // model_parallel_size
    train_batch_size = args.micro_batch_size * args.gradient_accumulation * (
        args.data_parallel_size
    )

    model = build_pipeline_model(args)
    if rank == 0:
        print(f"Model size: {sum(p.numel() for p in model.parameters())}")

    model_engine, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=[p for p in model.parameters() if p.requires_grad],
        config={
            "train_micro_batch_size_per_gpu": args.micro_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation,
            "train_batch_size": train_batch_size,
            "steps_per_print": 1000000,
            "optimizer": {"type": "AdamW", "params": {"torch_adam": True, "lr": 5e-5}},
            "bf16": {"enabled": True},
            "gradient_clipping": 0.0,
            "zero_optimization": {"stage": 0},
            "pipeline": {"pipe_partitioned": False, "grad_partitioned": False},
            "wall_clock_breakdown": False,
        },
    )

    dataset = RandomTokens(
        args.vocab_size,
        args.sequence_length,
        (args.iterations * args.gradient_accumulation + 2) * args.micro_batch_size,
    )
    data_iter = iter(DataLoader(dataset, batch_size=args.micro_batch_size))

    enable_function_tracer()
    duras, duras_wall = [], []
    try:
        for i in range(args.iterations):
            start, start_wall = time_pair()
            loss = model_engine.train_batch(data_iter=data_iter)
            torch.cuda.synchronize()
            end, end_wall = time_pair()
            if rank == 0:
                print(f"rank {rank} iter {i} loss: {float(loss.detach().cpu()):.4f} time: {end - start:.2f} wall: {end_wall - start_wall:.2f}")
            duras.append(end - start)
            duras_wall.append(end_wall - start_wall)
    finally:
        disable_function_tracer()
        dist.destroy_process_group()

    print(f"Rank {rank} Time: {duras} Avg Time: {sum(duras[1:]) / (len(duras) - 1):.2f}\n", end="")
    print(f"Rank {rank} Wall: {duras_wall} Avg Wall: {sum(duras_wall[1:]) / (len(duras_wall) - 1):.2f}\n", end="")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llama",
        choices=["llama", "gpt_oss"],
        help="llama = dense HF Llama via PipelineModule; gpt_oss = real HF MoE model "
             "(DP/ZeRO, experts local).")
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--zero_stage", type=int, default=0)
    # MoE knobs (gpt_oss): num_experts / experts_per_tok / head_dim; the FFN
    # intermediate size reuses ffn_hidden_size.
    parser.add_argument("--num_experts", type=int, default=8)
    parser.add_argument("--experts_per_tok", type=int, default=2)
    parser.add_argument("--head_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=32)
    parser.add_argument("--hidden_size", type=int, default=4096)
    parser.add_argument("--ffn_hidden_size", type=int, default=11008)
    parser.add_argument("--num_attention_heads", type=int, default=32)
    parser.add_argument("--num_key_value_heads", type=int, default=None,
        help="Number of KV heads for GQA (None = MHA, i.e., equal to num_attention_heads).")
    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    parser.add_argument("--sequence_length", type=int, default=4096)
    parser.add_argument("--micro_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--local_rank", type=int)
    main(parser.parse_args())
