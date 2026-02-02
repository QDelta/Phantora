from phantora_utils import (
    time_pair,
    enable_function_tracer,
    disable_function_tracer,
    RandomTokens,
)
import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import LlamaConfig, LlamaForCausalLM


def build_model(
    device,
    num_layers,
    hidden_size,
    ffn_hidden_size,
    num_attention_heads,
    vocab_size,
    seq_len,
):
    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=ffn_hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_attention_heads,
        max_position_embeddings=seq_len,
    )
    config._attn_implementation = "flash_attention_2"

    dtype_orig = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    with torch.device("meta"):
        model = LlamaForCausalLM(config)
    model = model.to_empty(device=device)
    torch.set_default_dtype(dtype_orig)
    return model


def main(
    num_layers,
    hidden_size,
    ffn_hidden_size,
    num_attention_heads,
    vocab_size,
    seq_len,
    micro_batch_size,
    iterations,
):
    from colossalai import launch_from_torch
    from colossalai.booster import Booster
    from colossalai.booster.plugin import TorchDDPPlugin

    launch_from_torch()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.cuda.memory.reset_peak_memory_stats(device)

    model = build_model(
        device=device,
        num_layers=num_layers,
        hidden_size=hidden_size,
        ffn_hidden_size=ffn_hidden_size,
        num_attention_heads=num_attention_heads,
        vocab_size=vocab_size,
        seq_len=seq_len,
    )
    model.train()

    if rank == 0:
        print(f"Model size: {sum(p.numel() for p in model.parameters())}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    dataset = RandomTokens(vocab_size, seq_len, iterations * micro_batch_size)
    data_loader = DataLoader(dataset, batch_size=micro_batch_size)

    booster = Booster(plugin=TorchDDPPlugin())
    model, optimizer, _, data_loader, _ = booster.boost(
        model, optimizer, dataloader=data_loader
    )

    duras = []
    duras_wall = []
    for step, (source, label) in enumerate(data_loader):
        if step >= iterations:
            break
        start, start_wall = time_pair()
        source = source.to(device)
        label = label.to(device)
        loss = model(source, labels=label).loss
        booster.backward(loss, optimizer)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        end, end_wall = time_pair()
        print(
            f"rank {rank} iter {step} time: {end - start:.2f} wall: {end_wall - start_wall:.2f}\n",
            end="",
        )
        duras.append(end - start)
        duras_wall.append(end_wall - start_wall)

    peak_vram_mib = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    if len(duras) > 1:
        avg_time = sum(duras[1:]) / (len(duras) - 1)
        avg_wall = sum(duras_wall[1:]) / (len(duras_wall) - 1)
    elif len(duras) == 1:
        avg_time = duras[0]
        avg_wall = duras_wall[0]
    else:
        avg_time = 0.0
        avg_wall = 0.0
    print(f"Rank {rank} Time: {duras} Avg Time: {avg_time:.2f}\n", end="")
    print(f"Rank {rank} Peak: {peak_vram_mib:<.2f}MiB\n", end="")
    print(f"Rank {rank} Wall: {duras_wall} Avg Wall: {avg_wall:.2f}\n", end="")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_layers", type=int, default=32)
    parser.add_argument("--hidden_size", type=int, default=4096)
    parser.add_argument("--ffn_hidden_size", type=int, default=11008)
    parser.add_argument("--num_attention_heads", type=int, default=32)
    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--sequence_length", type=int, default=4096)
    parser.add_argument("--micro_batch_size", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=4)
    args = parser.parse_args()
    enable_function_tracer()
    try:
        main(
            num_layers=args.num_layers,
            hidden_size=args.hidden_size,
            ffn_hidden_size=args.ffn_hidden_size,
            num_attention_heads=args.num_attention_heads,
            vocab_size=args.vocab_size,
            seq_len=args.sequence_length,
            micro_batch_size=args.micro_batch_size,
            iterations=args.iterations,
        )
    finally:
        disable_function_tracer()
