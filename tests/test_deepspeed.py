from phantora_utils import (
    time_pair,
    enable_function_tracer,
    disable_function_tracer,
    RandomTokens,
)
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from torch.utils.data import DataLoader
import torch.distributed as dist
import deepspeed

def main(
    local_rank,
    num_layers,
    hidden_size,
    ffn_hidden_size,
    num_attention_heads,
    vocab_size,
    seq_len,
    batch_size,
    local_batches
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
    with torch.device('meta'):
        model = LlamaForCausalLM(config)
    model = model.to_empty(device=local_rank)
    torch.set_default_dtype(dtype_orig)
    print(f"Model size: {sum(p.numel() for p in model.parameters())}")

    dataset = RandomTokens(config.vocab_size, seq_len, local_batches * batch_size)
    data_loader = DataLoader(dataset, batch_size=batch_size)

    deepspeed.init_distributed()
    model_engine, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config={
            "train_micro_batch_size_per_gpu": batch_size,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "torch_adam": True,
                    "lr": 5e-5,
                },
            },
            "bf16": {
                "enabled": True,
            },
            "zero_optimization": {
                "stage": 3,
                "overlap_comm": True,
                "allgather_bucket_size": 2e8,
                "reduce_bucket_size": 2e8,
            },
        },
    )

    enable_function_tracer()
    duras = []
    duras_wall = []
    for source, label in data_loader:
        start, start_wall = time_pair()
        source = source.to(local_rank)
        label = label.to(local_rank)
        loss = model_engine(source, labels=label).loss
        model_engine.backward(loss)
        model_engine.step()
        loss.cpu()  # trigger sync
        end, end_wall = time_pair()
        print(f"time: {end - start:.2f} wall: {end_wall - start_wall:.2f}\n", end="")
        duras.append(end - start)
        duras_wall.append(end_wall - start_wall)
    dist.destroy_process_group()
    disable_function_tracer()

    output = f"Time: {duras} Avg time: {sum(duras[1:]) / (len(duras) - 1)}\n"
    output += (
        f"Wall: {duras_wall} Avg wall: {sum(duras_wall[1:]) / (len(duras_wall) - 1)}\n"
    )
    print(output, end="")


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
    parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()
    main(
        args.local_rank,
        args.num_layers,
        args.hidden_size,
        args.ffn_hidden_size,
        args.num_attention_heads,
        args.vocab_size,
        args.sequence_length,
        args.micro_batch_size,
        args.iterations
    )
