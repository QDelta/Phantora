from phantora_utils import (
    time_pair,
    enable_function_tracer,
    disable_function_tracer,
)
import os
import functools
import torch
from torch.utils.data import DataLoader
from megatron.core import parallel_state
from megatron.core.datasets.blended_megatron_dataset_builder import (
    BlendedMegatronDatasetBuilder,
)
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig, MockGPTDataset
from megatron.core.datasets.megatron_tokenizer import MegatronTokenizer
from megatron.core.distributed import (
    DistributedDataParallelConfig as DDPConfig,
    DistributedDataParallel as DDP,
    finalize_model_grads,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.pipeline_parallel.schedules import forward_backward_no_pipelining
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.transformer_config import TransformerConfig


class NullTokenizer(MegatronTokenizer):
    def __init__(self, vocab_size):
        super().__init__(None, vocab_size=vocab_size)
        self._vocab_size_without_eod = int(vocab_size)
        self._eod_id = self._vocab_size_without_eod

    def tokenize(self, text):
        return [int(x) for x in text.split(" ")]

    def detokenize(self, ids):
        text = [str(x) for x in ids]
        return " ".join(text)

    def offsets(self, ids: list[int], text: str) -> list[int]:
        offsets, start_idx = [], 0
        for id_ in ids:
            offsets.append(start_idx)
            start_idx += 1 + len(str(id_))
        return offsets

    @property
    def vocab_size(self):
        return self._vocab_size_without_eod + 1

    @property
    def vocab(self):
        raise NotImplementedError

    @property
    def inv_vocab(self):
        raise NotImplementedError

    @property
    def cls(self):
        return -1

    @property
    def sep(self):
        return -1

    @property
    def mask(self):
        return -1

    @property
    def eod(self):
        return self._eod_id

    @property
    def additional_special_tokens_ids(self):
        return None


def get_model(
    tensor_parallel_size,
    num_layers,
    hidden_size,
    ffn_hidden_size,
    num_attention_heads,
    vocab_size,
    sequence_length,
    recompute_activations,
):
    transformer_config = TransformerConfig(
        tensor_model_parallel_size=tensor_parallel_size,
        num_layers=num_layers,
        hidden_size=hidden_size,
        ffn_hidden_size=ffn_hidden_size,
        num_attention_heads=num_attention_heads,
        perform_initialization=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        attention_backend=AttnBackend.flash,
        recompute_granularity="selective" if recompute_activations else None,
    )

    gpt_model = Float16Module(
        transformer_config,
        GPTModel(
            config=transformer_config,
            transformer_layer_spec=get_gpt_layer_local_spec(),
            vocab_size=vocab_size,
            max_sequence_length=sequence_length,
        ),
    )

    dp_config = DDPConfig(
        use_distributed_optimizer=True,
        overlap_grad_reduce=True,
        overlap_param_gather=True,
        bucket_size=40000000,
    )

    model = DDP(transformer_config, dp_config, gpt_model)
    transformer_config.no_sync_func = model.no_sync
    transformer_config.finalize_model_grads_func = finalize_model_grads

    return model


def get_optimizer(model):
    optim_config = OptimizerConfig(
        optimizer="adam",
        lr=5e-5,
        bf16=True,
        use_distributed_optimizer=True,
        clip_grad=0.0,  # no grad clipping because norm depends on communication and can cause error in simulation
    )

    optim = get_megatron_optimizer(optim_config, [model], use_gloo_process_groups=False)

    return optim


def get_train_data_iterator(vocab_size, micro_batch_size):
    config = GPTDatasetConfig(
        random_seed=42,
        sequence_length=4096,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        tokenizer=NullTokenizer(vocab_size-1),
    )
    config.mock = True
    # seems MockGPTDataset hardcoded sequence length to 4096
    datasets = BlendedMegatronDatasetBuilder(
        MockGPTDataset, [None, None, None], lambda: True, config
    ).build()
    train_dataloader = DataLoader(datasets[0], batch_size=micro_batch_size)
    return iter(train_dataloader)


def forward_step_func(device, data_iterator, model):
    def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
        loss_mask = loss_mask.view(-1).to(output_tensor.dtype)
        loss = torch.sum(output_tensor.view(-1) * loss_mask) / loss_mask.sum()
        return loss, {"lm-loss": loss}

    data = next(data_iterator)
    tokens = data["tokens"].to(device)
    attention_mask = data["attention_mask"].to(device)
    position_ids = data["position_ids"].to(device)
    labels = data["labels"].to(device)
    loss_mask = data["loss_mask"].to(device)

    output_tensor = model(tokens, position_ids, attention_mask, labels=labels)

    return output_tensor, functools.partial(loss_func, loss_mask)


def main(
    tensor_parallel_size,
    num_layers,
    hidden_size,
    ffn_hidden_size,
    num_attention_heads,
    vocab_size,
    micro_batch_size,
    gradient_accumulation,
    recompute_activations,
    iterations,
):
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.cuda.memory.reset_peak_memory_stats(device)

    torch.distributed.init_process_group(
        world_size=world_size, rank=rank, device_id=device
    )
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=tensor_parallel_size
    )

    model_parallel_cuda_manual_seed(42)

    model = get_model(
        tensor_parallel_size,
        num_layers,
        hidden_size,
        ffn_hidden_size,
        num_attention_heads,
        vocab_size,
        4096,
        recompute_activations
    )
    model = model.to(device)
    model.train()
    if rank == 0:
        print(f"Model size: {sum(p.numel() for p in model.parameters())}")
    optim = get_optimizer(model)
    train_iterator = get_train_data_iterator(vocab_size, micro_batch_size)

    duras = []
    duras_wall = []
    for i in range(iterations):
        start, start_wall = time_pair()
        model.zero_grad_buffer()
        optim.zero_grad()
        forward_backward_no_pipelining(
            forward_step_func=functools.partial(forward_step_func, device),
            data_iterator=train_iterator,
            model=model,
            num_microbatches=gradient_accumulation,
            seq_length=4096,
            micro_batch_size=micro_batch_size,
            forward_only=False,
        )
        optim.step()
        torch.cuda.synchronize()
        end, end_wall = time_pair()
        print(f"rank {rank} iter {i} time: {end - start:.2f} wall: {end_wall - start_wall:.2f}\n", end="")
        duras.append(end - start)
        duras_wall.append(end_wall - start_wall)

    peak_vram_mib = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    print(f"Rank {rank} Time: {duras} Avg Time: {sum(duras[1:]) / (iterations - 1):.2f}\n", end="")
    print(f"Rank {rank} Peak: {peak_vram_mib:<.2f}MiB\n", end="")
    print(f"Rank {rank} Wall: {duras_wall} Avg Wall: {sum(duras_wall[1:]) / (iterations - 1):.2f}\n", end="")
    parallel_state.destroy_model_parallel()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor_parallel_size", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=32)
    parser.add_argument("--hidden_size", type=int, default=4096)
    parser.add_argument("--ffn_hidden_size", type=int, default=11008)
    parser.add_argument("--num_attention_heads", type=int, default=32)
    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--micro_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation", type=int, default=1)
    parser.add_argument("--recompute_activations", action="store_true")
    parser.add_argument("--iterations", type=int, default=4)
    args = parser.parse_args()

    enable_function_tracer()
    main(
        tensor_parallel_size=args.tensor_parallel_size,
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        ffn_hidden_size=args.ffn_hidden_size,
        num_attention_heads=args.num_attention_heads,
        vocab_size=args.vocab_size,
        micro_batch_size=args.micro_batch_size,
        gradient_accumulation=args.gradient_accumulation,
        recompute_activations=args.recompute_activations,
        iterations=args.iterations,
    )
    disable_function_tracer()
