from phantora_utils import (
    disable_function_tracer,
    enable_function_tracer,
    install_phantora_torchtitan_patches,
    install_phantora_torchtitan_moe_patches,
    time_pair,
)

install_phantora_torchtitan_patches()
# Must run before the model is built (the MoE dispatch hook is captured at
# parallelize time); no-op for dense models / non-MoE runs.
install_phantora_torchtitan_moe_patches()

import torch
from torchtitan.tools.logging import init_logger
from torchtitan.config import ConfigManager  # torchtitan >= 0.2.0 (was torchtitan.config_manager)
from torchtitan.train import Trainer

if __name__ == '__main__':
    import sys
    if len(sys.argv) <= 1:
        args = ["--job.config_file=tests/test_torchtitan_llama3_8b.toml"]
    else:
        args = sys.argv[1:]

    init_logger()
    config_manager = ConfigManager()
    config = config_manager.parse_args(args)
    trainer = None

    try:
        trainer = Trainer(config)
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        duras, duras_wall = [], []
        train_step = trainer.train_step

        def timed_train_step(*args, **kwargs):
            start, start_wall = time_pair()
            result = train_step(*args, **kwargs)
            torch.cuda.synchronize()
            end, end_wall = time_pair()
            duras.append(end - start)
            duras_wall.append(end_wall - start_wall)
            print(f"rank {rank} iter {len(duras) - 1} time: {end - start:.2f} wall: {end_wall - start_wall:.2f}\n", end="")
            return result

        trainer.train_step = timed_train_step
        enable_function_tracer()
        trainer.train()
        if len(duras) > 1:
            avg_time = sum(duras[1:]) / (len(duras) - 1)
            avg_wall = sum(duras_wall[1:]) / (len(duras_wall) - 1)
        else:
            avg_time = duras[0] if duras else 0.0
            avg_wall = duras_wall[0] if duras_wall else 0.0
        print(f"Rank {rank} Time: {duras} Avg Time: {avg_time:.2f}\n", end="")
        print(f"Rank {rank} Wall: {duras_wall} Avg Wall: {avg_wall:.2f}\n", end="")
    finally:
        disable_function_tracer()
        if trainer:
            trainer.close()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
