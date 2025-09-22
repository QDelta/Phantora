from phantora_utils import (
    enable_function_tracer, disable_function_tracer
)
import torch
from torchtitan.tools.logging import init_logger
from torchtitan.config_manager import ConfigManager
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

    enable_function_tracer()

    try:
        trainer = Trainer(config)
        trainer.train()
    finally:
        if trainer:
            trainer.close()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    disable_function_tracer()