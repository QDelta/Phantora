import ctypes
import os
import time as _time
import torch
from torch.profiler import (
    enable_function_tracer as _enable_function_tracer,
    disable_function_tracer as _disable_function_tracer,
)
import torch.nn.parameter
from torch.utils.data import Dataset

_PHANTORA_METADATA_PROCESS_GROUP_CACHE = {}


def get_phantora_metadata_process_group(group=None):
    """Return the cached Gloo process group for Phantora metadata messages."""
    import torch.distributed as dist

    if os.environ.get("PHANTORA") is None or not dist.is_initialized():
        return group
    if group is None:
        ranks = tuple(range(dist.get_world_size()))
    else:
        ranks = tuple(dist.get_process_group_ranks(group))
    if ranks not in _PHANTORA_METADATA_PROCESS_GROUP_CACHE:
        _PHANTORA_METADATA_PROCESS_GROUP_CACHE[ranks] = dist.new_group(list(ranks), backend="gloo")
    return _PHANTORA_METADATA_PROCESS_GROUP_CACHE[ranks]


if os.environ.get('PHANTORA') is None:
    def time() -> float:
        return _time.perf_counter()

    def time_pair() -> float:
        t = _time.perf_counter()
        return t, t
else:
    LIB = ctypes.CDLL('libcuda.so.1')
    _read_timer = LIB.read_timer
    LIB.get_time_double.restype = ctypes.c_double
    _get_time = LIB.get_time_double
    _perf_counter = _time.perf_counter

    def time() -> float:
        _read_timer()
        return _get_time()

    def time_pair() -> float:
        _read_timer()
        t = _get_time()
        t_wall = _perf_counter()
        return t, t_wall

    _time.perf_counter = time

    # seems cannot patch `assert_ints_same_as_other_ranks`
    # maybe due to decorator, but cannot reproduce in a mini example
    # patch `get_lst_from_rank0` instead
    try:
        def identity(x):
            return x
        import deepspeed.runtime.zero.utils
        deepspeed.runtime.zero.utils.get_lst_from_rank0 = identity
    except ImportError:
        pass

    # DTensor's OffsetBasedRNGTracker keeps the RNG state on the GPU
    # (`get_rng_state().to(device)`) and reads the philox offset back out of it.
    # Under Phantora, values round-tripped through GPU memory are arbitrary, so
    # the offset comes back non-multiple-of-4 and `set_rng_state` rejects it
    # ("offset must be a multiple of 4"), breaking FSDP2 (e.g. TorchTitan) weight
    # init. Keep the state on CPU so the real offset is read, and skip the
    # rank-0 state broadcast (a NCCL collective that moves no real data here).
    try:
        from torch.distributed.tensor import _random as _dtensor_random

        _orig_rng_tracker_init = _dtensor_random.OffsetBasedRNGTracker.__init__

        def _no_sync_rng_tracker_init(self, device_mesh, run_state_sync=True, *args, **kwargs):
            _orig_rng_tracker_init(self, device_mesh, False, *args, **kwargs)

        def _cpu_device_state(self):
            # Original returns `get_rng_state().to(self._device)`; drop the .to()
            # so the offset stays a real, CPU-side value under simulation.
            return self._device_handle.get_rng_state()

        _dtensor_random.OffsetBasedRNGTracker.__init__ = _no_sync_rng_tracker_init
        _dtensor_random.OffsetBasedRNGTracker._get_device_state = _cpu_device_state
    except (ImportError, AttributeError):
        pass


def install_phantora_deepspeed_patches() -> None:
    """Patch DeepSpeed PP tensor metadata exchange.

    Used by tests/test_deepspeed.py after deepspeed.init_distributed().

    Patch targets:
      - deepspeed.runtime.pipe.engine.PipelineEngine._send_tensor_meta
      - deepspeed.runtime.pipe.engine.PipelineEngine._recv_tensor_meta

    Original: allocate an int32 metadata tensor on self.device and send/recv it
    with deepspeed.runtime.pipe.p2p.
    Replacement: encode the same metadata layout on CPU and send/recv it through
    get_phantora_metadata_process_group().
    """
    if os.environ.get("PHANTORA") is None:
        return
    try:
        import torch.distributed as dist
        import deepspeed.runtime.bf16_optimizer as ds_bf16
        import deepspeed.runtime.pipe.engine as ds_engine
    except (ImportError, AttributeError):
        return

    get_phantora_metadata_process_group()

    BF16_Optimizer = ds_bf16.BF16_Optimizer
    if not getattr(BF16_Optimizer.step, "_phantora_skip_zero_norm_assert", False):
        # Patch target: deepspeed.runtime.bf16_optimizer.BF16_Optimizer.step.
        # Original: computes all_groups_norm, stores self._global_grad_norm, then
        # unconditionally asserts all_groups_norm > 0 before optional grad clipping.
        # Replacement: patch only that assert line so positive norm is required
        # only when DeepSpeed will actually clip gradients.
        import inspect
        import textwrap

        source = textwrap.dedent(inspect.getsource(BF16_Optimizer.step))
        old = "\n    assert all_groups_norm > 0.\n    if self.clip_grad > 0.:\n"
        new = (
            "\n    if self.clip_grad > 0.:\n"
            "        assert all_groups_norm > 0.\n"
            "    if self.clip_grad > 0.:\n"
        )
        if old not in source:
            raise RuntimeError("DeepSpeed BF16_Optimizer.step assert layout changed")
        namespace = dict(ds_bf16.__dict__)
        exec(compile(source.replace(old, new), ds_bf16.__file__, "exec"), namespace)
        step = namespace["step"]
        step._phantora_skip_zero_norm_assert = True
        BF16_Optimizer.step = step

    PipelineEngine = ds_engine.PipelineEngine
    if getattr(PipelineEngine, "_phantora_cpu_meta", False):
        return

    # Helper mirrors DeepSpeed's original metadata layout:
    # [kind, dtype_id, ndims, *shape] for tensors and
    # [kind=2, num_tensors, dtype_id, ndims, *shape, ...] for tuples.
    def _meta_list(self, buffer):
        if isinstance(buffer, torch.Tensor):
            return [0, self.DTYPE_TO_ID[buffer.dtype], len(buffer.size()), *buffer.size()]
        if isinstance(buffer, tuple):
            meta = [2, len(buffer)]
            for tensor in buffer:
                meta.extend([self.DTYPE_TO_ID[tensor.dtype], len(tensor.size()), *tensor.size()])
            return meta
        raise NotImplementedError(f"Could not send meta type {type(buffer)}")

    def _send_tensor_meta(self, buffer, recv_stage):
        meta = _meta_list(self, buffer)
        assert len(meta) <= ds_engine.TENSOR_META_SIZE
        meta_buffer = torch.zeros(ds_engine.TENSOR_META_SIZE, dtype=torch.int32)
        meta_buffer[:len(meta)] = torch.tensor(meta, dtype=torch.int32)
        dist.send(
            meta_buffer,
            dst=self.grid.stage_to_global(stage_id=recv_stage),
            group=get_phantora_metadata_process_group(),
        )

    def _recv_tensor_meta(self, send_stage):
        meta_buffer = torch.empty(ds_engine.TENSOR_META_SIZE, dtype=torch.int32)
        dist.recv(
            meta_buffer,
            src=self.grid.stage_to_global(stage_id=send_stage),
            group=get_phantora_metadata_process_group(),
        )
        recv_type = meta_buffer[0].item()
        if recv_type == 0:
            dtype = self.ID_TO_DTYPE[meta_buffer[1].item()]
            ndims = meta_buffer[2].item()
            shape = meta_buffer[3:3 + ndims].tolist()
            return self._allocate_or_extend_buffers(0, shape, dtype)
        if recv_type in (1, 2):
            buffers, offset = [], 2
            for idx in range(meta_buffer[1].item()):
                dtype = self.ID_TO_DTYPE[meta_buffer[offset].item()]
                ndims = meta_buffer[offset + 1].item()
                shape = meta_buffer[offset + 2:offset + 2 + ndims].tolist()
                offset += 2 + ndims
                buffers.append(self._allocate_or_extend_buffers(idx, shape, dtype))
            return tuple(buffers) if recv_type == 2 else buffers
        raise NotImplementedError(f"Could not receive type {recv_type}")

    # Only PipelineEngine metadata helpers are patched here.
    PipelineEngine._send_tensor_meta = _send_tensor_meta
    PipelineEngine._recv_tensor_meta = _recv_tensor_meta
    PipelineEngine._phantora_cpu_meta = True


def install_phantora_torchtitan_patches() -> None:
    """Patch TorchTitan PP shape-inference metadata exchange.

    Used by tests/test_torchtitan.py before constructing TorchTitan Trainer.

    Patch targets:
      - torchtitan.distributed.utils.init_distributed
      - torchtitan.models.llama3.infra.pipeline.PipelineStage

    Original: PipelineStage._shape_inference sends/receives meta shapes with
    dist.send_object_list/recv_object_list using group=self.group,
    device=self.device, and use_batch=True.
    Replacement: create a CPU/Gloo group after init_distributed(), then replace
    TorchTitan's imported PipelineStage with a subclass whose _shape_inference
    sends/receives the same objects through get_phantora_metadata_process_group().
    """
    if os.environ.get("PHANTORA") is None:
        return
    try:
        import torch.distributed as dist
        import torch.distributed.pipelining.stage as stage_mod
        from torch.distributed.pipelining import PipelineStage
        import torchtitan.distributed.utils as tt_dist_utils
        import torchtitan.models.llama3.infra.pipeline as tt_pipeline
    except (ImportError, AttributeError):
        return

    if not getattr(tt_dist_utils.init_distributed, "_phantora_cpu_meta", False):
        # Original init_distributed initializes the default process group.
        # Replacement calls it first, then creates the CPU group used below.
        _orig_init_distributed = tt_dist_utils.init_distributed

        def _init_distributed_with_cpu_group(*args, **kwargs):
            result = _orig_init_distributed(*args, **kwargs)
            get_phantora_metadata_process_group()
            return result

        _init_distributed_with_cpu_group._phantora_cpu_meta = True
        tt_dist_utils.init_distributed = _init_distributed_with_cpu_group

    if getattr(tt_pipeline.PipelineStage, "_phantora_cpu_meta", False):
        return

    class CpuMetaPipelineStage(PipelineStage):
        _phantora_cpu_meta = True

        def _shape_inference(self, args, kwargs=None):
            # Original: recv/send object-list metadata with self.group/self.device.
            # Replacement: recv/send the same metadata with get_phantora_metadata_process_group().
            if kwargs is None:
                kwargs = {}
            if self.is_first or self.stage_index_to_group_rank[self.stage_index - 1] == self.group_rank:
                args = stage_mod.tree_map_only(torch.Tensor, lambda x: x.to("meta"), args)
            else:
                objects = [None]
                pp_group = self.group or dist.distributed_c10d._get_default_group()
                dist.recv_object_list(
                    objects,
                    src=dist.get_global_rank(
                        pp_group,
                        self.stage_index_to_group_rank[self.stage_index - 1],
                    ),
                    group=get_phantora_metadata_process_group(),
                )
                args = objects[0]

            self.inputs_meta = args
            real_args = stage_mod.tree_map_only(
                torch.Tensor,
                lambda x: torch.zeros_like(x, device=self.device),
                args,
            )
            with torch.no_grad():
                outputs = self.submod(*real_args, **kwargs)
            if isinstance(outputs, torch.Tensor):
                outputs = [outputs]
            outputs_meta = tuple(
                stage_mod.tree_map_only(torch.Tensor, lambda x: x.to("meta"), outputs)
            )
            self._configure_outputs_meta(outputs_meta)

            if self.is_last or self.stage_index_to_group_rank[self.stage_index + 1] == self.group_rank:
                return outputs_meta

            pp_group = self.group or dist.distributed_c10d._get_default_group()
            dist.send_object_list(
                [outputs_meta],
                dst=dist.get_global_rank(
                    pp_group,
                    self.stage_index_to_group_rank[self.stage_index + 1],
                ),
                group=get_phantora_metadata_process_group(),
            )
            return tuple()

    tt_pipeline.PipelineStage = CpuMetaPipelineStage


def enable_function_tracer() -> None:
    if os.environ.get('PHANTORA') is not None:
        prefix = os.environ['PHANTORA_SOCKET_PREFIX']
        _enable_function_tracer(prefix + ".simulator.sock")

def disable_function_tracer() -> None:
    if os.environ.get('PHANTORA') is not None:
        _disable_function_tracer()

def enable_parameter_sharing() -> None:
    if os.environ.get('PHANTORA') is not None:
        torch.nn.parameter._enable_aggressive_sharing = True

def disable_parameter_sharing() -> None:
    if os.environ.get('PHANTORA') is not None:
        torch.nn.parameter._enable_aggressive_sharing = False

class RandomTokens(Dataset):
    def __init__(self, vocab_size, seq_len, size):
        self.len = size
        self.vocab_size = vocab_size
        self.seq_len = seq_len

    def __getitem__(self, index):
        data = torch.randint(0, self.vocab_size, (self.seq_len,))
        label = torch.randint(0, self.vocab_size, (self.seq_len,))
        return data, label

    def __len__(self):
        return self.len

class RandomImages(Dataset):
    def __init__(self, num_labels, shape, size):
        self.len = size
        self.num_labels = num_labels
        self.shape = shape
    
    def __getitem__(self, index):
        data = torch.randn(self.shape)
        label = torch.randint(0, self.num_labels, (1,))
        return data, label
    
    def __len__(self):
        return self.len

class RandomDiffusionImages(Dataset):
    def __init__(self, seq_len, shape, size):
        self.len = size
        self.seq_len = seq_len
        self.shape = shape
    
    def __getitem__(self, index):
        data = torch.randn(self.shape)
        embed = torch.randn((self.seq_len, 1024))
        return data, embed
    
    def __len__(self):
        return self.len
