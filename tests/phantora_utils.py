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
