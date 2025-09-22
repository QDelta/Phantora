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
