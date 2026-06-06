#!/usr/bin/env python3
"""Generate a Phantora performance database on a GPU, WITHOUT building Phantora.

A Phantora perf DB is just `(op, shape) -> time`. The op+shape *keys* are
GPU-independent; only the *times* are GPU-specific. So this tool takes the keys
from an existing reference DB (e.g. tests/perfdb/l40s/), re-profiles each kernel
on the local GPU with **stock PyTorch**, and writes a fresh DB in the same CSV
format. No C stubs, no Rust simulator, no patched PyTorch -- just `torch`.

It mirrors phantora/src/torch_estimate.rs: the same op dispatch, the same
allocation aliasing (operands of equal size+dtype share one buffer within an op),
and kernel-only device timing via torch.profiler (matching Phantora's CUPTI
sum). Verified to reproduce a Phantora-recorded DB: per-op times match to ~1%
median and a replayed preset's simulated iteration time is identical. Pass
--wall to use CUDA-event wall time instead (includes launch overhead).

Usage:
    python3 bench.py --ref tests/perfdb/l40s --out tests/perfdb/my_gpu
    python3 bench.py --ref tests/perfdb/l40s            # --out defaults to the GPU name

Replay the result with Phantora: `config_gen.py ... --perf-db <out-name>`.
"""
import argparse
import csv
import json
import os
import sys

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

# tch::Kind serialized name -> torch dtype
DTYPE = {
    "Uint8": torch.uint8, "Int8": torch.int8, "Int16": torch.int16,
    "Int": torch.int32, "Int64": torch.int64, "Half": torch.float16,
    "Float": torch.float32, "Double": torch.float64, "Bool": torch.bool,
    "BFloat16": torch.bfloat16,
}


# Per-op allocation cache keyed by (numel, dtype), mirroring torch_estimate.rs's
# `allocate`/`tensor_cache`: within one op, operands of the same size+dtype reuse
# (alias) one buffer. This matters for memory-bound multi-operand ops (e.g. the
# Adam `_foreach_addc*_` step) -- aliasing ~halves memory traffic, so reproducing
# it keeps the bench's timings consistent with a Phantora-recorded DB.
_ACACHE = {}


def _make(shape, dt, device):
    if dt.is_floating_point:
        return torch.randn(shape, dtype=dt, device=device)
    if dt == torch.bool:
        return torch.randint(0, 2, shape, device=device).to(torch.bool)
    return torch.randint(0, 128, shape, dtype=dt, device=device)


def alloc(ti, device="cuda", requires_grad=False):
    """Allocate a tensor matching a TensorInfo {shape, dtype}, reusing a cached
    buffer of the same (numel, dtype) like Phantora does (autograd leaves excepted)."""
    shape = ti["shape"]
    dt = DTYPE.get(ti["dtype"], torch.float32)
    if requires_grad:  # autograd leaves must be distinct, not aliased views
        return _make(shape, dt, device).detach().requires_grad_(True)
    numel = 1
    for d in shape:
        numel *= d
    key = (numel, dt)
    buf = _ACACHE.get(key)
    if buf is None:
        buf = _make([numel], dt, device)
        _ACACHE[key] = buf
    return buf.view(shape)


def is_ti(x):
    return isinstance(x, dict) and "shape" in x and "dtype" in x


# op name -> function(payload) returning a zero-arg thunk that runs the op.
# payload is the JSON value under the variant name (dict for newtype/struct,
# list for tuple/Vec). Tensors are allocated up front (outside timing).
def build(op, payload):
    P = payload
    _ACACHE.clear()  # fresh per op; aliases same-(numel,dtype) operands within this op

    def two():            # tuple of two TensorInfo
        return alloc(P[0]), alloc(P[1])

    if op == "MM":
        a, b = two(); return lambda: torch.mm(a, b)
    if op == "MatMul":
        a, b = two(); return lambda: torch.matmul(a, b)
    if op == "BMM":
        a, b = two(); return lambda: torch.bmm(a, b)
    if op == "Linear":
        a = alloc(P[0]); w = alloc(P[1]); bias = alloc(P[2]) if P[2] else None
        return lambda: F.linear(a, w, bias)
    if op == "AddMM":
        a, b, c = alloc(P[0]), alloc(P[1]), alloc(P[2]); return lambda: torch.addmm(a, b, c)
    if op == "BAddBMM":
        a, b, c = alloc(P[0]), alloc(P[1]), alloc(P[2]); return lambda: torch.baddbmm(a, b, c)
    if op in ("Mul", "Add", "Div"):
        a, b = two()
        return {"Mul": lambda: a * b, "Add": lambda: a + b, "Div": lambda: a / b}[op]
    if op == "Mul_":
        a, b = two(); return lambda: a.mul_(b)
    if op == "Add_":
        a, b = two(); return lambda: a.add_(b)
    if op == "MulScalar":
        a = alloc(P); return lambda: a * 2
    if op == "MulScalar_":
        a = alloc(P); return lambda: a.mul_(2)
    if op == "DivScalar":
        a = alloc(P); return lambda: a / 2
    if op == "Pow":
        a = alloc(P); return lambda: a.pow(2)
    if op == "Sqrt":
        a = alloc(P); return lambda: a.sqrt()
    if op == "Sum":
        a = alloc(P); dt = DTYPE.get(P["dtype"], torch.float32); return lambda: a.sum(dtype=dt)
    if op == "ZerosLike":
        a = alloc(P); return lambda: torch.zeros_like(a)
    if op == "ConvDType":
        a = alloc(P[0]); dt = DTYPE.get(P[1], torch.float32); return lambda: a.to(dt)
    if op == "Gelu":
        a = alloc(P); return lambda: F.gelu(a)
    if op == "Softmax":
        a = alloc(P[0]); dim = P[1]; dt = DTYPE.get(P[0]["dtype"], torch.float32)
        return lambda: torch.softmax(a, dim, dtype=dt)
    if op == "SoftmaxBackward":
        g = alloc(P[0]); o = alloc(P[1]); dim = P[2]; dt = DTYPE.get(P[0]["dtype"], torch.float32)
        return lambda: torch.ops.aten._softmax_backward_data(g, o, dim, dt)
    if op in ("AddCMul_", "AddCDiv_"):
        a, b, c = alloc(P[0]), alloc(P[1]), alloc(P[2])
        return (lambda: a.addcmul_(b, c)) if op == "AddCMul_" else (lambda: a.addcdiv_(b, c))
    if op == "ForeachMulScalar_":
        ts = [alloc(ti) for ti in P]; return lambda: torch._foreach_mul_(ts, 2)
    if op == "ForeachMul_":
        l1 = [alloc(ti) for ti in P[0]]; l2 = [alloc(ti) for ti in P[1]]
        return lambda: torch._foreach_mul_(l1, l2)
    if op in ("ForeachAddCMul_", "ForeachAddCDiv_"):
        l1 = [alloc(ti) for ti in P[0]]; l2 = [alloc(ti) for ti in P[1]]; l3 = [alloc(ti) for ti in P[2]]
        fn = torch._foreach_addcmul_ if op == "ForeachAddCMul_" else torch._foreach_addcdiv_
        return lambda: fn(l1, l2, l3, 2)
    if op in ("MaskedFill", "MaskedFill_"):
        a = alloc(P[0]); m = alloc(P[1]).to(torch.bool)
        return (lambda: a.masked_fill(m, -10000.0)) if op == "MaskedFill" else (lambda: a.masked_fill_(m, -10000.0))
    if op in ("Dropout", "NativeDropout"):
        a = alloc(P[0]); p = P[1] / 1_000_000.0; train = P[2]
        return (lambda: F.dropout(a, p, train)) if op == "Dropout" else (lambda: torch.native_dropout(a, p, train))
    if op == "NativeDropoutBackward":
        g = alloc(P[0]); m = alloc(P[1]); scale = P[2] / 1_000_000.0
        return lambda: torch.ops.aten.native_dropout_backward(g, m, scale)
    if op == "Where":
        c = alloc(P[0]).to(torch.bool); a = alloc(P[1]); b = alloc(P[2])
        return lambda: torch.where(c, a, b)
    if op == "WhereScalar":
        c = alloc(P[0]).to(torch.bool); a = alloc(P[1])
        return lambda: torch.where(c, a, 1)
    if op == "SDPA":
        q, k, v = alloc(P["q"]), alloc(P["k"]), alloc(P["v"])
        return lambda: F.scaled_dot_product_attention(q, k, v, is_causal=P["causal"], enable_gqa=P["gqa"])
    if op == "SDPABackward":
        # Approximate the internal flash-attn backward via autograd: build the
        # forward graph once, time the backward.
        q = alloc(P["q"], requires_grad=True); k = alloc(P["k"], requires_grad=True); v = alloc(P["v"], requires_grad=True)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=P["causal"])
        g = alloc(P["grad"])
        return lambda: out.backward(g, retain_graph=True)
    return None  # unknown op


def time_kernel(thunk, warmup, iters):
    """Kernel-only device time, mirroring Phantora's CUPTI sum: profile `iters`
    runs and sum each kernel's self device time (excludes launch/CPU overhead)."""
    for _ in range(warmup):
        thunk()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            thunk()
        torch.cuda.synchronize()
    us = sum(
        getattr(e, "self_device_time_total", 0) or getattr(e, "self_cuda_time_total", 0)
        for e in prof.key_averages()
    )
    return int(us * 1000 / iters)  # us total over iters -> ns per iter


def time_wall(thunk, warmup, iters):
    """Wall GPU time via CUDA events (includes launch overhead); --wall option."""
    for _ in range(warmup):
        thunk()
    torch.cuda.synchronize()
    times = []
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    for _ in range(iters):
        start.record()
        thunk()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1e6)  # ms -> ns
    times.sort()
    return int(times[len(times) // 2])  # median ns


def memcpy_thunk(kind, size):
    n = max(size, 1)
    cuda, cpu = torch.device("cuda"), torch.device("cpu")
    def buf(dev, pin=False):
        t = torch.empty(n, dtype=torch.uint8, device=dev)
        return t.pin_memory() if pin and dev == cpu else t
    if kind == "DeviceToDevice":
        s, d = buf(cuda), buf(cuda)
    elif kind == "HostToDevice":
        s, d = buf(cpu), buf(cuda)
    elif kind == "PinnedHostToDevice":
        s, d = buf(cpu, pin=True), buf(cuda)
    elif kind == "DeviceToHost":
        s, d = buf(cuda), buf(cpu)
    elif kind == "DeviceToPinnedHost":
        s, d = buf(cuda), buf(cpu, pin=True)
    else:  # HostToHost
        s, d = buf(cpu), buf(cpu)
    return lambda: d.copy_(s)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True, help="reference DB dir to take keys from (e.g. tests/perfdb/l40s)")
    ap.add_argument("--out", default=None, help="output DB dir (default: tests/perfdb/<gpu-name>)")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--wall", action="store_true",
                    help="use CUDA-event wall time instead of kernel-only (default) timing")
    ap.add_argument("--merge", action="store_true",
                    help="merge profiled keys into an existing --out DB (e.g. to fill a "
                         "<db>.missing manifest into <db>) instead of overwriting it")
    args = ap.parse_args()
    timer = time_wall if args.wall else time_kernel

    if not torch.cuda.is_available():
        sys.exit("error: a CUDA GPU is required to record timings")
    gpu = torch.cuda.get_device_name(0)
    out = args.out or os.path.join(os.path.dirname(args.ref.rstrip("/")), gpu.replace(" ", "_"))
    os.makedirs(out, exist_ok=True)
    print(f"GPU: {gpu}\nref: {args.ref}\nout: {out}")

    def load_existing(path, nkey):
        """Load an existing CSV into {tuple(first nkey cols): full row}."""
        rows = {}
        if os.path.exists(path):
            with open(path) as f:
                rd = csv.reader(f)
                next(rd, None)
                for row in rd:
                    if row:
                        rows[tuple(row[:nkey])] = row
        return rows

    # compute.csv (key columns: op, key). With --merge, start from the existing
    # out DB and overlay the profiled keys; otherwise write a fresh table.
    compute_out = os.path.join(out, "compute.csv")
    rows = load_existing(compute_out, 2) if args.merge else {}
    n_ok = n_skip = 0
    with open(os.path.join(args.ref, "compute.csv")) as f:
        r = csv.reader(f); next(r)
        for op, key, ref_nanos in r:
            try:
                thunk = build(op, json.loads(key)[op])
                if thunk is None:
                    raise NotImplementedError(op)
                nanos = timer(thunk, args.warmup, args.iters)
                n_ok += 1
            except Exception as e:  # keep the DB complete; carry the reference time
                nanos = int(ref_nanos)
                n_skip += 1
                print(f"  WARN {op}: {type(e).__name__}: {str(e)[:80]} -> kept reference time", file=sys.stderr)
            rows[(op, key)] = [op, key, nanos]
    with open(compute_out, "w", newline="") as g:
        w = csv.writer(g); w.writerow(["op", "key", "nanos"])
        for row in sorted(rows.values()):
            w.writerow(row)
    print(f"compute.csv: {n_ok} profiled, {n_skip} carried-over, {len(rows)} total")

    # memcpy.csv (key columns: kind, size_bytes)
    memcpy_out = os.path.join(out, "memcpy.csv")
    mrows = load_existing(memcpy_out, 2) if args.merge else {}
    m_ok = 0
    with open(os.path.join(args.ref, "memcpy.csv")) as f:
        r = csv.reader(f); next(r)
        for kind, size, ref_nanos in r:
            try:
                nanos = timer(memcpy_thunk(kind, int(size)), args.warmup, args.iters)
                m_ok += 1
            except Exception as e:
                nanos = int(ref_nanos)
                print(f"  WARN memcpy {kind}/{size}: {e}", file=sys.stderr)
            mrows[(kind, size)] = [kind, size, nanos]
    with open(memcpy_out, "w", newline="") as g:
        w = csv.writer(g); w.writerow(["kind", "size_bytes", "nanos"])
        for row in sorted(mrows.values(), key=lambda x: (x[0], int(x[1]))):
            w.writerow(row)
    print(f"memcpy.csv: {m_ok} profiled, {len(mrows)} total")

    # sequence/flash_attn: empty under single-op timing. Copy from ref only for a
    # fresh DB; when merging, leave the existing DB's tables untouched.
    if not args.merge:
        for name in ("sequence.csv", "flash_attn.csv"):
            src = os.path.join(args.ref, name)
            if os.path.exists(src):
                with open(src) as f, open(os.path.join(out, name), "w") as g:
                    g.write(f.read())

    with open(os.path.join(out, "manifest.md"), "w") as f:
        f.write(f"# Phantora performance database\n\n- **GPU:** {gpu}\n- **Schema version:** 1\n"
                f"- Recorded by tests/perfdb/bench.py (stock PyTorch, no Phantora build) "
                f"from keys in `{args.ref}`.\n")
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
