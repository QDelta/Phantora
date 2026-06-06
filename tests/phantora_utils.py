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

    def _positive_norm(norm):
        if torch.is_tensor(norm):
            return torch.where(norm > 0, norm, torch.ones_like(norm))
        return norm if norm > 0 else 1.0

    if not getattr(ds_bf16.get_global_norm_of_tensors, "_phantora_positive_norm", False):
        # Patch target: deepspeed.runtime.bf16_optimizer.get_global_norm_of_tensors.
        # Original: may return zero when Phantora does not preserve gradient values.
        # Replacement: keep the original norm unless it is non-positive.
        _orig_get_global_norm_of_tensors = ds_bf16.get_global_norm_of_tensors

        def get_global_norm_of_tensors(*args, **kwargs):
            return _positive_norm(_orig_get_global_norm_of_tensors(*args, **kwargs))

        get_global_norm_of_tensors._phantora_positive_norm = True
        ds_bf16.get_global_norm_of_tensors = get_global_norm_of_tensors

    if not getattr(ds_bf16.get_norm_with_moe_layers, "_phantora_positive_norm", False):
        # Patch target: deepspeed.runtime.bf16_optimizer.get_norm_with_moe_layers.
        # Original: may return zero after combining MoE/non-MoE norms.
        # Replacement: keep the original norm unless it is non-positive.
        _orig_get_norm_with_moe_layers = ds_bf16.get_norm_with_moe_layers

        def get_norm_with_moe_layers(*args, **kwargs):
            return _positive_norm(_orig_get_norm_with_moe_layers(*args, **kwargs))

        get_norm_with_moe_layers._phantora_positive_norm = True
        ds_bf16.get_norm_with_moe_layers = get_norm_with_moe_layers

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


def install_phantora_torchtitan_moe_patches() -> None:
    """Make TorchTitan (>=0.2.0) MoE runnable under Phantora's payload-free sim.

    Must run BEFORE the model is built (the dispatch hook and GroupedExperts are
    captured at parallelize/build time). Call from tests/test_torchtitan.py at
    import. No-op for dense models / non-MoE runs.

    Three patches, all assuming experts are LOAD BALANCED (uniform tokens/expert):

    1. Force GroupedExperts off the triton grouped-GEMM path. GroupedExperts.forward
       branches on ``self.use_grouped_mm``; the grouped-mm kernel is a triton GEMM
       whose compiled runtime needs CUDA driver symbols (cuFuncSetAttribute, ...) the
       Phantora stub does not export -> ImportError. The for-loop path uses plain
       torch matmul that Phantora already simulates.

    2. Replace ``ExpertParallel._token_dispatch``'s split sizing. The real code
       derives the expert all-to-all splits from ``num_tokens_per_expert`` (per-expert
       routing counts); under simulation those GPU values are garbage and ``.tolist()``
       of the derived splits yields garbage sizes -> the all-to-all allocates
       astronomically (OOM). Instead size the splits analytically from the real shape
       ``routed_input.shape[0]`` (each of ``num_experts`` experts gets ``total //
       num_experts``), as uniform Python int lists. The all-to-all itself still runs.

    3. Reproduce torchtitan's ``_permute`` analytically (``_analytic_permute``). The
       real one calls ``generate_permute_indices`` (another triton kernel) and does
       ``.item()``/``.tolist()`` on GPU count tensors (garbage); the analytic version
       builds the permutation on CPU from the load-balanced counts and returns the
       per-expert sizes on CPU so the for-loop expert path's ``.tolist()`` is real.

    (torchtitan's own --training.debug_moe_force_load_balance is not enough: it
    balances the assignment but the counts are still read from garbage GPU tensors.)
    """
    if os.environ.get("PHANTORA") is None:
        return
    try:
        import torchtitan.distributed.expert_parallel as ep_mod
        from torchtitan.distributed.expert_parallel import ExpertParallel
    except (ImportError, AttributeError):
        return
    if getattr(ExpertParallel, "_phantora_balanced", False):
        return

    # Force the for-loop expert path instead of the triton grouped-GEMM path.
    # GroupedExperts.forward branches on self.use_grouped_mm; the grouped-mm
    # kernel is a triton GEMM whose compiled runtime needs CUDA driver symbols
    # (cuFuncSetAttribute, ...) the Phantora stub does not export -> ImportError.
    # The for-loop path uses plain torch bmm/matmul that Phantora already
    # simulates. Wrap GroupedExperts.__init__ to clear the flag post-init.
    try:
        from torchtitan.models.moe.moe import GroupedExperts
        if not getattr(GroupedExperts, "_phantora_no_grouped_mm", False):
            _orig_ge_init = GroupedExperts.__init__

            def _ge_init(self, *a, **kw):
                _orig_ge_init(self, *a, **kw)
                self.use_grouped_mm = False

            GroupedExperts.__init__ = _ge_init
            GroupedExperts._phantora_no_grouped_mm = True
    except (ImportError, AttributeError):
        pass

    # Disable flex attention on every registered model flavor. TorchTitan's
    # FlexAttentionWrapper is a torch.compile(mode="max-autotune") flex_attention
    # -> triton kernels the Phantora stub can't load; and the block_causal path
    # builds a document mask (get_document_mask_mod) that does data-dependent
    # indexing on token values (e.g. eos positions). Forcing use_flex_attn=False
    # selects the standard causal SDPA path (which Phantora simulates), and
    # train.py then skips get_attention_masks entirely. model_args entries are
    # shared module-level instances, so mutating them here (before the model is
    # built) is picked up by the Trainer.
    try:
        from torchtitan.models import _supported_models
        from torchtitan.protocols.train_spec import get_train_spec
        for _name in _supported_models:
            try:
                spec = get_train_spec(_name)
            except Exception:
                continue
            for _ma in spec.model_args.values():
                if hasattr(_ma, "use_flex_attn"):
                    _ma.use_flex_attn = False
    except (ImportError, AttributeError):
        pass

    # Round-up helper (matches torchtitan.tools.utils._round_up) without importing
    # so the shim works regardless of internal moves.
    def _round_up(x, m):
        return ((x + m - 1) // m) * m

    def _analytic_permute(x, ep_degree, num_local_experts, per_expert):
        """Reproduce torchtitan's _permute analytically for the load-balanced
        case, avoiding generate_permute_indices (a triton kernel that needs CUDA
        driver symbols the Phantora stub lacks) AND avoiding any .item()/.tolist()
        on GPU tensors (garbage under payload-free sim).

        Mirrors torchtitan.models.moe.utils._permute + generate_permute_indices
        with all per-(rank,expert) token counts equal to ``per_expert``.
        """
        import torchtitan.models.moe.utils as _mu
        align = _mu.TOKEN_GROUP_ALIGN_SIZE_M
        num_ranks = ep_degree
        experts_per_rank = num_local_experts
        # vstack a zero row (target for -1 / padding indices), as _permute does.
        x = torch.vstack((x, x.new_zeros((x.shape[-1]))))
        input_shape = x.shape
        orig_rows = x.shape[0] - 1  # == num_experts * per_expert
        padded_max_len = _round_up(orig_rows + experts_per_rank * align, align)
        # m_sizes: per local expert, total tokens summed over ranks, padded to align.
        total_per_local = max(num_ranks * per_expert, align)
        m_size_val = _round_up(total_per_local, align)
        # CPU tensor so the for-loop expert path can .tolist() real values.
        m_sizes = torch.full((experts_per_rank,), m_size_val, dtype=torch.int32)
        # Build permuted_indices on CPU (real values), then move to x.device.
        permuted_indices = torch.full((padded_max_len,), -1, dtype=torch.int64)
        for e in range(experts_per_rank):
            write_start = e * m_size_val
            for r in range(num_ranks):
                i = r * experts_per_rank + e
                start_index = i * per_expert
                end = min(write_start + per_expert, padded_max_len)
                if end > write_start:
                    n = end - write_start
                    permuted_indices[write_start:end] = torch.arange(
                        start_index, start_index + n, dtype=torch.int64
                    )
                write_start += per_expert
        permuted_indices = permuted_indices.to(x.device)
        x = x[permuted_indices, :]
        return input_shape, x, permuted_indices, m_sizes

    def _balanced_token_dispatch(self, mod, inputs, device_mesh):
        routed_input, num_tokens_per_expert = inputs
        ep_degree = device_mesh.shape[0]
        num_experts = num_tokens_per_expert.shape[0]
        num_local_experts = num_experts // ep_degree
        total_routed = routed_input.shape[0]  # real shape (num_tokens * top_k locally)
        per_expert = total_routed // num_experts
        # Uniform, load-balanced splits as Python int lists (sum == total_routed
        # when total_routed is divisible by num_experts, which holds under balance).
        self.input_splits = [num_local_experts * per_expert] * ep_degree
        self.output_splits = [num_local_experts * per_expert] * ep_degree
        routed_input = ep_mod.all_to_all_single_autograd(
            routed_input, self.output_splits, self.input_splits, device_mesh.get_group()
        )
        (
            self.input_shape,
            routed_input,
            self.permuted_indices,
            num_tokens_per_expert_group,
        ) = _analytic_permute(
            routed_input, ep_degree, num_local_experts, per_expert
        )
        return routed_input, num_tokens_per_expert_group

    ExpertParallel._token_dispatch = _balanced_token_dispatch
    ExpertParallel._phantora_balanced = True


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


def install_phantora_megatron_moe_patches() -> None:
    """Make Megatron-core MoE runnable under Phantora's payload-free simulation.

    Used by tests/test_megatron.py when num_moe_experts is set. Assumes experts are
    LOAD BALANCED (each rank routes an equal share of tokens to every expert).

    Under simulation the GPU kernels never actually run, so *values* read back from
    any GPU tensor are garbage. MoE is the first path whose control flow depends on
    tensor values (the router decides token->expert; the dispatch counts, all-to-all
    split sizes and permutation index sets are all derived from that). The fix does
    NOT try to compute a correct routing on the GPU (impossible here); instead it
    makes the *shapes* analytic and correct — which is all the simulator needs,
    since PyTorch still computes output shapes on the CPU from input numel while the
    kernel data stays garbage. Megatron's collective calls themselves are untouched.

    Three patches:
      1. Bind ``moe_utils.te_general_gemm = None``: megatron-core 0.13.1 references
         it in the router gating without guarding on HAVE_TE, raising NameError when
         Transformer Engine is absent. With it None the router uses the torch GEMM.
      2. ``moe_utils.permute`` (and the copy imported into ``token_dispatcher``):
         the real code builds ``sorted_indices`` via ``masked_select`` of the
         garbage routing map, so its length is garbage and the following
         ``index_select`` allocates a multi-TB tensor. When ``num_out_tokens`` is
         known (dropless => num_tokens*topk), build ``sorted_indices`` with exactly
         that length instead (values irrelevant; only the shape drives the
         simulated permute).
      3. ``MoEAlltoAllTokenDispatcher.preprocess``: it derives input/output splits
         and per-expert counts from the garbage routing map and a count gather.
         Run the original (keeps those kernels in the timeline) but overwrite the
         split/count attributes with the analytic uniform values implied by load
         balancing, as real CPU tensors, so the expert all-to-all and the second
         permutation are sized correctly. The all-to-all itself is still simulated.
    """
    if os.environ.get("PHANTORA") is None:
        return
    try:
        import megatron.core.transformer.moe.moe_utils as moe_utils
        import megatron.core.transformer.moe.token_dispatcher as token_dispatcher
        from megatron.core.transformer.moe.token_dispatcher import (
            MoEAlltoAllTokenDispatcher,
        )
    except (ImportError, AttributeError):
        return

    # (1) Transformer-Engine-absence shim for the router gating GEMM.
    if not hasattr(moe_utils, "te_general_gemm"):
        moe_utils.te_general_gemm = None

    # (2) Correct-length permutation index set (shape, not values).
    if not getattr(moe_utils, "_phantora_permute_patched", False):
        _orig_permute = moe_utils.permute

        def _phantora_permute(
            tokens, routing_map, probs=None, num_out_tokens=None,
            fused=False, drop_and_pad=False,
        ):
            if fused or drop_and_pad or num_out_tokens is None:
                return _orig_permute(
                    tokens, routing_map, probs, num_out_tokens, fused, drop_and_pad
                )
            n_out = int(num_out_tokens)
            # Right length, on the right device; contents are irrelevant under
            # payload-free simulation (the index_select kernel never really runs).
            sorted_indices = torch.empty(n_out, dtype=torch.long, device=tokens.device)
            permuted_input = tokens.index_select(0, sorted_indices)
            # permuted_probs must stay connected to `probs` in the autograd graph so
            # the router gating weight still receives a gradient (else Megatron's DDP
            # bucket asserts "grad not available"). n_out = num_tokens*topk <= probs
            # .numel() (topk <= num_experts), so a flattened slice has the right shape.
            permuted_probs = None if probs is None else probs.reshape(-1)[:n_out]
            return permuted_input, permuted_probs, sorted_indices

        moe_utils.permute = _phantora_permute
        # token_dispatcher imported `permute` into its own namespace.
        if hasattr(token_dispatcher, "permute"):
            token_dispatcher.permute = _phantora_permute
        moe_utils._phantora_permute_patched = True

    # (3) Analytic uniform dispatch metadata.
    if not getattr(MoEAlltoAllTokenDispatcher, "_phantora_balanced", False):
        _orig_preprocess = MoEAlltoAllTokenDispatcher.preprocess

        def _phantora_preprocess(self, routing_map):
            # Run the original for timeline fidelity (count gather etc.), then
            # overwrite the garbage-derived sizes with the load-balanced values.
            tokens_per_expert = _orig_preprocess(self, routing_map)
            num_tokens = routing_map.size(0)
            topk = self.config.moe_router_topk
            routed = num_tokens * topk
            ne, nle = self.num_experts, self.num_local_experts
            ep, tp = self.ep_size, self.tp_size
            per_expert = routed // ne  # tokens this rank sends to each expert
            dev = self.permute_idx_device
            long = torch.long
            self.num_out_tokens = routed
            if ep > 1 or tp > 1:
                # input_splits/output_splits are consumed by dist.all_to_all_single,
                # which requires List[int] (the dispatcher's DtoH helper only numpy-
                # ifies CUDA tensors; our CPU values must already be the final form).
                # Under load balance every rank is identical => symmetric all-to-all.
                self.input_splits = [per_expert * nle] * ep
                self.output_splits = [per_expert * nle] * ep
                # output_splits_tp sizes the TP all-gather in dispatch_postprocess;
                # it must sum to tp*ep*nle*per_expert (== num_global_tokens_per_local_expert
                # sum) so the gathered tensor matches the subsequent sort_chunks split.
                # Megatron: output_splits_tp[i] = sum_ep num_global_tokens_per_rank = ep*nle*per_expert.
                # Has .tolist() called on it -> keep it array-like (numpy).
                self.output_splits_tp = (
                    None if tp == 1
                    else torch.full((tp,), ep * nle * per_expert, dtype=long).numpy()
                )
                # [tp*ep, num_local_experts]: tokens each sender group sends to
                # each of this rank's local experts.
                self.num_global_tokens_per_local_expert = torch.full(
                    (tp * ep, nle), per_expert, dtype=long, device=dev
                )
                tokens_per_expert = torch.full(
                    (nle,), per_expert * tp * ep, dtype=long, device="cpu"
                )
            else:
                self.num_global_tokens_per_local_expert = torch.full(
                    (ne,), per_expert, dtype=long, device=dev
                )
                tokens_per_expert = torch.full((ne,), per_expert, dtype=long, device="cpu")
            return tokens_per_expert

        MoEAlltoAllTokenDispatcher.preprocess = _phantora_preprocess
        MoEAlltoAllTokenDispatcher._phantora_balanced = True

    # (4) EP+TP requires sequence parallelism, but the torch LayerNorm/RMSNorm
    # fallback (used without Apex/TE) asserts it "does not support sequence
    # parallel" in WrappedTorchNorm.__new__. The assert is conservative: the norm
    # is per-token over the hidden dim and is SP-agnostic (SP's comms live in the
    # surrounding linear layers). Let it build by clearing config.sequence_parallel
    # just for construction, then restoring it. No-op when SP is off (dense/EP=1).
    try:
        from megatron.core.transformer.torch_norm import WrappedTorchNorm
    except (ImportError, AttributeError):
        WrappedTorchNorm = None
    if WrappedTorchNorm is not None and not getattr(
        WrappedTorchNorm, "_phantora_sp_ok", False
    ):
        _orig_norm_new = WrappedTorchNorm.__new__

        def _phantora_norm_new(cls, config, *args, **kwargs):
            sp = config.sequence_parallel
            config.sequence_parallel = False
            try:
                return _orig_norm_new(cls, config, *args, **kwargs)
            finally:
                config.sequence_parallel = sp

        WrappedTorchNorm.__new__ = staticmethod(_phantora_norm_new)
        WrappedTorchNorm._phantora_sp_ok = True


def install_phantora_gpt_oss_patches() -> None:
    """Make HF gpt-oss MoE runnable under Phantora's payload-free simulation.

    Used by tests/test_deepspeed.py when --model gpt_oss.

    gpt-oss's GptOssExperts has two paths: a training/sparse path that does
    ``one_hot(router_indices)`` then ``.nonzero()`` to find hit experts and gathers
    tokens per expert (all data-dependent sizes), and a dense path that computes
    every expert for every token with FIXED shapes [num_experts, num_tokens,
    hidden] and ignores router_indices. Under simulation the router indices are
    garbage, so the sparse path's ``.nonzero()`` returns a garbage-sized tensor
    (numel overflow / multi-GB balloon). Force the dense path by flipping the
    module's ``training`` flag off just for the experts forward (autograd still
    flows, so backward/timing are unaffected).
    """
    if os.environ.get("PHANTORA") is None:
        return
    try:
        from transformers.models.gpt_oss import modeling_gpt_oss as gpt_oss
        GptOssExperts = gpt_oss.GptOssExperts
    except (ImportError, AttributeError):
        return
    if getattr(GptOssExperts, "_phantora_dense", False):
        return
    _orig_experts_forward = GptOssExperts.forward

    def _dense_experts_forward(self, hidden_states, router_indices=None, routing_weights=None):
        was_training = self.training
        self.training = False  # select the fixed-shape dense expert path
        try:
            return _orig_experts_forward(self, hidden_states, router_indices, routing_weights)
        finally:
            self.training = was_training

    GptOssExperts.forward = _dense_experts_forward
    GptOssExperts._phantora_dense = True


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
