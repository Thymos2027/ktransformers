# SPDX-License-Identifier: Apache-2.0
"""Experimental TP1/NPU decode residency, installed into the isolated SGLang copy.

Physical-byte staging, stable slots and publish-after-fence are adapted from
vllm-project/vllm-ascend PR #18604 (Apache-2.0), commit f687e5b.
CPU MoE remains active for misses; promotion occurs outside graph replay.
"""
import json
import logging
import os
import threading
import time
from collections import OrderedDict

from .residency_policy import Policy, transact

logger = logging.getLogger(__name__)
ENABLED = os.environ.get('KT_DECODE_RESIDENCY', '0') == '1'
INTERVAL = int(os.environ.get('KT_RESIDENCY_INTERVAL', '32'))
BUDGET = int(os.environ.get('KT_RESIDENCY_BUDGET', '8'))
CACHE_BYTES = int(os.environ.get('KT_RESIDENCY_CACHE_MIB', '1024')) * 1024**2
NAMES = ('w13_weight', 'w13_weight_scale', 'w2_weight', 'w2_weight_scale')
_runtime = None


def emit(event, **fields):
    logger.info('[KT RESIDENCY] %s', json.dumps(dict(event=event, **fields), sort_keys=True))


def physical_bytes(tensor):
    import torch_npu
    return int(torch_npu.get_storage_size(tensor)) * tensor.element_size()


def signature(tensor):
    import torch_npu
    return (tuple(tensor.shape[1:]), str(tensor.dtype), torch_npu.get_npu_format(tensor))


class PreparedBundle:
    """Owned pinned bytes for one expert, including NZ padding and scale storage."""
    def __init__(self, tensors):
        import acl
        import torch
        self.data, self.layouts = {}, {}
        for name in NAMES:
            tensor = tensors[name]
            if tensor.shape[0] != 1:
                raise ValueError('Preparation requires exactly one expert')
            size = physical_bytes(tensor)
            host = torch.empty(size, dtype=torch.uint8, pin_memory=True)
            error = acl.rt.memcpy(host.data_ptr(), size,
                                  tensor.untyped_storage().data_ptr(), size, 2)
            if error:
                raise RuntimeError(f'Expert preparation D2H failed: {error}')
            self.data[name] = host
            self.layouts[name] = signature(tensor)
        self.bytes = sum(t.numel() for t in self.data.values())

    def validate(self, layer):
        for name in NAMES:
            target = getattr(layer, name)
            if signature(target) != self.layouts[name] or physical_bytes(target) != target.shape[0] * self.data[name].numel():
                raise RuntimeError(f'Unsupported physical expert stride for {name}')

    def copy_to(self, layer, slot):
        import acl
        import torch
        self.validate(layer)
        if not 0 <= slot < layer.w13_weight.shape[0]:
            raise ValueError('Invalid resident slot')
        stream = torch.npu.current_stream(layer.w13_weight.device)
        for name, host in self.data.items():
            target = getattr(layer, name)
            size = host.numel()
            destination = target.untyped_storage().data_ptr() + slot * size
            error = acl.rt.memcpy_async(destination, size, host.data_ptr(), size, 1, stream.npu_stream)
            if error:
                raise RuntimeError(f'Expert slot H2D failed: {error}')


def convert_expert(layer_id, expert, layer):
    """Convert a selected GGUF expert with the existing 910B AscendC kernel."""
    import numpy as np
    import torch
    import torch_npu
    from . import kt_stream_prefill as stream
    gate, up, down = stream._gguf_layer_blocks(layer_id)
    device = layer.w13_weight.device
    host13 = torch.empty((1, gate.shape[1] + up.shape[1], gate.shape[2]), dtype=torch.uint8, pin_memory=True)
    host2 = torch.empty((1, down.shape[1], down.shape[2]), dtype=torch.uint8, pin_memory=True)
    host13[:, :gate.shape[1]].copy_(torch.from_numpy(np.array(gate[expert:expert+1], copy=True)))
    host13[:, gate.shape[1]:].copy_(torch.from_numpy(np.array(up[expert:expert+1], copy=True)))
    host2.copy_(torch.from_numpy(np.array(down[expert:expert+1], copy=True)))
    w13 = torch_npu.npu_format_cast(torch.empty((1, *layer.w13_weight.shape[1:]), dtype=layer.w13_weight.dtype, device=device), 29)
    w2 = torch_npu.npu_format_cast(torch.empty((1, *layer.w2_weight.shape[1:]), dtype=layer.w2_weight.dtype, device=device), 29)
    _, hidden, intermediate, _ = stream._get_cfg()
    values = stream._mxfp4_convert_blk_fn()(host13.to(device, non_blocking=True), host2.to(device, non_blocking=True),
                                           hidden, intermediate, out_w13=w13, out_w2=w2)
    torch.npu.current_stream(device).synchronize()
    return dict(zip(NAMES, values))


class Runtime:
    def __init__(self):
        if INTERVAL < 8 or BUDGET < 1 or CACHE_BYTES < 0:
            raise ValueError('Invalid bounded decode-residency configuration')
        self.bindings = {}
        self.policy = Policy(budget=BUDGET)
        self.lock = threading.Lock()
        self.cache = OrderedDict()
        self.cache_bytes = 0
        self.active = False
        self.request = self.dispatches = self.epochs = self.promotions = 0
        self.cache_hits = self.cache_misses = self.transferred = 0
        self.resync = False

    def register(self, wrap, layer):
        if wrap.tp_rank != 0 or wrap.wrapper is None or wrap.wrapper.max_deferred_experts_per_token:
            raise RuntimeError('Decode residency requires TP1, CPU ownership and no deferred experts')
        if layer.w13_weight.device.type != 'npu' or wrap.num_gpu_experts != 32:
            raise RuntimeError('This experiment requires NPU K32')
        self.bindings[wrap.kt_config.layer_idx] = (wrap, layer)
        emit('registered', layer=wrap.kt_config.layer_idx)

    def observe(self, wrapper, hidden_states):
        if not self.active or hidden_states.shape[0] != 1 or not self.policy.slots:
            return
        from kt_kernel.experts_base import KExpertsCPUBuffer
        # The existing graph callback has already waited for D2H and CPU MoE.
        # Reading this buffer adds no device-to-host transfer or nested callback.
        buffers = KExpertsCPUBuffer.get_buffer(hidden_states, wrapper.num_experts_per_tok)
        slot = wrapper.layer_idx % KExpertsCPUBuffer.buffer_depth
        ids = buffers[1][slot].numpy().reshape(-1)
        with self.lock:
            self.policy.observe(wrapper.layer_idx, ids)

    def before(self, batch):
        import torch
        if batch.input_ids.device.type != 'npu':
            return
        decode = batch.forward_mode.is_decode()
        if not decode:
            # The experiment only measures full, single-request long-prefill batches.
            # Short functional probes retain ordinary hybrid execution.
            if self.bindings:
                torch.npu.synchronize()
            with self.lock:
                self.active = batch.input_ids.numel() >= 512
                self.resync = self.active
                self.request += 1
                self.dispatches = self.epochs = self.promotions = 0
                self.cache_hits = self.cache_misses = self.transferred = 0
                self.policy.slots = {}
            return
        if not self.active:
            return
        if batch.batch_size != 1 or batch.input_ids.numel() != 1:
            raise RuntimeError('Decode residency is limited to ordinary single-request decode')
        if self.resync:
            torch.npu.synchronize()
            if sorted(self.bindings) != list(range(43)):
                raise RuntimeError('Not all 43 MoE layers registered')
            # Prefill top-K selection always assigns slots in ascending expert order.
            slots = {index: wrap.wrapper.gpu_experts_mask.numpy().nonzero()[0].tolist()
                     for index, (wrap, _) in self.bindings.items()}
            if any(len(ids) != 32 for ids in slots.values()):
                raise RuntimeError('Prefill resident masks do not have K32 entries')
            with self.lock:
                self.policy.reset(slots)
                self.resync = False
            emit('request', request=self.request, interval=INTERVAL, budget=BUDGET)
        if self.dispatches and self.dispatches % INTERVAL == 0:
            self.update()
        self.dispatches += 1

    def prepare(self, item):
        key = (item.layer, item.new)
        if key in self.cache:
            self.cache_hits += 1
            bundle = self.cache.pop(key)
            self.cache[key] = bundle
            return bundle
        self.cache_misses += 1
        layer = self.bindings[item.layer][1]
        tensors = convert_expert(item.layer, item.new, layer)
        bundle = PreparedBundle(tensors)
        bundle.validate(layer)
        if bundle.bytes <= CACHE_BYTES:
            while self.cache_bytes + bundle.bytes > CACHE_BYTES:
                _, old = self.cache.popitem(last=False)
                self.cache_bytes -= old.bytes
            self.cache[key] = bundle
            self.cache_bytes += bundle.bytes
        return bundle

    def update(self):
        import torch
        from . import kt_stream_prefill as stream
        started = time.perf_counter()
        # A full step boundary: drain compute/side streams and their CPU callbacks.
        # No resident tensor or CPU mask changes while an old step can still read it.
        torch.npu.synchronize()
        with self.lock:
            if set(self.policy.steps.values()) != {self.dispatches}:
                raise RuntimeError(f'Incomplete graph route observations: {self.policy.steps}')
            plan = self.policy.plan()
        copied = time.perf_counter()
        pointers = {i: tuple(getattr(layer, n).data_ptr() for n in NAMES)
                    for i, (_, layer) in self.bindings.items()}

        def copy(item, bundle):
            bundle.copy_to(self.bindings[item.layer][1], item.slot)
            self.transferred += bundle.bytes

        def publish(items):
            with self.lock:
                self.policy.accept(items)
                for index in {item.layer for item in items}:
                    wrap, _ = self.bindings[index]
                    stream._set_resident_masks(wrap, torch.tensor(self.policy.slots[index], dtype=torch.long),
                                               32, self.policy.experts)
            torch.npu.synchronize()

        if plan:
            transact(plan, self.prepare, copy, torch.npu.synchronize, publish)
        if any(tuple(getattr(layer, n).data_ptr() for n in NAMES) != pointers[index]
               for index, (_, layer) in self.bindings.items()):
            raise RuntimeError('Captured resident addresses changed')
        self.epochs += 1
        self.promotions += len(plan)
        emit('update', request=self.request, completed_decodes=self.dispatches,
             epoch=self.epochs, promotions=len(plan), promotions_total=self.promotions,
             route_choices=self.policy.routes, resident_hits=self.policy.hits,
             cache_hits=self.cache_hits, cache_misses=self.cache_misses, cache_bytes=self.cache_bytes,
             transfer_bytes=self.transferred, drain_ms=1000*(copied-started),
             update_ms=1000*(time.perf_counter()-copied),
             replacements=[dict(layer=p.layer, slot=p.slot, old=p.old, new=p.new) for p in plan])


def runtime():
    global _runtime
    if _runtime is None:
        _runtime = Runtime()
    return _runtime


def register(wrap, layer):
    if ENABLED:
        runtime().register(wrap, layer)


def before_forward(batch):
    if ENABLED:
        runtime().before(batch)


def record_decode(wrapper, hidden_states):
    if ENABLED:
        try:
            runtime().observe(wrapper, hidden_states)
        except Exception:
            # Host callback exceptions may otherwise be swallowed by torch_npu.
            logger.exception('Decode residency route observation failed; stopping worker')
            os._exit(1)
