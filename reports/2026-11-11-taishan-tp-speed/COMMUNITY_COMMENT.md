### Hardware follow-up: experimental TP2 CPU offload on Kunpeng 920 + 2× Ascend 910B

Thanks for the Ascend backend. I tested whether one DeepSeek-V4-Flash instance could use **two NPUs with TP=2 / EP=1**, retaining KT CPU-expert offload. These are hardware observations from 2026-11-11, using a fixed fork and an experimental opt-in patch.

**Setup:** dual-socket Kunpeng 920, 128 cores, four NUMA nodes, approximately 502 GiB RAM, and 2× Ascend 910B4-1 with 64 GiB HBM each. `npu-smi` reports `SYS` between the cards. CANN 9.0.0 / torch_npu 2.10.0; DeepSeek-V4-Flash W8A8 plus per-layer MXFP4 CPU experts. KT commit `c9873312643b615a717832946fba0346e69ab035`; [SGLang fork commit `ff46d660`](https://github.com/Pan-Boyi/sglang/tree/ff46d660d16cbf11ef393c80cc6b3ab7f6259780). Both TP settings used the same local environment snapshot.

**Functional checks:** in a separate source copy, the only changes were an opt-in exception to the TP guard for TP2/EP1 and diagnostic logging; the existing authors' CPU execution, expert partitioning, and reductions were retained. Two-rank HCCL all-reduce passed for 1, 1,024, and 262,144 FP32 elements. Short requests passed in eager mode and with batch=1 decode graphs captured on both ranks. Diagnostics across all 43 layers showed intermediate dimensions 1,024 per rank / 2,048 full, with the CPU expert executor owned only by rank 0. This is limited functional validation; numerical equivalence has not been established.

**KV sizing observation:** without an explicit total-token cap, weights loaded at about 21.93 GiB per rank, but automatic sizing selected `full_token=3766016` and later hit `torch.OutOfMemoryError` in `NPUDeepSeekV4IndexerPool._create_buffer()`. Adding `--max-total-tokens 16384` allowed startup; resolved pools were `full=16384, swa=1536, c4=4096, c128=128`. This is a workaround on the pinned fork; the budgeting root cause and applicability to SGLang main are unverified.

**Speed comparison:** graph mode for both TP settings, the same 32 NPU-resident experts per layer, 96 CPU inference threads, four threadpools, context limit 8,192, chunked prefill 512, total-token cap 16,384, and one request at a time. Four fresh server starts in **TP1a → TP2a → TP2b → TP1b** order; each start had one warmup and three measured requests per input length: 24 measured requests plus eight warmups. All performance responses reached 512 output tokens (`temperature=0`, `seed=20261111`, `ignore_eos=true`).

| Actual input / output tokens | TP1 decode | TP2 decode | Change | TP1 / TP2 total time | TP1 / TP2 TTFT |
| --- | ---: | ---: | ---: | ---: | ---: |
| 144 / 512 | 15.329 tok/s | 15.845 tok/s | +3.37% | 35.752 / 34.726 s | 2.423 / 2.474 s |
| 792 / 512 | 15.381 tok/s | 15.724 tok/s | +2.23% | 44.489 / 43.797 s | 11.275 / 11.296 s |

Values are pooled medians of six measured requests per TP/input, from two starts. Decode is `511 / (last_content_time - first_content_time)`; TTFT is client request to first nonempty SSE content. All requests had usage, `finish_reason=length`, and `[DONE]`. Five simple natural-stop tasks passed at each start (20 checks).

**Interpretation:** this fixed-expert configuration showed a small decode improvement and no TTFT improvement. Greedy outputs were repeatable within each TP setting but **different across TP1 and TP2**, so these are equal-input/equal-output-length measurements, not identical-output or numerically equivalent speedups. Only two starts per TP and two synthetic inputs were tested; no statistical significance, representative accuracy, long-context, concurrency, or sustained-stability claim is made. CPU experts still execute through rank 0; no component timings were collected to establish the bottleneck. Controlled shutdown produced exception traces and exit code 137 after serving completed, so graceful shutdown remains unvalidated.

Would an opt-in validation note be useful for this backend, and which numerical checks would you want before considering TP2 support? Reproduction parameters, the exact candidate diff, and individual measurements are included below.

<details>
<summary>Launch parameters and exact candidate patch</summary>

These commands assume the single-card tutorial environment is already built, the pinned SGLang source copy is selected on `PYTHONPATH`, and its NPU environment is initialized. Set `MODEL_PATH` to the W8A8 directory and `CPU_EXPERT_TEMPLATE` to the per-layer GGUF path template. CPU experts used: [per-layer MXFP4 package](https://modelscope.cn/models/kmon03f/DeepSeek-V4-Flash-KT-MXFP4-CPU-Experts). This is a configuration excerpt, not a standalone installer or a recommendation to bypass the default guard in production.

```bash
export HCCL_CONNECT_TIMEOUT=120 HCCL_EXEC_TIMEOUT=120
export SGLANG_SET_CPU_AFFINITY=0
export KT_PREFILL_STREAM=0 KT_DYNAMIC_RESIDENT=0 KT_SIDE_STREAM=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True TASK_QUEUE_ENABLE=1
ulimit -n 65536

launch_trial() {
  python3 -m sglang.launch_server \
    --model-path "$MODEL_PATH" --device npu --attention-backend ascend \
    --tensor-parallel-size "$1" --expert-parallel-size 1 --moe-a2a-backend none \
    --page-size 128 --quantization compressed-tensors \
    --disable-shared-experts-fusion --dtype bfloat16 --trust-remote-code \
    --disable-radix-cache --mem-fraction-static 0.81 \
    --context-length 8192 --max-prefill-tokens 8191 --chunked-prefill-size 512 \
    --max-total-tokens 16384 --random-seed 20261111 \
    --watchdog-timeout 600 --kt-method LLAMAFILE --kt-num-gpu-experts 32 \
    --kt-weight-path "$CPU_EXPERT_TEMPLATE" --kt-threadpool-count 4 --kt-cpuinfer 96 \
    --max-running-requests 1 --host 127.0.0.1 --port 18082
}

# Separate server starts; stop each before launching the next.
ASCEND_RT_VISIBLE_DEVICES=0 launch_trial 1
# TP2 additionally needs the exact candidate patch below.
ASCEND_RT_VISIBLE_DEVICES=0,1 KT_NPU_EXPERIMENTAL_TP2=1 launch_trial 2
# For the separate eager functional check, add --disable-cuda-graph.
```

The local snapshot was reused, not published as a pullable image. Its image ID was `sha256:f03917b88b3696659b70e2e11c3190978af4732d87be05fc624245b9bb6f6b3e`.

Candidate file: `python/sglang/srt/layers/moe/kt_ep_wrapper.py`.

- Base file SHA-256: `ac454c4aa5410181625a8c9aa520b7c18b6ab210e2706586e0fc7d64aad4dd8f`
- Candidate file SHA-256: `3368f3bd975512e9cac32d45220aebe020f14399988a53d44745071abfdb9739`

```diff
--- a/python/sglang/srt/layers/moe/kt_ep_wrapper.py
+++ b/python/sglang/srt/layers/moe/kt_ep_wrapper.py
@@ -177,7 +177,12 @@
                 "KT expert offload on NPU currently requires "
                 "--kt-num-gpu-experts >= 1"
             )
-        if server_args.tp_size != 1 or server_args.ep_size != 1:
+        experimental_tp2 = (
+            os.environ.get("KT_NPU_EXPERIMENTAL_TP2") == "1"
+            and server_args.tp_size == 2
+            and server_args.ep_size == 1
+        )
+        if (server_args.tp_size != 1 or server_args.ep_size != 1) and not experimental_tp2:
             raise ValueError(
                 "KT expert offload on NPU currently supports only "
                 "--tensor-parallel-size 1 and --expert-parallel-size 1"
@@ -404,6 +409,17 @@
                 self.gpu_experts_mask,
                 torch.arange(num_experts, dtype=torch.long),
                 torch.full((num_experts,), -1, dtype=torch.long),
+            )
+
+        if os.environ.get("KT_NPU_EXPERIMENTAL_TP2") == "1":
+            if layer.moe_tp_size != 2:
+                raise ValueError("TP2 trial requires MoE tensor parallel size 2")
+            logger.warning(
+                "[KT TP2 EXPERIMENT] layer=%s rank=%s moe_tp=%s local_intermediate=%s "
+                "full_intermediate=%s resident_experts=%s cpu_owner=%s",
+                self.kt_config.layer_idx, self.tp_rank, layer.moe_tp_size,
+                intermediate_size_per_partition, intermediate_size_full,
+                self.num_gpu_experts, self.tp_rank == 0,
             )
 
         # 3. Initialize KT wrapper for CPU experts
```

</details>

<details>
<summary>Frozen input construction and all 24 measured requests</summary>

Inputs used one user message, constructed as follows; actual token lengths were checked with the pinned model's chat template at every server start.

```python
paragraph = (
    "NUMA placement changes which memory controller serves CPU expert weights. "
    "Measure prefill and decode separately, and keep the model and request fixed. "
)
instruction = (
    "\nWrite a detailed engineering guide on benchmarking CPU and NPU inference. "
    "Cover measurements, interpretation, reproducibility, and examples. "
    "Write at least 1500 words."
)
messages = [{"role": "user", "content": paragraph * repetitions + instruction}]
# repetitions=4 -> 144 actual prompt tokens; repetitions=28 -> 792.
# Streaming chat-completions request:
# temperature=0, seed=20261111, max_tokens=512, ignore_eos=True,
# stream=True, stream_options={"include_usage": True}.
```

Times below are seconds from the same client clock; loading, capture, task checks, and warmups are excluded. Individual rows are rounded to six decimals; headline medians use full precision.

| Start | Input tokens | Repeat | TTFT (s) | Decode (tok/s) | Total (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| tp1a | 144 | 1 | 2.390547 | 15.422993 | 35.523359 |
| tp1a | 144 | 2 | 2.470154 | 15.398905 | 35.654900 |
| tp1a | 144 | 3 | 2.440593 | 15.374660 | 35.677891 |
| tp1a | 792 | 1 | 11.282067 | 15.344001 | 44.585576 |
| tp1a | 792 | 2 | 11.241273 | 15.379776 | 44.467161 |
| tp1a | 792 | 3 | 11.259935 | 15.397976 | 44.446588 |
| tp2a | 144 | 1 | 2.478882 | 15.990207 | 34.436439 |
| tp2a | 144 | 2 | 2.471236 | 16.000449 | 34.408347 |
| tp2a | 144 | 3 | 2.464054 | 15.987517 | 34.426954 |
| tp2a | 792 | 1 | 11.293895 | 15.932864 | 43.366596 |
| tp2a | 792 | 2 | 11.278204 | 15.881174 | 43.455179 |
| tp2a | 792 | 3 | 11.298144 | 15.944008 | 43.348250 |
| tp2b | 144 | 1 | 2.492518 | 15.680486 | 35.081387 |
| tp2b | 144 | 2 | 2.474088 | 15.703138 | 35.015986 |
| tp2b | 144 | 3 | 2.474152 | 15.616020 | 35.197491 |
| tp2b | 792 | 1 | 11.313335 | 15.567131 | 44.139342 |
| tp2b | 792 | 2 | 11.301734 | 15.561160 | 44.140396 |
| tp2b | 792 | 3 | 11.277996 | 15.524140 | 44.194903 |
| tp1b | 144 | 1 | 2.388657 | 15.282585 | 35.825993 |
| tp1b | 144 | 2 | 2.481053 | 15.257884 | 35.972445 |
| tp1b | 144 | 3 | 2.405936 | 15.269639 | 35.871572 |
| tp1b | 792 | 1 | 11.268948 | 15.369538 | 44.517092 |
| tp1b | 792 | 2 | 11.296522 | 15.401171 | 44.476361 |
| tp1b | 792 | 3 | 11.281498 | 15.382812 | 44.500828 |

Each of the following output hashes occurred in all six measured requests for its TP/input pair. Cross-TP hashes differ.

| Input tokens | TP | Output SHA-256 |
| ---: | ---: | --- |
| 144 | 1 | `e44e73567cc4cc1a8e7b204550e79cbe05e32fac344e95c528c96c06855365ac` |
| 144 | 2 | `70fa9b10a9991a444513f243407709993843dacd5fa87549b6f676176eb4e56b` |
| 792 | 1 | `63a946663802ed1c9d674e96432edbbf199c561335aee8e9032a85c7db1f92f8` |
| 792 | 2 | `bd40384847a49c5547b9530f41894b9bcac3fa0cb96a4a3962ce064ac6d28f2b` |

</details>
