## Motivation

Enable experimental KT CPU-expert offload on Ascend, with TP=1/EP=1 by default and an explicit TP=2/EP=1 opt-in. Upstream's KT wrapper currently uses CUDA streams directly; adding a TP switch alone would not supply the missing NPU execution path.

This is a focused adaptation of the Apache-2.0 Ascend implementation by **Pan-Boyi and contributors**, from [Pan-Boyi/sglang at ff46d660](https://github.com/Pan-Boyi/sglang/tree/ff46d660d16cbf11ef393c80cc6b3ab7f6259780), plus our experimental TP2 enablement and hardware observations. Original source attribution is retained in the adapted files. The CPU executor, expert TP partitioning, and reductions are existing authors' work, not a newly invented TP algorithm.

Related context: #34673 (closed, unmerged Ascend KT integration) and [KT hardware follow-up with the original candidate patch](https://github.com/kvcache-ai/ktransformers/pull/2157#issuecomment-6810709188).

## Modifications

- Adapt KT submission/synchronization to NPU streams, including the pinned CPU host callback required for decode graph capture and early checking of the companion kt-kernel ABI.
- Replace the CUDA `-1` expert sentinel with zero-weight valid resident routes on NPU. Submit the unmodified routing data to CPU first; CPU output is added on rank 0 only.
- Match the Ascend kt-kernel's resident-mask and SwiGLU-limit interface; resolve per-layer GGUF path templates.
- Retain official main's **standard dispatcher and Ascend runner**. The fork's older dispatcher hooks, optional streaming-prefill implementation, dynamic expert placement, packaging changes, and unrelated model changes are not imported.
- Require no A2A, EP=1, at least one resident expert, and matching model/MoE TP sizes. TP2 requires `KT_NPU_EXPERIMENTAL_TP2=1`; other TP sizes remain rejected on NPU.
- Add a short usage/limitations note and CPU-only tests. CUDA keeps its existing constructor arguments, routing, and execution calls.

Requires the matching Ascend-enabled kt-kernel described in [KT's Ascend tutorial](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/DeepSeek-V4-Flash_tutorial_for_Ascend_NPU.md); this does not make older CUDA-only kt-kernel builds support NPU.

## Accuracy Tests

**This port to official main has not been run on NPU hardware.** The hardware evidence below is from the pinned fork plus the original TP2 candidate, not end-to-end validation of this upstream adaptation.

Local validation on the submitted code:

- `python test/manual/ascend/test_kt_npu_offload.py`: 7 tests pass, including 108 configuration combinations; resident/CPU routing, unmodified CPU inputs, all-CPU and empty routes, per-layer paths, callback ordering, and selective ACL error handling.
- The real wrapper `apply()` is exercised with mocked native/serving dependencies for both ranks, in eager and graph modes: CPU submit/output addition occurs on rank 0 only.
- Python syntax, Black, isort, and Git whitespace checks pass. These are CPU/mock checks, not NPU kernel or CUDA regression validation.

Existing fork hardware evidence (2026-11-11 through 2026-11-16): dual-socket Kunpeng 920, 2x Ascend 910B4-1 (64 GiB each), CANN 9.0.0, torch_npu 2.10.0; DeepSeek-V4-Flash W8A8 with per-layer MXFP4 CPU experts; KT `c9873312643b615a717832946fba0346e69ab035`, SGLang `ff46d660d16cbf11ef393c80cc6b3ab7f6259780` plus the original opt-in patch.

- Two-rank HCCL all-reduce passed for 1, 1,024, and 262,144 FP32 elements. Eager and batch=1 decode graph smoke/SSE requests passed. All 43 layers reported intermediate dimensions 1,024 per rank / 2,048 full and rank-0 CPU ownership.
- A fixed-history probe covered 8 inputs x 9 positions, repeated twice, in TP1/TP2 x eager/graph. Repeats were bitwise identical, and eager/graph logits matched within each TP setting.
- **Cross-TP numerical correctness remains unresolved:** 5 of 72 distinct positions had different top-1 tokens; maximum absolute logit difference was 3.75, maximum total variation distance 0.33250. These differences have not been established as expected rounding error. Independent task-quality evaluation has not been completed.
- A narrow layer-0 trace found differences already at the MoE input. Its sampled all-CPU routes showed CPU output added on rank 0 only and the expected actual BF16 all-reduce. This does not establish correctness for mixed routes or other layers.

## Speed Tests and Profiling

No performance measurements are available for this upstream port. On the pinned fork, ABBA TP1/TP2 starts completed 24 measured requests (144/792 input tokens, 512 output tokens) with 32 resident experts and 96 CPU threads. TP2 decode medians were 15.845/15.724 tok/s versus TP1 15.329/15.381 tok/s. Text differed across TP settings; this is a limited equal-input/equal-length observation, not a general speedup claim.

Automatic KV/indexer sizing OOMed in that environment; `--max-total-tokens 16384` allowed startup. Long context, concurrent requests, other models/TP/EP sizes, graceful shutdown, and the integrated upstream execution path remain unvalidated.
