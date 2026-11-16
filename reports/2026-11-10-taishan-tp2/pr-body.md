## Motivation

The Ascend KT wrapper currently rejects TP > 1. This adds an explicit experimental opt-in for CPU-expert offload with two NPUs (TP=2, EP=1), based on hardware runs of this branch. Default behavior remains TP=1.

## Modifications

- Accept TP=2/EP=1 only with `KT_NPU_EXPERIMENTAL_TP2=1`; keep the existing resident-expert and no-A2A requirements.
- Check MoE TP size before creating the CPU wrapper and log the rank, expert partition dimensions, and CPU owner.
- Document launch parameters, the explicit KV capacity workaround, and validation limits.

This reuses the existing authors' rank-0 CPU executor, NPU expert partitioning, and TP reduction paths; it does not introduce a new TP algorithm.

## Accuracy Tests

Hardware: dual-socket Kunpeng 920, 2x Ascend 910B4-1 (64 GiB each), CANN 9.0.0, torch_npu 2.10.0. Model: DeepSeek-V4-Flash W8A8 with per-layer MXFP4 CPU experts. KT: `c9873312643b615a717832946fba0346e69ab035`; SGLang base: `ff46d660d16cbf11ef393c80cc6b3ab7f6259780`.

Existing candidate hardware runs (2026-11-11 through 2026-11-16):

- Two-rank HCCL all-reduce passed for 1, 1,024, and 262,144 FP32 elements.
- Eager and batch=1 decode graph runs passed capital, arithmetic, and SSE smoke requests. All 43 layers logged intermediate dimensions 1,024 per rank / 2,048 full, with CPU ownership on rank 0 only.
- Fixed-history probe: 8 inputs x 9 positions, repeated twice, in TP1/TP2 x eager/graph. Repeats were bitwise identical within each configuration; eager and graph logits matched within each TP setting.
- **TP1 vs TP2 is not numerically accepted:** 5 of 72 distinct positions had different top-1 tokens; maximum absolute logit difference was 3.75 and maximum total variation distance was 0.33250. Whether this is expected numerical error or an implementation defect remains unresolved. Independent task-quality evaluation has not been completed.
- A narrow layer-0 trace found differences already at the MoE input. For the sampled all-CPU routes, CPU output was added on rank 0 only, and TP2's actual BF16 all-reduce matched the sum of the two local contributions. This does not establish correctness for other layers or mixed CPU/NPU routes.

Local checks: Python syntax compilation, Black, whitespace checks, and 168 AST-isolated configuration/MoE guard cases (including disabled opt-in, TP1 fallback, unsupported TP/EP, A2A and resident-expert requirements). These do not execute accelerator code. The hardware results above belong to the original opt-in candidate, not a fresh hardware run of this commit.

## Speed Tests and Profiling

Fixed 32 resident experts, 96 CPU threads, four threadpools, single request, total-token cap 16,384. ABBA TP1/TP2 independent starts completed 24 measured requests with 144/792 input tokens and 512 output tokens. Pooled median TP2 decode throughput was 15.845/15.724 tok/s versus TP1 15.329/15.381 tok/s (+3.37%/+2.23%). Generated text differed across TP settings; this is a limited equal-input/equal-length observation, not a general speedup or accuracy claim.

Earlier hardware report, exact original patch, and reproduction parameters: https://github.com/kvcache-ai/ktransformers/pull/2157#issuecomment-6810709188

Automatic KV sizing OOMed on the pinned environment; `--max-total-tokens 16384` allowed startup. Long context, concurrency, other TP/EP sizes, and graceful shutdown have not been validated.
