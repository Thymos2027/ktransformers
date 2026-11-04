The Ascend recipe currently requires downloading the native checkpoint and converting its routed experts locally. This adds an optional community download for the 43 per-layer CPU expert GGUF files, with reproducible provenance and validation records.

ModelScope: https://modelscope.cn/models/kmon03f/DeepSeek-V4-Flash-KT-MXFP4-CPU-Experts

- Expert-only native MXFP4 repack: 43 files, 147,169,763,520 bytes. A separate compatible W8A8 checkpoint is still required.
- All 43 layers × 256 experts × 3 projections checked against the native checkpoint, including packed-code/scale-byte comparison and the upstream element-wise bit-exact verifier. SHA-256 was independently re-read, and uploaded file sizes and hashes were verified against ModelScope metadata.
- Tested with dual Kunpeng 920 / 512 GB RAM and one Ascend 910B4-1 64 GiB, CANN 9.0.0, torch/torch_npu 2.10.0, 32 resident experts, 8192 context, and single-request concurrency. NPU graph capture, health checks, chat and SSE streaming passed.

The model card includes source file revisions/hashes, KT `c9873312643b615a717832946fba0346e69ab035`, SGLang `ff46d660d16cbf11ef393c80cc6b3ab7f6259780`, conversion instructions, SHA256SUMS, and full validation results. This uses the existing KT conversion method and does not claim a new quantization algorithm, official endorsement, or general accuracy/performance equivalence. Model weights preserve the DeepSeek MIT license; bundled KT tools retain Apache-2.0 terms.

Validation report: https://modelscope.cn/models/kmon03f/DeepSeek-V4-Flash-KT-MXFP4-CPU-Experts/files

This is a third-party community artifact; maintainers can decide whether it is appropriate to list it in the recipe. No long-context, multi-request throughput or broad model-quality benchmark is claimed.
