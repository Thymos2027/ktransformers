# TP2 PR submission

## Official upstream PR

2026-11-16: opened regular (non-draft) official PR:
https://github.com/sgl-project/sglang/pull/50675

- Target: `sgl-project/sglang:main`, base `2f7dd856a0346fdb9a474e4362fcc0f3cdb67e1a`.
- Head: `Thymos2027/sglang:feat/ascend-kt-cpu-offload-tp2`.
- Commit: `2423f262728e59dcfef1a9792cf5d68db6062ef9`.
- Adapts necessary Apache-2.0 Ascend KT execution from Pan-Boyi and contributors, with source attribution and Pan-Boyi co-author trailer. Adds explicit TP2/EP1 opt-in.
- Keeps official main's standard dispatcher/NPU runner. Does not transplant fork-only dispatcher hooks or streaming prefill.
- Four files: KT wrapper, NPU helper, short usage note, CPU/mock tests.
- Validation: 7 tests (including 108 configuration cases and real wrapper apply with mocked dependencies across both ranks/eager/graph), syntax, Black, isort, staged whitespace checks.
- **No NPU hardware run of this official-main port.** Existing fork results and unresolved TP numerical differences are disclosed separately in the PR. No remote machine changes were made.
- Body: [official-pr-body.md](official-pr-body.md).
- Local checkout: `E:/Coder/LLM/sglang-taishan-tp2-contrib`, clean after commit.
- Earlier fork PR below remains open; it is distinct from the official-main adaptation.

## Earlier fork-only PR

2026-11-16: opened a regular (non-draft) PR:
https://github.com/Pan-Boyi/sglang/pull/4

- Target: `Pan-Boyi/sglang:dsv4-cann9-no-patch`.
- Head: `Thymos2027/sglang:feat/ascend-kt-experimental-tp2`.
- Commit: `f920aed` — `feat: add experimental ascend tp2 opt-in for kt cpu offload`.
- Local checkout: `E:/Coder/LLM/sglang-taishan-tp2-contrib`.
- Scope: opt-in NPU TP2/EP1 guard, NPU-only MoE TP check and diagnostics, and one short launch/validation note. Existing CPU execution, partitioning and reductions are reused.
- Local validation: syntax, Black, staged whitespace check, 168 AST-isolated guard cases. No new remote experiments or service modifications.
- PR body: [pr-body.md](pr-body.md); local guard check: [check_pr_guards.py](check_pr_guards.py).
- Existing hardware results and unresolved cross-TP numerical differences are explicitly disclosed; the final guard cleanup has not had a fresh hardware run.

The GitHub connector returned `403 Resource not accessible by integration` on PR creation. The authenticated local GitHub CLI created the PR successfully.
