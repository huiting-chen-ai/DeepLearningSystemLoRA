# LoRA Fine-Tuning and Multi-Adapter Serving in Needle

**Team:** Huiting Chen, Joshua Moorehead
**Draft:** 2026-11-10 (for TA mentor meeting 2026-11-11)

## 1. Summary

We extend Needle with LoRA fine-tuning for a small decoder-only transformer, then
study it from both ends of the model lifecycle:

- **Training side (Huiting):** LoRA vs. full fine-tuning — memory, throughput,
  and a forward/backward profile across LoRA ranks and batch sizes; implement
  one targeted optimization found by profiling and evaluate it.
- **Inference side (Josh):** a generate loop with a KV cache, merged vs.
  unmerged LoRA inference, and **multi-adapter batched serving** (S-LoRA /
  Punica style: one batch serves N different adapters) backed by a custom
  CUDA kernel for the low-rank path.

Both halves share one transformer (hw4_extra `Transformer`), one dataset (PTB),
one profiler, and one benchmark harness. Everything is implemented in Needle;
PyTorch + `peft` is used only as a correctness reference.

## 2. Headline results we want to show

| # | Result | Owner |
|---|--------|-------|
| R1 | Peak memory & tokens/sec: full FT vs. LoRA at rank r ∈ {1, 4, 16, 64}, batch ∈ {8, 32, 128} | Huiting |
| R2 | Per-op profile of fwd/bwd; speedup from the targeted optimization | Huiting |
| R3 | Per-token decode latency vs. sequence length, with and without KV cache | Josh |
| R4 | Merged vs. unmerged LoRA inference latency | Josh |
| R5 | Tokens/sec vs. number of concurrent adapters: naive loop vs. batched Python vs. CUDA BGMV kernel | Josh |
| R6 | Correctness: LoRA outputs match PyTorch + peft within tolerance | Huiting |

## 3. Repository layout (proposed)

Rebuild the repo on the **hw4_extra** tree (it is the fa25-updated superset of
hw3 and hw4, and it contains the transformer skeleton + tests). Carry over
existing hw3 solutions (`ndarray.py`, CPU/CUDA backends) and hw2/hw3 ops.

```
DeepLearningSystemLoRA/
├── python/needle/                # the framework (hw4_extra tree + solutions)
│   ├── nn/nn_transformer.py      # from hw4_extra; KV-cache hooks added here
│   ├── nn/nn_lora.py             # LoRALinear, apply_lora(), merge/unmerge     [Huiting]
│   ├── profiler.py               # per-op timing + NDArray allocation tracking [Huiting]
│   └── backend_ndarray/          # batched matmul, bgmv exposed to Python
├── src/
│   ├── ndarray_backend_cpu.cc
│   └── ndarray_backend_cuda.cu   # + batched matmul kernel [Huiting], BGMV kernel [Josh]
├── apps/
│   ├── models.py                 # LanguageModel using Transformer
│   ├── finetune.py               # full FT / LoRA training entrypoint        [Huiting]
│   ├── generate.py               # sampling loop, KV cache on/off             [Josh]
│   ├── serve.py                  # multi-adapter batched decode              [Josh]
│   └── bench/                    # one script per result R1–R6, writes CSV
├── tests/
│   ├── hw4/, hw4_extra/          # course tests kept as regression suite
│   └── project/                  # test_lora.py, test_kv_cache.py, test_serve.py, test_kernels.py
├── notebooks/colab.ipynb         # build + run on T4
├── docs/                         # this proposal, results/, final report
├── data/, checkpoints/           # gitignored
├── Makefile, CMakeLists.txt, pytest.ini, .gitignore
└── README.md
```

Removed from the current repo: `hw3.ipynb`, `make_archive.py`, mugrade
references.

Process: `main` is protected; work on `feature/*` branches; PRs reviewed by the
other person; CPU test suite runs in GitHub Actions on every PR. CUDA tests run
on Colab before merging kernel work.

## 4. Timeline (4 weeks)

Hard dates to confirm with TA: final report / video due date.

### Week 1 — Nov 10–16: foundation
Shared milestone: **a transformer LM trains on PTB on CPU and T4 from the project repo.**

| Josh | Huiting |
|------|---------|
| Rebuild repo on hw4_extra tree, merge hw3 solutions, CI, Colab notebook (PR #1) | `LoRALinear` + `apply_lora(model, targets, rank)` + freeze logic, unit tests |
| hw4 / hw4_extra solutions (own copy, for the course) | hw4 / hw4_extra solutions (own copy, for the course) |
| `generate.py`: greedy / temperature / top-k sampling, no cache | `profiler.py`: op timing + peak-memory tracking |
| | PyTorch + peft reference script for R6 |

### Week 2 — Nov 17–23: core features
| Josh | Huiting |
|------|---------|
| KV cache in `MultiHeadAttention` / `AttentionLayer`; R3 | `finetune.py`; R1 sweep (rank × batch) |
| Merged vs. unmerged inference; R4 | Profile fwd/bwd; R2 part 1 (identify bottleneck — expected: the broadcast-and-sum "batched matmul") |
| Multi-adapter batching in Python (adapter-id gather, shared base matmul) | |

### Week 3 — Nov 24–30 (Thanksgiving week, plan for ~60%)
| Josh | Huiting |
|------|---------|
| BGMV CUDA kernel (gathered low-rank matmul), tests, R5 | Batched-matmul CUDA kernel (fwd + bwd), tests; R2 part 2 |
| Stretch: continuous batching across adapters (requests join/leave between decode steps) | Re-run R1 with the optimization |

### Week 4 — Dec 1–7: results, report, video
Both: freeze features by Dec 3, generate all plots from `apps/bench/`, write
report, record video. Buffer for anything that slipped.

Stretch (only if ahead by Dec 1): speculative decoding with a 1-layer draft model.
Future-work slide: paged KV cache, quantized inference.

## 5. Division of work

| Area | Josh | Huiting | Shared |
|------|------|---------|--------|
| Repo scaffold, CI, Colab | ✓ | | |
| hw4 / hw4_extra transformer | | | each, merged once |
| LoRA layer + merge/unmerge | | ✓ | |
| Profiler + memory tracking | | ✓ | |
| Fine-tune harness, R1, R2, R6 | | ✓ | |
| Batched-matmul CUDA kernel | | ✓ | |
| Generate loop, KV cache, R3, R4 | ✓ | | |
| Multi-adapter serving, BGMV kernel, R5 | ✓ | | |
| Continuous batching (stretch) | ✓ | | |
| Benchmark scripts + plots | own results | own results | shared format |
| Report + video | | | ✓ |

## 6. Risks

| Risk | Mitigation |
|------|------------|
| Transformer in Needle is slow on CPU | Tiny config (2–4 layers, d=128, T=64–128); CPU for correctness, T4 for numbers |
| CUDA iteration only on Colab | Write kernels against CPU reference first; small parametrized tests; one Colab session per day for kernel work |
| hw4 due in the same window | Week-1 project tasks are small and independent of hw4; hw4_extra *is* the shared transformer |
| Scope creep | Ladder structure: every rung is a shippable result on its own |

## 7. Questions for the TA

2. Reusing hw4_extra's transformer and the course PTB loader as the base — OK?
4. Is the split (training-side vs. serving-side) a fair two-person scope, or should we cut / add?
5. Any preference between "one targeted optimization, evaluated deeply" vs. "several smaller ones"?
