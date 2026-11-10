# TA Check-in Answers (form due Nov 14)

**Project title:** LoRA Fine-Tuning and Multi-Adapter Serving in Needle
**Members:** Huiting Chen (andrew: ___), Joshua Moorehead (andrew: ___)

## Project clarity

**Idea in 1–2 sentences.** We add LoRA fine-tuning to Needle and study it end to
end: training-side memory/throughput/profiling vs. full fine-tuning, and
serving-side inference with a KV cache and S-LoRA-style batching of many
adapters in one batch, backed by custom CUDA kernels.

**Beyond the homeworks.** hw4_extra gives a transformer; it has no LoRA, no
inference path (no generate loop, no KV cache), no profiler, no batched matmul
(the skeleton materializes a B×H×T×T×D tensor), and no multi-adapter serving.
Everything above is new.

**Background.** LoRA (Hu et al. 2021); S-LoRA (Sheng et al. 2023) and Punica
(Chen et al. 2023) for multi-adapter serving / BGMV kernel; HF `peft` only as a
correctness reference.

## Implementation in Needle

**Components added/changed:**
- `nn/nn_lora.py`: `LoRALinear` module, `apply_lora`, merge/unmerge
- `backend_ndarray` + `src/*.cu`: batched matmul kernel (fwd + bwd); BGMV
  (gathered low-rank matmul) kernel; Python bindings + ops
- `nn/nn_transformer.py`: KV-cache path in attention
- `profiler.py`: per-op timing + NDArray allocation tracking
- `apps/`: `finetune.py`, `generate.py`, `serve.py`, benchmark scripts

**Dependencies beyond Needle:** numpy, pytest; torch + peft only for reference
checks. Colab T4 for GPU numbers.

## Progress so far

- Repo rebuilt on the hw4_extra tree with hw3 backends + hw2 ops merged;
  tests/hw3 72/72 passing on CPU; hw4 / hw4_extra solutions in progress.
- Correctness plan: course test suites as regression; LoRA layer vs. PyTorch +
  peft outputs; KV-cache decode vs. full recompute (exact match); kernels vs.
  CPU reference on random inputs.
- Demo: transformer LM on Penn Treebank (course dataset); fine-tune with LoRA,
  then generate with N adapters in one batch.

## Milestones before video (Nov 28) and report

1. **Nov 16:** transformer trains on PTB from the project repo; `LoRALinear`
   passes peft comparison; generate loop works.
2. **Nov 23:** LoRA vs. full-FT memory/throughput sweep done; KV cache done
   with latency-vs-length plot; multi-adapter batching in Python.
3. **Nov 28 (video):** batched-matmul + BGMV CUDA kernels with before/after
   numbers; continuous batching only if time remains.

## Team

- Huiting: LoRA module, profiler, fine-tune harness, training benchmarks,
  batched-matmul kernel, peft reference.
- Josh: repo/CI/Colab, generate loop, KV cache, merged-vs-unmerged, multi-adapter
  serving, BGMV kernel.
- Shared: hw4/hw4_extra transformer, report notebook, video.

## Attribution

All code, prose, and figures our own; papers cited; plots generated from our
benchmark CSVs.

## Risks

- **Biggest risk:** CUDA kernel iteration only on Colab; Needle transformer slow
  on CPU. Mitigation: tiny model config; kernels tested against CPU reference
  first; one Colab session/day for kernel work.
- **Fallback:** multi-adapter serving in Python (no kernel) + KV cache + the
  training-side comparison already meets the requirements.

## Questions for the TA

1. Final report deadline?
2. Reusing hw4_extra's transformer and PTB loader as the base is OK?
3. Colab T4 acceptable for GPU numbers?
4. Is the two-sided scope (training + serving) about right?
