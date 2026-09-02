# Next Steps

## Status

Batch 1 target met: ~31.3 -> ~116 tok/s (~3.8x) on Tesla T4, variant F
(StaticCache + `torch.compile(reduce-overhead)` + full INT8), all
correctness/quality gates passing. See RESULTS.md "Batch 1".

## Immediate action: validate batch 2 (kernel rework)

Profiling of the batch-1 result showed `int8_gemv_kernel` at ~38.5% of
CUDA time and `fused_int8_gate_up_kernel` at ~33%. Both kernels were
reworked in `kernels.py` (tiled K-loop instead of a `next_pow2(K)`-wide
masked load, fp16 multiply / fp32 accumulate) -- see RESULTS.md
"Batch 2" for the full reasoning. Run, in order:

```bash
python test_correctness.py       # CPU-safe quant-math check, should still pass
python microbench_kernels.py     # kernel-level: old vs new + BLOCK_K/num_warps sweep, per real shape
python run_t4_experiments.py     # end-to-end same-session A/B/A, decides keep/reject
```

Report back:
1. `microbench_kernels.py`'s per-shape "BEST" line (confirms whether
   `DEFAULT_BLOCK_K=256`/`DEFAULT_NUM_WARPS=4` in `kernels.py` are
   actually fastest, or should change).
2. Whether `test_correctness.py`'s CUDA-dependent tests
   (`int8_gemv_kernel[*]`, `fused_int8_gate_up_kernel`, `int8_quality`)
   still pass -- the math changed (tiling, fp16 multiply), so this is
   the real correctness gate, not the CPU-safe subset.
3. `run_t4_experiments.py`'s end-to-end tok/s vs the ~116 tok/s
   baseline. Keep the kernel change only if this improves; otherwise
   revert `kernels.py` (the pre-change kernel is preserved verbatim in
   `microbench_kernels.py` for comparison and as a rollback reference).

## If the kernel rework doesn't move end-to-end tok/s

That would mean `int8_gemv_kernel`'s 38.5% was dominated by something
other than masked-lane waste (e.g. launch count/occupancy rather than
per-program wasted work, or the CUDA-graph-captured path already hides
this cost). Re-profile variant F specifically after the change and read
the table rather than re-guessing -- if `int8_gemv_kernel`'s share of
CUDA time didn't drop, the tiling hypothesis was wrong and should be
reverted, not iterated on blindly.

## Deferred (do not build until measurement justifies it)

- **`triton.autotune`** for BLOCK_K/num_warps: correct tool in general,
  rejected specifically because these kernels run inside a
  `torch.compile(reduce-overhead)` CUDA-graph-captured region and
  autotune's dynamic first-call benchmarking isn't safe there. Static
  config + offline sweep (`microbench_kernels.py`) used instead.
- **INT4 weight-only quantization**: only worth it if, after the
  kernel rework above, re-profiling shows memory bandwidth (not
  per-program waste or launch/graph overhead) is still the binding
  constraint.
- **DP4A / INT8 tensor cores**: not applicable without also quantizing
  activations (this scheme is weight-only, activations stay fp16),
  which would change quantization semantics; also a compute-side lever
  on a memory-bandwidth-bound (M=1) kernel, so unlikely to matter much
  even if built.
- **Finer-grained fusion** (RMSNorm+QKV, O-proj+residual): launch count
  is no longer the bottleneck once CUDA graphs capture the whole decode
  step; the remaining case for this would be memory traffic, a smaller
  effect. Revisit only if a fresh profile shows these ops still matter.
- **A custom attention kernel**: never flagged as a major cost in any
  profiling run so far; SDPA is left untouched.
