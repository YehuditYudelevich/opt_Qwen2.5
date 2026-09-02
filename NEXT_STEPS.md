# Next Steps

## Status

Batch 1 target met: ~31.3 -> ~116 tok/s (~3.8x) on Tesla T4, variant F
(StaticCache + `torch.compile(reduce-overhead)` + full INT8), all
correctness/quality gates passing. See RESULTS.md "Batch 1".

## Immediate action: verify the batch-2 NaN fix, correctness FIRST

The first batch-2 attempt (tiling + fp16 multiply) produced a real
`perplexity_int8 = NaN` on the T4, root-caused to overflow: the fp16
multiply used the RAW unscaled int8 weight byte (range +-127) against
the real activation, before the dequant scale was applied -- any
activation channel with `|x| > ~515` overflows fp16 (confirmed by
direct CPU reproduction: `800 * int8(127)` -> `inf` in fp16, `101600`
in fp32). Isolated kernel tests never caught it because they use
unit-variance synthetic activations, not real (unnormalized, sometimes
outlier) ones. See RESULTS.md "Batch 2, 2.1" for the full writeup.

Fix applied (RESULTS.md "2.2"): kept the tiled/no-masked-waste loop,
reverted the multiply to fp32 (matching the batch-1 kernel's numerics
exactly). Verified so far only via CPU emulation (matches reference to
~1e-4 to 1e-3 error; the exact forced-overflow case that broke 2.1 now
produces a finite result). **Not yet run on the T4.** Run, strictly in
this order -- do not skip ahead to benchmarking:

```bash
python test_correctness.py       # must pass, especially int8_quality (no NaN), before anything else below
python debug_int8_nan.py         # optional: if anything still looks off, pinpoints which layer/module
python microbench_kernels.py     # only after int8_quality passes: kernel-level old vs new + BLOCK_K/num_warps sweep
python run_t4_experiments.py     # only after int8_quality passes: end-to-end same-session A/B/A
```

Report back:
1. Whether `test_correctness.py`'s CUDA-dependent tests -- especially
   `int8_quality` -- pass (no NaN/inf, top-1 agreement/perplexity ratio
   within the existing thresholds). **If not, stop here and report
   back; do not run the benchmarks.**
2. If it passes: `microbench_kernels.py`'s per-shape "BEST" line
   (confirms whether `DEFAULT_BLOCK_K=256`/`DEFAULT_NUM_WARPS=4` are
   actually fastest) and `run_t4_experiments.py`'s end-to-end tok/s vs
   the ~116 tok/s baseline. Keep the kernel change only if this
   improves; otherwise revert `kernels.py` to git commit `ff64322` (the
   exact kernel that produced the confirmed ~116 tok/s result) -- also
   preserved verbatim in `microbench_kernels.py` for comparison.

## If int8_quality still fails after the fp32 revert

Then the fp16-overflow theory was incomplete or something else is also
wrong. Run `debug_int8_nan.py` (hooks every layer's
self_attn/mlp/lm_head for inf/nan, plus reports max-abs activation into
o_proj/down_proj per layer via `optimized_model.DEBUG_ACTIVATION_STATS`)
and report exactly which layer/module it flags -- don't re-guess at the
arithmetic again without that data.

## If the kernel rework (once passing) doesn't move end-to-end tok/s

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
