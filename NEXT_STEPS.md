# Next Steps

## Immediate action

Run the batch-1 experiment script on the Tesla T4 and report back the
output (or paste `results/run_<timestamp>.json` + the printed SUMMARY
block):

```bash
pip install -r requirements.txt
python run_t4_experiments.py --quick   # sanity check the harness runs end-to-end, ~2 min
python run_t4_experiments.py           # full run, ~15-20 min depending on Colab GPU state
```

The script gates speed measurements on correctness: if a variant's
prerequisite test fails, it is skipped and logged as skipped, never
silently benchmarked anyway. Read the console "Gate summary" and
"SUMMARY" sections first -- they tell you which of the six variants
(A baseline, B manual+dynamic, C manual+static, D +compile/CUDA-graphs,
E +INT8, F compile+INT8) actually produced trustworthy numbers.

## Current understanding of the bottleneck (from prior profiling, batch 0)

With the fused INT8 MLP alone enabled, CUDA time was still:

- fused INT8 gate/up kernel: ~20%
- remaining `aten::mm` (everything else, still FP16 PyTorch): ~39%
- lm_head alone: ~16%
- 1536->1536 projections (Q/O): ~8-9%
- K/V projections, attention, copies/casts, elementwise: the remainder,
  spread across "many thousands of small operations and launches", plus
  "substantial CPU overhead"

Combining a fast INT8 MLP with a fast INT8 lm_head *still regressed*
end-to-end by 6.2%. That is strong evidence that the binding constraint
is not any single operator's speed -- it's the volume of Python
dispatch and non-fused work per decode step. Amdahl's law: if launch/
dispatch overhead is a large, roughly-fixed cost per op regardless of
that op's compute time, then making one or two ops individually faster
barely moves the total, and can even move it the wrong way if the
replacement adds any extra Python-side branching.

This is exactly why batch 1 (see RESULTS.md) treats "reduce overhead"
(manual loop, StaticCache, `torch.compile(reduce-overhead)` /
CUDA graphs) and "reduce memory traffic" (INT8 across every matrix,
fused QKV, fused gate/up) as two separate, both-necessary levers to
measure, rather than continuing to swap individual `nn.Linear` calls.

## Highest-value next experiment, conditional on batch-1 results

This section should be rewritten once real numbers exist. As placeholders,
the decision tree is:

- **If variant D (compile/CUDA-graphs, still FP16) alone gets close to
  or past 2x:** overhead was in fact the dominant term. Next step is
  making sure `fullgraph=True` actually held (check `compile_modes` in
  the saved JSON) and profiling variant D to see what's left -- likely
  memory bandwidth, at which point INT8/INT4 becomes the refinement,
  not the headline.

- **If variant E (INT8 everywhere, no compile) alone gets a real but
  insufficient gain (e.g. 1.2-1.4x) and variant B/C show `generate()`/
  cache overhead was small:** bandwidth matters but isn't sufficient
  alone either. Look at INT4 (halves INT8's bytes again) and finer
  fusion (e.g. RMSNorm+QKV, O-proj+residual) as the next lever, but only
  after re-profiling variant E specifically to confirm bandwidth (not
  launch count) is now the binding constraint.

- **If variant F (combined) reaches >=2x:** stop here for this metric,
  write up the result, and separately verify quality more thoroughly
  (larger perplexity sample, more generation prompts) before calling it
  done, per the "performance does not count if the model becomes
  materially worse" rule.

- **If variant F reaches most of the way but not 2x (e.g. 1.5-1.8x):**
  re-profile variant F specifically (the script already does this for
  whichever variant wins). The profiler table will say directly whether
  remaining time is (a) still uncaptured/graph-broken Python regions
  (look for `cudaLaunchKernel` counts or explicit graph-break warnings
  from `torch._dynamo`), (b) attention/SDPA (not yet touched by this
  batch), (c) INT8 GEMV kernel time itself (candidate for INT4), or
  (d) something profiling didn't anticipate. Do not guess; read the
  table.

- **If `torch.compile(fullgraph=True)` failed for variant F specifically
  (the INT8 path) but succeeded for D (FP16 path):** that confirms the
  raw-Triton-kernel-call boundary is the graph break. The fix is
  registering the INT8 GEMV/fused-gate-up wrappers as proper
  `torch.library` custom ops (with an FX-traceable meta/fake
  implementation) instead of calling `kernel[grid](...)` directly from
  plain Python, so dynamo can capture the call into the graph instead of
  breaking around it.

## Deferred from this batch (do not build until batch-1 results justify it)

- **INT4 weight-only quantization.** instructions.md explicitly warns
  not to assume it wins; it only becomes worth building if batch-1
  profiling shows memory bandwidth (not overhead) is still the binding
  constraint after CUDA graphs are in place.
- **Finer-grained fusion** (RMSNorm+QKV, O-proj+residual, second
  RMSNorm+MLP-entry). Once CUDA graphs are capturing the whole decode
  step, launch count stops being the reason to fuse further -- the
  remaining reason would be memory traffic (fewer intermediate FP16
  writes/reads), which is a much smaller win than eliminating launches
  entirely. Worth measuring only if variant F's re-profile shows these
  ops still matter.
- **A custom attention kernel.** Prior profiling never flagged attention
  as a major cost at this sequence-length regime; SDPA is left
  untouched. Revisit only if a future profile says otherwise.
