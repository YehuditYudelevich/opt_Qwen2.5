# Results Log

Every experiment that changed the optimized decode path, in chronological
order. Per CLAUDE.md / instructions.md methodology: a microbenchmark
number is never treated as an end-to-end claim on its own, and every
change is only kept if it survives a same-session A/B/A on the real
Tesla T4.

Legend: **kept** = enabled in the optimized path going forward,
**rejected** = implemented, measured, and turned back off because it did
not help end-to-end.

---

## Batch 0 (prior sessions, summarized from README/known_results.json)

### 0.1 Specialized FP16 Triton GEMV (1536 -> 8960)

- Hypothesis: a shape-specialized one-program-per-row Triton GEMV beats
  `F.linear` for the decode-time (M=1) gate/up projection shape.
- Microbenchmark: PyTorch 160.86 us vs Triton 111.57 us -> 1.44x. Correct.
- End-to-end (replacing gate_proj/up_proj independently): **regression**
  (~21.5 -> ~19.9 tok/s in a slower Colab state).
- **Rejected** as a standalone per-module replacement. Cause: wrapper/
  launch/allocation overhead around two independent kernel calls erased
  the kernel-level win. This is what motivated fusion (0.2) and, in this
  batch, coarse-grained replacement instead of per-op swaps.

### 0.2 Fused gate+up+SiLU+mul (FP16)

- Hypothesis: fusing gate_proj + up_proj + SiLU + multiply into one
  kernel removes 3 extra launches/allocations per layer versus computing
  them as separate PyTorch ops.
- Microbenchmark: PyTorch 431 us vs fused Triton 213.42 us -> 2.02x. Correct.
- End-to-end A/B/A: PyTorch A 21.00, Fused 22.22, PyTorch B 22.40 tok/s;
  baseline avg 21.70 -> **1.024x (+2.4%)**.
- **Kept** (first positive end-to-end result) but far short of 2x --
  down_proj and everything else in the layer was still FP16 PyTorch,
  and per-op Python overhead was still present since only the MLP's
  first half was touched.

### 0.3 INT8 weight-only GEMV (kernel-level, gate projection)

- Hypothesis: halving weight bytes read (per-row symmetric INT8,
  register dequant) helps further since decode is memory-bandwidth
  bound, not compute bound (M=1).
- Microbenchmark: PyTorch FP16 267.27 us, Triton FP16 112.44 us, Triton
  INT8 v1 75.20 us, INT8 v2 71.09 us -> INT8 vs FP16-Triton ~1.58x.
  Correct locally.

### 0.4 Fused INT8 gate/up MLP

- Microbenchmark: PyTorch FP16 325.09 us vs fused INT8 Triton 131.78 us
  -> 2.47x vs PyTorch, 1.82x vs the FP16-fused kernel. Max error 0.119,
  mean error 0.005, allclose True.
- End-to-end, INT8 fused MLP across all 28 layers only:
  PyTorch A 19.27, INT8 fused 19.03, PyTorch B 18.49 tok/s; baseline avg
  18.88 -> **1.008x (+0.8%)**.
- **Rejected as sufficient on its own.** A 1.82x kernel win on the
  single most-improved op produced under 1% end-to-end change. This is
  the clearest prior evidence that per-operator optimization, however
  good in isolation, is not the lever that moves tokens/sec -- confirmed
  again below.

### 0.5 INT8 lm_head

- Microbenchmark: PyTorch 3689.36 us vs INT8 Triton 1597.06 us -> 2.31x
  locally (lm_head measured at ~16% of CUDA time in profiling, one of
  the single largest ops in the model due to the ~152k vocabulary).
- Combined with 0.4 (fused INT8 MLP + INT8 lm_head), end-to-end:
  PyTorch A 22.04, Optimized 20.34, PyTorch B 21.33; baseline avg 21.68
  -> **0.938x (-6.2%), a regression.**
- **Rejected in that form.** Two individually-fast kernels combined and
  made things *worse* end-to-end. This is the central problem this
  batch is designed to diagnose: profiling with the fused INT8 MLP
  enabled still showed `aten::mm` at ~39% of CUDA time (i.e. most of the
  model was still FP16 PyTorch dispatch) plus "substantial CPU
  overhead" and "many thousands of small operations and launches" --
  the bottleneck was never purely which kernel computes a given GEMV,
  it was the volume of Python-dispatched, non-fused work surrounding
  every op, which two isolated kernel swaps cannot fix and can even
  make marginally worse (extra dtype branching, extra Python calls).

**Conclusion carried into this batch:** stop replacing individual
`nn.Linear` modules piecemeal. Attack the two things Amdahl's law says
must dominate given these numbers -- (a) the volume of Python/launch
overhead per decode step, and (b) memory traffic across *all* weight
matrices at once, not one at a time -- and measure both together.

---

## Batch 1 (this session's code; **numbers below are PENDING** -- see note)

> **This entire batch was implemented in an environment with no CUDA
> GPU available.** Every hypothesis, implementation, and microbenchmark
> claim in previous batches was graduated only after a real same-session
> A/B/A run; that discipline is kept here by *not* filling in fake
> numbers. The code is complete and self-tests its own correctness
> gates, but the "end-to-end result" and "kept/rejected" columns below
> are placeholders until `python run_t4_experiments.py` is actually run
> on the Tesla T4 and its `results/run_<timestamp>.json` + summary table
> are available. Update this section from that output, not from
> expectation.

Run with:

```bash
pip install -r requirements.txt
python run_t4_experiments.py            # full run (~15-20 min)
python run_t4_experiments.py --quick    # smoke test (~2 min) to sanity check the harness first
```

### 1.1 Manual greedy decode loop + DynamicCache (variant B)

- Hypothesis: `model.generate()`'s Python-side machinery (logits
  processors, stopping criteria, sampling dispatch, repeated
  `model_kwargs` bookkeeping) is itself a meaningful fraction of
  per-token wall time, independent of the forward pass or any kernel
  change. instructions.md explicitly asks for this comparison.
- Implementation: `decode_loop.greedy_decode_dynamic` -- calls the real
  `model(...)` forward directly with `DynamicCache`, replicating
  `generate()`'s exact greedy-argmax semantics (including
  `num_logits_to_keep=1`, which `generate()` sets automatically and
  which the manual loop must match to avoid an unfair prefill penalty).
- Correctness gate: `manual_loop_dynamic_matches_generate` -- must
  produce a token-for-token identical continuation to
  `model.generate(do_sample=False)`. Gates variants B and E.
- End-to-end result: **PENDING**
- Kept/rejected: **PENDING**

### 1.2 Manual greedy decode loop + StaticCache (variant C)

- Hypothesis: `DynamicCache`'s `torch.cat`-per-step growth is itself
  overhead beyond plain Python dispatch; a preallocated, fixed-address
  `StaticCache` (HF's own class, not reimplemented) should be at least
  as fast and is a prerequisite for CUDA graph capture (1.3).
- Implementation: `decode_loop.greedy_decode_static`, reusing one
  `StaticCache` instance across runs via `.reset()` (fresh instances
  would break the static addresses CUDA graphs depend on).
- Correctness gate: `manual_loop_static_matches_generate`. Gates
  variants C, D, F.
- End-to-end result: **PENDING**
- Kept/rejected: **PENDING**

### 1.3 `torch.compile(mode="reduce-overhead")` decode step (variant D)

- Hypothesis: this is the highest-leverage lever suggested by Amdahl's
  law given prior profiling ("many thousands of small operations and
  launches" + "substantial CPU overhead"). Rather than hand-rolling raw
  `torch.cuda.CUDAGraph` capture/replay (high bug risk with no GPU
  available to test against), this uses PyTorch's own documented
  static-cache + compile recipe, which captures the entire per-token
  decoder forward into one or more CUDA graphs and replays them,
  eliminating Python dispatch overhead for the graphed region almost
  entirely.
- Implementation: `decode_loop.make_compiled_decode_step`, with an
  automatic fallback cascade (`fullgraph=True` -> `fullgraph=False` ->
  eager StaticCache) so the run never silently substitutes an easier
  measurement without labeling it -- `compile_modes` in the saved JSON
  records exactly what happened.
- End-to-end result: **PENDING**
- Kept/rejected: **PENDING**

### 1.4 Full weight-only INT8 decode path, fused QKV + fused gate/up (variant E)

- Hypothesis: batch 0's INT8 experiments only quantized the MLP and
  then separately the lm_head; the profiler showed the *rest* of the
  model (`aten::mm` ~39%, Q/K/V/O ~8-9%) was still FP16 PyTorch. This
  batch quantizes every decode-heavy matrix at once -- Q, K, V (fused
  into a single kernel launch since they share the same input
  activation), O, gate+up (fused), down, and lm_head -- so the
  memory-traffic reduction is coarse-grained across the whole layer
  instead of one lonely kernel improvement buried in an otherwise-FP16
  stack.
- Implementation: `quant.py` (per-row symmetric INT8), `kernels.py`
  (`int8_gemv`, `fused_int8_gate_up`), `optimized_model.py`
  (`Int8QwenAttention`, `Int8QwenMLP`, `Int8LMHead` -- all delegate to
  the untouched original module for prefill/non-decode shapes; RoPE,
  cache update, and SDPA attention are NOT reimplemented, they call
  HF's own `apply_rotary_pos_emb`/`repeat_kv` so the only new/risky
  surface area is the projection math).
- Quantization footprint: printed by the run script
  (`quantization_report`), expected close to 2x smaller weight bytes
  across all quantized matrices.
- Correctness gates: `int8_gemv_kernel[*]`, `fused_int8_gate_up_kernel`
  (kernel arithmetic vs a pure-PyTorch INT8 reference, tight tolerance)
  and `int8_quality` (perplexity ratio on a fixed teacher-forced text
  sample + greedy continuation token overlap on a fresh prompt --
  *not* a single `allclose` on one random projection, per
  instructions.md).
- End-to-end result: **PENDING**
- Kept/rejected: **PENDING**

### 1.5 Compiled StaticCache + full INT8 combined (variant F)

- Hypothesis: 1.3 (overhead) and 1.4 (bandwidth) are largely orthogonal
  levers; combining them should be closer to multiplicative than
  additive and is this batch's primary candidate for reaching >=2x.
- Caveat logged honestly in advance: our INT8 modules call raw Triton
  kernels from plain Python wrapper functions
  (`kernels.int8_gemv`/`fused_int8_gate_up`). `torch.compile` support
  for user-defined Triton kernel calls is real in recent PyTorch but not
  guaranteed to produce a single unbroken graph the way pure-PyTorch
  variant D might. The same fallback cascade as 1.3 applies; if
  `fullgraph=True` fails here specifically, that is itself useful
  information about where the "coarse-grained fusion" boundary needs to
  move next (e.g. writing the projection+dequant as a registered
  `torch.library` custom op instead of a bare kernel launch).
- End-to-end result: **PENDING**
- Kept/rejected: **PENDING**

---

## How to fill this in

After running `python run_t4_experiments.py` on the T4:

1. Take the printed SUMMARY table and the saved
   `results/run_<timestamp>.json`.
2. Replace each **PENDING** above with the actual median tok/s, the
   speedup vs. that variant's local (sandwiched) baseline, and a
   kept/rejected verdict using the same bar as batch 0: kept only if it
   improved end-to-end throughput without failing a correctness/quality
   gate.
3. If profiling was captured for the best variant, summarize what
   still dominates CUDA time and feed that into NEXT_STEPS.md.
