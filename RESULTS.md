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

## Batch 1 -- CONFIRMED on Tesla T4: target met

Overall result reported back from a real Colab T4 run: baseline
**~31.3 tok/s** -> optimized **~116 tok/s**, **~3.8x** end-to-end decode
speedup, all correctness/quality gates passing. This is variant F
(StaticCache + `torch.compile(mode="reduce-overhead")` + full INT8 with
fused QKV/gate-up). The >=2x target from instructions.md is met.

Per-variant (B/C/D/E individually, vs. F combined) breakdowns below are
still marked PENDING -- only the combined/final number above has been
reported back in enough detail to log here. Update the individual
entries from a saved `results/run_<timestamp>.json` if/when available.

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

## Batch 2 (kernel-level rework of int8_gemv_kernel / fused_int8_gate_up_kernel)

Starting point: with the batch-1 stack at ~116 tok/s, profiling showed
`int8_gemv_kernel` at ~38.5% of CUDA time and `fused_int8_gate_up_kernel`
at ~33%.

- Hypothesis (both 2.1 and 2.2 below): `BLOCK_K = triton.next_power_of_2(K)`
  combined with a single masked load over the whole row wastes a large,
  computable fraction of every GEMV on provably-zero padding, and holds
  that whole padded width as live per-program registers. For this
  model's actual K values: QKV-fused/O/lm_head (K=1536, BLOCK_K was
  2048) waste 25%; down_proj (K=8960, BLOCK_K was 16384) wastes 45.3%
  -- and `int8_gemv_kernel` covers all four of those projections, which
  is also why it outweighs the gate/up-only fused kernel in CUDA time.
- Implementation, common to both: reworked both kernels to walk K in
  fixed `BLOCK_K=256`-wide tiles (a Python-level, trace-time loop since
  K/BLOCK_K are `tl.constexpr`) instead of one `next_pow2(K)`-wide
  masked load. 256 evenly divides both 1536 and 8960, so for this model
  the compiled kernel has zero masked/padded lanes; a masked remainder
  tile is still generated for any K that doesn't divide evenly, so
  arbitrary shapes stay correct. `block_k`/`num_warps` exposed as
  keyword args on `int8_gemv()`/`fused_int8_gate_up()` (default
  `DEFAULT_BLOCK_K=256`, `DEFAULT_NUM_WARPS=4`, new module constants in
  `kernels.py`). Both wrapper function signatures are otherwise
  unchanged -- `optimized_model.py`/`test_correctness.py` call them
  positionally and need no changes.

### 2.1 Tiling + fp16 multiply -- REJECTED (real NaN, root-caused)

- Additional change on top of the tiling: multiply in fp16 (weight
  dequantized to fp16, not fp32), accumulating each tile's product into
  an fp32 running sum, intended to reproduce the "INT8 v2" change
  already validated in this project's history (instructions.md: 75.2us
  -> 71.1us from exactly this change).
- Result: **`test_int8_quality` produced `perplexity_int8 = NaN`**,
  reported back from a real T4 run. Isolated kernel correctness tests
  (`int8_gemv_kernel[*]`, `fused_int8_gate_up_kernel`) still passed.
- Root cause, confirmed by CPU-only reproduction (not just theory):
  the dequant scale is applied once, *after* the full reduction, so
  every per-tile product actually computed is `activation *
  raw_unscaled_int8_byte` (byte range +-127), not `activation *
  dequantized_weight`. Computed in fp16 (max ~65504), any activation
  channel with `|x| > ~515` overflows to `inf` the instant it hits a
  near-+-127 byte -- confirmed directly: `torch.tensor(800.0,
  dtype=float16) * torch.tensor(127, dtype=int8).to(float16)` ->
  `inf`, while the same multiply in fp32 -> `101600.0` (fine). fp32's
  headroom (~3.4e38) is why the original (batch-1) kernel, and the
  "INT8 v2" precedent it was modeled on, never hit this -- that
  precedent almost certainly applied the scale *before* multiplying
  (i.e. multiplied by the small dequantized weight, not the raw byte),
  which is a materially different, safer computation.
  Isolated kernel tests use `torch.randn(k)` (unit-variance, `|x|`
  rarely > ~4) and never approach the threshold; real decode-path
  activations do, specifically the two GEMV inputs that are NOT
  freshly RMSNorm'd: attention output before `o_proj`
  (`Int8QwenAttention`) and `SiLU(gate)*up` before `down_proj`
  (`Int8QwenMLP`). QKV's and lm_head's GEMV inputs are RMSNorm'd
  (bounded scale), consistent with the failure being isolated to the
  two paths above.
- Diagnostic added either way: `debug_int8_nan.py` hooks every layer's
  `self_attn`/`mlp`/`lm_head` for inf/nan on a real forward pass, and
  `optimized_model.DEBUG_ACTIVATION_STATS` (off by default, zero cost
  when disabled) records the actual max-abs activation feeding
  `o_proj`/`down_proj` per layer -- run this against real hardware to
  confirm/quantify the above rather than trust the reasoning alone.
- **Rejected.** fp16 multiply-of-raw-int8-byte is unsafe in general,
  independent of the tiling change.

### 2.2 Tiling only, fp32 multiply restored -- current candidate, PENDING verification

- Change from 2.1: reverted the multiply back to fp32 (matching the
  batch-1 kernel's numerics exactly), while keeping the tiled/
  no-masked-waste loop structure from 2.1. This isolates the tiling
  variable from the arithmetic-precision variable, per instruction --
  the NaN was caused by the fp16 multiply, not by tiling K instead of
  using one big masked load.
- Verified so far (CPU only, no GPU in this environment):
  - Tiled fp32 algorithm reproduces the untiled reference to within
    ~1e-4 to 1e-3 max error for K=1536, K=8960, and a deliberately
    non-dividing K=777 (remainder-tile path), matching the batch-1
    kernel's own precision -- tiling changes summation order, not
    correctness.
  - The exact forced-overflow case from 2.1 (`x=800`, weight byte
    `127`) produces no inf/nan when run through the tiled fp32
    algorithm (output `3232.0`, finite) using the same emulation
    harness that reproduced the bug in 2.1.
- **Not yet run on real hardware. Do not benchmark or mark
  kept/rejected until `python test_correctness.py`'s CUDA-dependent
  tests -- especially `int8_quality` -- pass on the T4.** Deliberately
  NOT re-attempting a "dequantize-then-multiply-in-fp16" variant that
  might avoid the overflow while keeping fp16's speed: unverifiable
  without GPU access and explicitly out of scope until the fp32
  version is confirmed correct first.
- Known-good rollback if this still doesn't pass: git commit `ff64322`
  (the exact kernel that produced the confirmed ~116 tok/s, 3.8x
  result) or the frozen `_int8_gemv_kernel_v1`/
  `_fused_int8_gate_up_kernel_v1` copies in `microbench_kernels.py`.
- Deliberately NOT done (applies to this whole batch), with reasoning:
  - **`triton.autotune`**: would let the runtime pick `BLOCK_K`/
    `num_warps` empirically, which is normally the right tool for
    exactly this question. Rejected here because these kernels are
    called from inside the `torch.compile(mode="reduce-overhead")`
    CUDA-graph-captured decode step (variant F / `qwen_optimizer`'s
    fast path) -- autotune's first-call dynamic benchmarking is not
    safe to run from inside graph capture. Used a static default plus
    an offline sweep script instead.
  - **DP4A / INT8 tensor cores**: T4's native int8xint8 throughput
    instructions require both operands to be int8; this scheme is
    weight-only (activations stay fp16), so they don't apply without
    also quantizing activations, which would change the quantization
    semantics the task said to preserve. Also, at M=1 (batch=1 decode)
    this is memory-bandwidth-bound, not compute-bound, so the multiply
    representation is a second-order effect either way -- consistent
    with (0.3)'s own historical numbers, where the fp16-vs-fp32
    multiply change was worth ~5%, not a multiple.
- Microbenchmark: **PENDING** `python microbench_kernels.py`.
- End-to-end result: **PENDING** `python run_t4_experiments.py` --
  only after `int8_quality` passes.
- Kept/rejected: **PENDING**.

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
4. For batch 2 specifically, in this order:
   a. `python test_correctness.py`'s CUDA-dependent tests, especially
      `int8_quality` -- must pass before anything below. If it doesn't,
      stop and report back; do not benchmark.
   b. (optional) `python debug_int8_nan.py` if anything still looks
      off, to pinpoint which layer/module.
   c. `python microbench_kernels.py` -- paste its BEST BLOCK_K/num_warps
      per shape here, so `kernels.DEFAULT_BLOCK_K`/`DEFAULT_NUM_WARPS`
      can be corrected from data if 256/4 isn't actually fastest.
   d. `python run_t4_experiments.py` for the end-to-end number, only
      after (a) passes.
