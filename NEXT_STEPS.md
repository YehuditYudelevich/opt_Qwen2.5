# Next Steps

## Status

Batch 1 target met: ~31.3 -> ~116 tok/s (~3.8x) on Tesla T4, variant F
(StaticCache + `torch.compile(reduce-overhead)` + full INT8), all
correctness/quality gates passing. See RESULTS.md "Batch 1".

A kernel-level rework of `int8_gemv_kernel`/`fused_int8_gate_up_kernel`
was attempted and reverted (introduced a real NaN, root cause understood,
fix existed but was never re-verified on hardware before the whole
attempt was discarded per explicit instruction -- see git history,
commit "Revert kernel optimization attempt..."). `kernels.py` is back to
the exact known-good state.

## Immediate action: validate batch 3 (profiling-methodology pass)

vLLM issue #421's methodology (Nsight-profile for redundant memory
traffic, idle-GPU stretches, small repeated CPU<->GPU round trips) was
applied to our own decode loop. Two of that checklist's items don't
apply to our architecture (no custom attention kernel; greedy batch=1
has no per-sequence sampling loop to batch) and are documented as such
in RESULTS.md "Batch 3" rather than silently skipped. Two real,
previously-unmeasured synchronization costs were found and fixed in
`decode_loop.greedy_decode_static`:

1. A per-step `torch.tensor([cur_pos], device=...)` reconstruction ->
   now a persistent buffer updated via `.fill_()`. Output-identical, no
   API change.
2. An unconditional per-step `next_token.item()` for EOS checking ->
   now gated by an optional `eos_check_interval` (default 1 = unchanged
   behavior). This one specifically affects `qwen_optimizer.optimize()`'s
   packaged API (which always resolves a real `eos_token_id`), NOT the
   `run_t4_experiments.py` harness that produced the confirmed ~116
   tok/s (it never passes `eos_token_ids`) -- so this cost was real but
   previously invisible to our own benchmarks.

Run, in order:

```bash
python test_correctness.py         # must still pass (CPU-safe subset already reverified here)
python bench_decode_overhead.py    # sections A (synthetic), B (Fix 1 before/after), C (Fix 2 interval sweep)
python bench_vs_vllm.py --engine ours   # end-to-end, compare against the ~116 tok/s baseline
```

Report back:
1. `bench_decode_overhead.py` section B: does it confirm `torch.equal()`
   between before/after (it asserts this itself, so a crash there means
   stop and report immediately), and what's the measured speedup, if
   any, of `.fill_()` vs per-step `torch.tensor()`?
2. Section C: at what `eos_check_interval` (if any) does the time
   improvement become worth the token-overshoot? Is the overshoot even
   inside `max_new_tokens` in practice, or does natural EOS happen early
   enough that this barely matters for typical chat-length generations?
3. End-to-end tok/s via `qwen_optimizer.optimize()` before vs. after
   these changes are on `main`. Per RESULTS.md's own bar: 3.1 is
   output-identical so "keep" is just "did it not get slower"; 3.2
   defaults to unchanged behavior already, so there's nothing to revert
   there regardless -- the only live question is whether raising the
   default above 1 would ever be worth recommending, which needs
   real numbers, not intuition.

## Occupancy / warp-stall investigation (needs `ncu`, not done here)

Applying the same *tool* issue #421 used (Nsight Compute), not just the
same idea, since occupancy/warp-stall analysis isn't something
`torch.profiler`'s CPU/CUDA activity tables expose. Self-contained repro
(doesn't depend on any removed/reverted files):

```bash
ncu --kernel-name regex:"int8_gemv_kernel|fused_int8_gate_up_kernel" \
    --metrics sm__warps_active.avg.pct_of_peak_sustained_active,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed \
    -o ncu_int8_kernels \
    python3 -c "
import torch
from kernels import int8_gemv, fused_int8_gate_up
from quant import QuantizedWeightINT8

x = torch.randn(1536, device='cuda', dtype=torch.float16)
w_down = (torch.randn(8960, 1536, device='cuda') * 0.02).to(torch.float16)
qw_down = QuantizedWeightINT8(w_down)
x_up = torch.randn(1536, device='cuda', dtype=torch.float16)
gate_w = (torch.randn(8960, 1536, device='cuda') * 0.02).to(torch.float16)
up_w = (torch.randn(8960, 1536, device='cuda') * 0.02).to(torch.float16)
qgate, qup = QuantizedWeightINT8(gate_w), QuantizedWeightINT8(up_w)

for _ in range(10):
    int8_gemv(x, qw_down.qweight, qw_down.scale)
    fused_int8_gate_up(x_up, qgate.qweight, qgate.scale, qup.qweight, qup.scale)
"
```

Read `sm__warps_active` (occupancy) and compare `dram__throughput` vs
`sm__throughput` (bandwidth- vs compute-bound, same framing issue #421
used: "15% compute, 50% bandwidth" was their smoking gun for a
memory-traffic bug). Report the three percentages back -- if occupancy
is already high and DRAM throughput is near peak, these GEMVs are
already near their memory-bandwidth roofline and further kernel tuning
has little room left; if occupancy is low, that's a real, separate
finding worth its own investigation before touching anything else.

## Deferred (do not build until measurement justifies it)

- **INT4 weight-only quantization**: only worth it if a fresh profile
  (post batch-3) shows memory bandwidth, not overhead, is still the
  binding constraint. Not indicated by anything found so far.
- **Finer-grained fusion** (RMSNorm+QKV, O-proj+residual): launch count
  stopped being the bottleneck once CUDA graphs capture the whole
  decode step; the remaining case would be memory traffic, a smaller
  effect. Revisit only if a fresh profile shows these ops still matter.
- **A custom attention kernel**: never flagged as a major cost in any
  profiling run so far (batch 0, batch 1, or this methodology pass);
  SDPA is left untouched.
- **Mirroring the `cache_position` persistent-buffer fix into
  `greedy_decode_dynamic`**: same idea would apply, not done in batch 3
  to keep blast radius minimal (that function isn't on the actual
  current optimized path -- only `greedy_decode_static` is). Low risk,
  low priority; revisit if `greedy_decode_dynamic` is ever back in a
  critical path.
