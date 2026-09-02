# Claude Code instructions for Mini-DOS

You are working on a GPU inference optimization research project.

## Objective

Achieve >=2x end-to-end autoregressive decode throughput for
`Qwen/Qwen2.5-1.5B-Instruct` on the same NVIDIA Tesla T4 GPU, while preserving
model quality.

## Current state

Read README.md before changing code.

Key measurements already established:

- Specialized FP16 Triton GEMV for 1536 -> 8960:
  ~1.44x faster than PyTorch in a same-session microbenchmark.
- Replacing gate/up separately caused an end-to-end regression.
- Fusing gate_proj + up_proj + SiLU + multiply:
  ~2.02x microbenchmark speedup.
- Integrated fused MLP:
  only ~2.4% end-to-end speedup in A/B/A.

## Required methodology

For each proposed optimization:

1. State the hypothesis.
2. Add or update a correctness test.
3. Measure the relevant microbenchmark.
4. Run end-to-end A/B/A in the same GPU session.
5. Re-profile if the end-to-end result is surprising.
6. Log result and conclusion.

Do not infer an end-to-end improvement from a kernel microbenchmark.

## Priority areas

Investigate, in order of measured value:

- down_proj
- full/partial MLP fusion
- weight-only INT8/INT4 GEMV
- lm_head
- kernel launch overhead
- memory traffic / bandwidth
- other profiler-supported bottlenecks

## Constraints

- Same GPU for baseline and optimized measurement.
- Batch=1 autoregressive decode is the primary target.
- Keep prefill fallback correct unless explicitly optimizing it.
- Do not intentionally choose a weak PyTorch baseline.
- Keep numerical correctness tests.
- Avoid introducing external optimized inference engines merely to claim a win;
  low-level implementation and understanding are the point of this project.
