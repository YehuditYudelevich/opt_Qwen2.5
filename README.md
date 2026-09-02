# opt-qwen

Small-scale GPU inference optimization project inspired by the idea of a
hardware-aware optimization stack.

## Goal

Optimize `Qwen/Qwen2.5-1.5B-Instruct` autoregressive decode on the same
NVIDIA Tesla T4 GPU and target:

> **>= 2x end-to-end throughput vs a clean PyTorch FP16 baseline**

The project deliberately focuses on understanding and implementing low-level
optimizations rather than producing an artificial benchmark win.

**Status: target met.** INT8 weight-only quantization + fused Triton
kernels + StaticCache + `torch.compile` CUDA graphs measured ~3.75x
end-to-end decode throughput on Tesla T4, same-session A/B/A. See
[RESULTS.md](RESULTS.md).

## Current target

- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- GPU: NVIDIA Tesla T4
- Compute capability: 7.5
- Precision: FP16
- Batch size: 1
- Decode: autoregressive, KV cache enabled
- Hidden size: 1536
- MLP intermediate size: 8960
- Transformer layers: 28

## What profiling found

Decode is dominated by Linear/GEMV-like operations.

Important shapes:

- `gate_proj` / `up_proj`: `[1,1536] x [1536,8960]`
- `down_proj`: `[1,8960] x [8960,1536]`
- attention Q/O projections: roughly `1536 -> 1536`
- K/V projections: roughly `1536 -> 256`
- `lm_head`: `1536 -> 151936`

The original profiling showed that Linear/GEMV operations account for the
majority of GPU time during decode.

## Experiments so far

Full experiment log (hypothesis / implementation / microbenchmark /
end-to-end result / kept-or-rejected / why) lives in **[RESULTS.md](RESULTS.md)**.
Summary of the arc so far:

1. A specialized FP16 Triton GEMV was ~1.44x faster than PyTorch in
   isolation, but replacing `gate_proj`/`up_proj` independently
   **regressed** end-to-end -- wrapper/launch overhead erased the win.
2. Fusing gate+up+SiLU+mul into one kernel gave a genuine but small
   end-to-end gain (+2.4%).
3. INT8 weight-only quantization of the fused MLP was ~1.82x faster in
   isolation but only **+0.8%** end-to-end; adding an INT8 lm_head on
   top **regressed end-to-end by 6.2%** even though both kernels were
   individually much faster than PyTorch.

That last result is the central problem this project is solving:
profiling showed most of the model was still FP16 PyTorch dispatch
(`aten::mm` ~39% of CUDA time) plus "many thousands of small operations
and launches" and substantial CPU overhead -- so no amount of
one-operator-at-a-time kernel swapping was going to reach 2x. See
**[NEXT_STEPS.md](NEXT_STEPS.md)** for the current diagnosis and
**[instructions.md](instructions.md)** for the full methodology this
project follows.

## Important lesson

Do not claim an optimization from a microbenchmark alone.

Every meaningful change must be tested:

1. correctness
2. microbenchmark
3. end-to-end A/B/A on the same live GPU session
4. profiler before/after when needed

## Current batch: attacking overhead and bandwidth together

Rather than continuing to swap individual `nn.Linear` modules, the
current batch (see RESULTS.md "Batch 1") measures two levers together,
same session:

- **Overhead**: a manual greedy decode loop that bypasses
  `model.generate()`, first with `DynamicCache`, then with `StaticCache`,
  then with `torch.compile(mode="reduce-overhead")` (CUDA graphs) on top.
- **Bandwidth**: INT8 weight-only quantization applied to *every*
  decode-heavy matrix at once (fused QKV, fused gate+up, O, down,
  lm_head), not one operator in isolation.

Everything is gated on correctness first (`test_correctness.py`) --
Triton kernel arithmetic vs. a PyTorch reference, manual-loop token-exact
match against `generate()`, and INT8 quality via perplexity delta +
greedy-continuation token overlap, not a single `allclose` call.

## Reusable API

The optimizations above are packaged behind one call:

```python
from qwen_optimizer import optimize

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", torch_dtype=torch.float16, device_map="cuda")
model = optimize(model)

output = model.generate(**inputs, do_sample=False, max_new_tokens=64)
```

`optimize()` installs the INT8/Triton/StaticCache/compile path and wraps
`generate()` so supported greedy, batch=1 calls use it automatically
(including early stopping on EOS, like normal `generate()`); sampling,
beam search, batch>1, or anything else it isn't confident about
transparently falls back to the original HF `generate()`. Safe to call
more than once. See `qwen_optimizer/core.py` and `test_qwen_optimizer.py`.

## Usage

Install:

```bash
pip install -r requirements.txt
```

Run the full batch-1 experiment suite (correctness gate -> same-session
interleaved A/B/A across all variants -> profiling -> saved JSON ->
one summary table) -- this is the main entry point going forward:

```bash
python run_t4_experiments.py --quick   # ~2 min smoke test
python run_t4_experiments.py           # full run, ~15-20 min
```

Older, narrower scripts (still useful for quick isolated checks):

```bash
python microbench.py           # kernel-level FP16 microbenchmarks
python microbench_kernels.py   # INT8 kernel sweep: old vs new, BLOCK_K/num_warps
python benchmark.py --runs 5   # end-to-end A/B/A, FP16 fused-MLP only
python profile_model.py        # profile PyTorch baseline
python profile_model.py --fused  # profile fused FP16 MLP
python test_correctness.py     # CPU-only quantization-math sanity check
python test_qwen_optimizer.py  # qwen_optimizer package: optimize()/generate()/fallback checks
```

## Rules for future optimization work

- Keep the baseline switchable.
- Optimize decode separately from prefill.
- Do not intentionally weaken the baseline.
- Preserve model output quality / numerical correctness.
- Log every experiment.
- Prefer measured bottlenecks over guesses.
- End-to-end throughput is the real target.
