"""
Correctness and quality gates.

Per CLAUDE.md / instructions.md methodology: nothing gets benchmarked for
speed until it has passed correctness here, and INT8 is judged on
perplexity + greedy-token-overlap, not a single allclose call.

test_quant_math() needs only PyTorch (CPU is fine) and is meant to be
run in any environment as a fast sanity check. Everything else needs a
CUDA GPU with the real model loaded, since it exercises Triton kernels
and the actual decode shapes.
"""

import torch
import torch.nn.functional as F

from quant import (
quantize_int8_per_row,
dequantize_int8_per_row,
quantization_error,
)

Fixed sample text for the perplexity / quality gate. Deliberately not

downloaded at runtime so results are reproducible offline and the

script has no extra network dependency.

QUALITY_TEXT = (
"The history of computing hardware spans several generations of "
"technology, from mechanical calculators to vacuum tubes, transistors, "
"and integrated circuits. Each generation brought dramatic increases "
"in speed and reliability while reducing size and cost. Modern "
"processors contain billions of transistors switching billions of "
"times per second, enabling applications that would have seemed like "
"science fiction only a few decades ago. Understanding how memory "
"bandwidth, cache hierarchies, and instruction pipelines interact is "
"essential for anyone trying to make software run faster on real "
"hardware rather than just in theory."
)

QUALITY_PROMPT = "Explain how a CPU works in a few sentences."

def _unwrap_attn(attn):
"""
Return the underlying HF attention module. run_t4_experiments.py
installs the INT8 wrappers (Int8QwenAttention) before this suite
runs, so layer.self_attn may already be a wrapper that keeps the
real module at .original. Works whether or not it has been wrapped.
"""
return getattr(attn, "original", attn)

def _unwrap_mlp(mlp):
"""Return the underlying HF MLP module, wrapped (Int8QwenMLP) or not."""
return getattr(mlp, "original", mlp)

def test_quant_math():
"""
Pure-PyTorch, CPU-safe: checks the INT8 quantize/dequantize math is
self-consistent and bounded, independent of any kernel or GPU.
"""
torch.manual_seed(0)
results = []

# Real gate/up, down, and K/V projection shapes, plus a vocab-sized
# stand-in (20k rows instead of the full 151936) so this stays cheap
# enough to run on CPU with no GPU present.
for n, k in [(8960, 1536), (1536, 8960), (256, 1536), (20000, 1536)]:
    w = torch.randn(n, k) * 0.02
    qw, scale = quantize_int8_per_row(w)

    assert qw.dtype == torch.int8
    assert qw.shape == (n, k)
    assert scale.shape == (n,)
    assert qw.min().item() >= -127 and qw.max().item() <= 127

    max_err, mean_err, max_rel_err = quantization_error(w, qw, scale)
    # per-row symmetric INT8 with 127 levels: worst-case quantization
    # step is scale/2 in absolute terms.
    max_scale = scale.max().item()
    passed = max_err <= max_scale * 0.51

    results.append(
        {
            "name": f"quant_math[{n}x{k}]",
            "passed": bool(passed),
            "details": {
                "max_abs_error": max_err,
                "mean_abs_error": mean_err,
                "max_relative_error": max_rel_err,
            },
        }
    )

return results

def _cuda_gemv_correctness(kernels_module, weight, x):
from quant import QuantizedWeightINT8

y_ref_fp16 = F.linear(x, weight)

qw = QuantizedWeightINT8(weight)
y_int8 = kernels_module.int8_gemv(x, qw.qweight, qw.scale)

y_ref_int8_math = dequantize_int8_per_row(qw.qweight, qw.scale) @ x.to(torch.float32)

diff_vs_kernel_math = (y_int8.to(torch.float32) - y_ref_int8_math).abs()
diff_vs_fp16 = (y_int8.to(torch.float32) - y_ref_fp16.to(torch.float32)).abs()

return {
    "kernel_vs_int8_reference_max_err": diff_vs_kernel_math.max().item(),
    "kernel_vs_fp16_max_err": diff_vs_fp16.max().item(),
    "kernel_vs_fp16_mean_err": diff_vs_fp16.mean().item(),
}

def test_int8_gemv_kernel(model):
"""
Requires CUDA. Checks the Triton INT8 GEMV kernel matches a pure
PyTorch INT8 reference computation (kernel-arithmetic correctness),
and reports (not gates on) drift vs the original FP16 weight (that's
expected quantization error, judged separately by the quality gate).
"""
import kernels

device = next(model.parameters()).device
layer = model.model.layers[0]
attn = _unwrap_attn(layer.self_attn)
mlp = _unwrap_mlp(layer.mlp)

results = []
torch.manual_seed(0)
checks = [
    ("q_proj", attn.q_proj.weight),
    ("k_proj", attn.k_proj.weight),
    ("gate_proj", mlp.gate_proj.weight),
    ("down_proj", mlp.down_proj.weight),
]

for name, weight in checks:
    k = weight.shape[1]
    x = torch.randn(k, device=device, dtype=torch.float16)
    stats = _cuda_gemv_correctness(kernels, weight, x)
    passed = stats["kernel_vs_int8_reference_max_err"] < 0.5
    results.append(
        {
            "name": f"int8_gemv_kernel[{name}]",
            "passed": bool(passed),
            "details": stats,
        }
    )

return results

def test_fused_kernels(model):
"""Requires CUDA. Checks fused INT8 gate/up kernel arithmetic."""
import kernels
from quant import QuantizedWeightINT8

device = next(model.parameters()).device
mlp = _unwrap_mlp(model.model.layers[0].mlp)

gate_w = mlp.gate_proj.weight
up_w = mlp.up_proj.weight
k = gate_w.shape[1]

torch.manual_seed(0)
x = torch.randn(k, device=device, dtype=torch.float16)

gate_ref = F.linear(x, gate_w)
up_ref = F.linear(x, up_w)
fused_ref = F.silu(gate_ref.to(torch.float32)) * up_ref.to(torch.float32)

gate_q = QuantizedWeightINT8(gate_w)
up_q = QuantizedWeightINT8(up_w)

fused_int8 = kernels.fused_int8_gate_up(
    x, gate_q.qweight, gate_q.scale, up_q.qweight, up_q.scale
)

diff = (fused_int8.to(torch.float32) - fused_ref).abs()
max_err = diff.max().item()
mean_err = diff.mean().item()

return [
    {
        "name": "fused_int8_gate_up_kernel",
        "passed": bool(max_err < 2.0),
        "details": {"max_abs_error": max_err, "mean_abs_error": mean_err},
    }
]

def test_manual_loop_matches_generate(model, tokenizer, max_new_tokens=32):
"""
Requires CUDA. Variant B/C sanity gate: the hand-rolled greedy loop
(DynamicCache and StaticCache) must reproduce the exact same greedy
token sequence as model.generate() when no quantization/fusion is
enabled -- any mismatch means the manual loop's cache/position
wiring is wrong, independent of any optimization.
"""
from decode_loop import greedy_decode_dynamic, greedy_decode_static, make_static_cache
from transformers import GenerationConfig

device = next(model.parameters()).device
messages = [{"role": "user", "content": QUALITY_PROMPT}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(device)

# The manual loops are a bare greedy argmax with NO logits processing.
# Qwen2.5-1.5B-Instruct ships a generation_config that turns on sampling
# *and* repetition_penalty=1.05; `min_new_tokens` additionally installs an
# EOS-suppression processor. Any of those makes `generate()` diverge from
# a plain argmax loop even at do_sample=False (repetition_penalty is a
# logits *processor*, applied regardless of do_sample). Build a fully
# neutral config so the reference is exactly "greedy argmax, no
# processors, no early stop" -- which is what the loops implement.
greedy_cfg = GenerationConfig(
    max_new_tokens=max_new_tokens,
    do_sample=False,
    num_beams=1,
    repetition_penalty=1.0,
    temperature=None,
    top_p=None,
    top_k=None,
    eos_token_id=None,  # no EOS stopping, and no min-new-tokens processor
    pad_token_id=tokenizer.pad_token_id,
    use_cache=True,
)

with torch.inference_mode():
    ref = model.generate(**inputs, generation_config=greedy_cfg)

dyn = greedy_decode_dynamic(model, inputs["input_ids"], max_new_tokens)

prompt_len = inputs["input_ids"].shape[1]
static_cache = make_static_cache(model, max_cache_len=prompt_len + max_new_tokens)
stat = greedy_decode_static(model, inputs["input_ids"], max_new_tokens, static_cache)

dyn_match = torch.equal(ref, dyn)
stat_match = torch.equal(ref, stat)

return [
    {
        "name": "manual_loop_dynamic_matches_generate",
        "passed": bool(dyn_match),
        "details": {"exact_match": dyn_match},
    },
    {
        "name": "manual_loop_static_matches_generate",
        "passed": bool(stat_match),
        "details": {"exact_match": stat_match},
    },
]

@torch.inference_mode()
def _teacher_forced_nll(model, token_ids):
"""
One-token-at-a-time teacher-forced negative log-likelihood, using the
real per-step decode path (seq_len=1 after the first token) so that
INT8 modules actually take their fast path during this measurement
instead of silently falling back to the FP16 prefill path.
"""
from transformers import DynamicCache

device = token_ids.device
past_key_values = DynamicCache()

prefix = token_ids[:, :1]
out = model(input_ids=prefix, past_key_values=past_key_values, use_cache=True)
past_key_values = out.past_key_values

total_nll = 0.0
count = 0
cur_pos = 1  # position of the next token to feed (prefix already occupies position 0)

for i in range(1, token_ids.shape[1]):
    target = token_ids[:, i]

    logits = out.logits[:, -1, :].to(torch.float32)
    log_probs = F.log_softmax(logits, dim=-1)
    nll = -log_probs[0, target.item()]
    total_nll += nll.item()
    count += 1

    next_input = token_ids[:, i : i + 1]
    cache_position = torch.tensor([cur_pos], device=device, dtype=torch.long)
    out = model(
        input_ids=next_input,
        past_key_values=past_key_values,
        use_cache=True,
        cache_position=cache_position,
    )
    past_key_values = out.past_key_values
    cur_pos += 1

return total_nll / count

@torch.inference_mode()
def _teacher_forced_preds(model, full_ids, prompt_len, n):
"""
argmax(logits) for positions [prompt_len, prompt_len + n), each
conditioned on the TRUE prefix full_ids[:, :pos] (teacher forcing --
the context is held identical regardless of what the model would have
generated on its own). Uses the real per-step decode path (seq_len=1)
so INT8 modules take their fast path. Returns a CPU LongTensor [n].

This is the robust replacement for a free-running greedy-continuation
match: free-running greedy decode is chaotic, so a single early token
flip -- inevitable for any non-bitexact model -- forces every later
token to differ, making that metric measure perturbation sensitivity
rather than quality. Teacher forcing asks the decision-level question
directly: given the same context, does the model pick the same token?
"""
from transformers import DynamicCache

device = full_ids.device
past_key_values = DynamicCache()

cache_position = torch.arange(prompt_len, device=device)
out = model(
    input_ids=full_ids[:, :prompt_len],
    past_key_values=past_key_values,
    use_cache=True,
    cache_position=cache_position,
    num_logits_to_keep=1,
)
past_key_values = out.past_key_values
preds = [out.logits[:, -1, :].argmax(dim=-1)]  # predicts position prompt_len

for j in range(1, n):
    pos = prompt_len + j - 1  # feed the true token occupying this position
    cache_position = torch.tensor([pos], device=device, dtype=torch.long)
    out = model(
        input_ids=full_ids[:, pos : pos + 1],
        past_key_values=past_key_values,
        use_cache=True,
        cache_position=cache_position,
        num_logits_to_keep=1,
    )
    past_key_values = out.past_key_values
    preds.append(out.logits[:, -1, :].argmax(dim=-1))  # predicts position pos + 1

return torch.cat(preds).cpu()

def test_int8_quality(model, tokenizer, set_int8_fn, continuation_tokens=64):
"""
Requires CUDA. The real quality gate for INT8:

  1. Perplexity ratio on a fixed teacher-forced text sample
     (decode-shape path), and
  2. Teacher-forced top-1 agreement: given the identical context (the
     FP16 greedy continuation), does the INT8 model pick the same
     argmax token at each position?

A free-running INT8 greedy continuation is still generated and its
overlap reported, but it is NOT gated on -- see `_teacher_forced_preds`
for why that number is close to meaningless for a slightly-perturbed
model.
"""
device = next(model.parameters()).device
ids = tokenizer(QUALITY_TEXT, return_tensors="pt").input_ids.to(device)

set_int8_fn(model, False)
nll_fp16 = _teacher_forced_nll(model, ids)

set_int8_fn(model, True)
nll_int8 = _teacher_forced_nll(model, ids)

set_int8_fn(model, False)

ppl_fp16 = float(torch.exp(torch.tensor(nll_fp16)))
ppl_int8 = float(torch.exp(torch.tensor(nll_int8)))
ppl_ratio = ppl_int8 / ppl_fp16

from decode_loop import greedy_decode_dynamic

messages = [{"role": "user", "content": QUALITY_PROMPT}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(device)
prompt_len = inputs["input_ids"].shape[1]

# FP16 greedy continuation -> the fixed reference token path.
set_int8_fn(model, False)
fp16_out = greedy_decode_dynamic(model, inputs["input_ids"], continuation_tokens)
ref_tokens = fp16_out[0, prompt_len:].cpu()

# Teacher-forced top-1 agreement on that fixed context.
set_int8_fn(model, False)
fp16_preds = _teacher_forced_preds(model, fp16_out, prompt_len, continuation_tokens)
set_int8_fn(model, True)
int8_preds = _teacher_forced_preds(model, fp16_out, prompt_len, continuation_tokens)
set_int8_fn(model, False)

# Harness sanity: teacher-forcing the FP16 model on its own greedy
# output must reproduce it (guards the measurement, not INT8).
fp16_self_agreement_pct = (fp16_preds == ref_tokens).float().mean().item() * 100.0
tf_agreement_pct = (int8_preds == ref_tokens).float().mean().item() * 100.0

agree_mask = (int8_preds == ref_tokens).tolist()
first_divergence = None
for i, same in enumerate(agree_mask):
    if not same:
        first_divergence = i
        break

# Free-running INT8 continuation -- reported for eyeballing only.
set_int8_fn(model, True)
int8_out = greedy_decode_dynamic(model, inputs["input_ids"], continuation_tokens)
set_int8_fn(model, False)
int8_cont = int8_out[0, prompt_len:].cpu()
free_running_overlap_pct = (ref_tokens == int8_cont).float().mean().item() * 100.0

fp16_text = tokenizer.decode(ref_tokens, skip_special_tokens=True)
int8_text = tokenizer.decode(int8_cont, skip_special_tokens=True)

harness_ok = fp16_self_agreement_pct > 99.0
passed = harness_ok and ppl_ratio < 1.15 and tf_agreement_pct >= 90.0

return [
    {
        "name": "int8_quality",
        "passed": bool(passed),
        "details": {
            "perplexity_fp16": ppl_fp16,
            "perplexity_int8": ppl_int8,
            "perplexity_ratio": ppl_ratio,
            "teacher_forced_top1_agreement_pct": tf_agreement_pct,
            "fp16_self_agreement_pct": fp16_self_agreement_pct,
            "free_running_greedy_overlap_pct": free_running_overlap_pct,
            "first_divergence_index": first_divergence,
            "fp16_continuation": fp16_text,
            "int8_continuation": int8_text,
        },
    }
]

def run_all(model, tokenizer, set_int8_fn):
"""
Runs every CUDA-dependent test plus the CPU-safe math test, catching
exceptions per-test so one failure doesn't hide the rest.
"""
all_results = []

def _safe(fn, *args):
    try:
        all_results.extend(fn(*args))
    except Exception as e:  # noqa: BLE001 - want to record any failure
        all_results.append(
            {
                "name": fn.__name__,
                "passed": False,
                "details": {"exception": f"{type(e).__name__}: {e}"},
            }
        )

_safe(test_quant_math)
_safe(test_int8_gemv_kernel, model)
_safe(test_fused_kernels, model)
_safe(test_manual_loop_matches_generate, model, tokenizer)
_safe(test_int8_quality, model, tokenizer, set_int8_fn)

return all_results

if name == "main":
# CPU-safe subset only, for a quick sanity check without a GPU.
results = test_quant_math()
for r in results:
status = "PASS" if r["passed"] else "FAIL"
print(f"[{status}] {r['name']}: {r['details']}")