"""
Correctness and quality gates.

Per CLAUDE.md / instructions.md methodology: nothing gets benchmarked for
speed until it has passed correctness here, and INT8 is judged on
perplexity + greedy-token-overlap, not a single allclose call.

`test_quant_math()` needs only PyTorch (CPU is fine) and is meant to be
run in *any* environment as a fast sanity check. Everything else needs a
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

# Fixed sample text for the perplexity / quality gate. Deliberately not
# downloaded at runtime so results are reproducible offline and the
# script has no extra network dependency.
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

    results = []
    torch.manual_seed(0)
    checks = [
        ("q_proj", layer.self_attn.q_proj.weight),
        ("k_proj", layer.self_attn.k_proj.weight),
        ("gate_proj", layer.mlp.gate_proj.weight),
        ("down_proj", layer.mlp.down_proj.weight),
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
    layer = model.model.layers[0]

    gate_w = layer.mlp.gate_proj.weight
    up_w = layer.mlp.up_proj.weight
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
    token sequence as `model.generate()` when no quantization/fusion is
    enabled -- any mismatch means the manual loop's cache/position
    wiring is wrong, independent of any optimization.
    """
    from decode_loop import greedy_decode_dynamic, greedy_decode_static, make_static_cache

    device = next(model.parameters()).device
    messages = [{"role": "user", "content": QUALITY_PROMPT}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.inference_mode():
        ref = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

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


def test_int8_quality(model, tokenizer, set_int8_fn, continuation_tokens=64):
    """
    Requires CUDA. The real quality gate for INT8: perplexity delta on a
    fixed teacher-forced text sample, plus greedy-continuation token
    overlap on a fresh prompt. Both exercise the decode-shape path.
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

    set_int8_fn(model, False)
    fp16_out = greedy_decode_dynamic(model, inputs["input_ids"], continuation_tokens)

    set_int8_fn(model, True)
    int8_out = greedy_decode_dynamic(model, inputs["input_ids"], continuation_tokens)

    set_int8_fn(model, False)

    prompt_len = inputs["input_ids"].shape[1]
    fp16_cont = fp16_out[0, prompt_len:]
    int8_cont = int8_out[0, prompt_len:]

    match_mask = fp16_cont == int8_cont
    overlap_pct = match_mask.float().mean().item() * 100.0

    first_divergence = None
    for i, m in enumerate(match_mask.tolist()):
        if not m:
            first_divergence = i
            break

    fp16_text = tokenizer.decode(fp16_cont, skip_special_tokens=True)
    int8_text = tokenizer.decode(int8_cont, skip_special_tokens=True)

    passed = ppl_ratio < 1.15 and overlap_pct > 50.0

    return [
        {
            "name": "int8_quality",
            "passed": bool(passed),
            "details": {
                "perplexity_fp16": ppl_fp16,
                "perplexity_int8": ppl_int8,
                "perplexity_ratio": ppl_ratio,
                "greedy_token_overlap_pct": overlap_pct,
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


if __name__ == "__main__":
    # CPU-safe subset only, for a quick sanity check without a GPU.
    results = test_quant_math()
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"[{status}] {r['name']}: {r['details']}")
