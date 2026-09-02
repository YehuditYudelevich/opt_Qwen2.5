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
    """Return the original HF attention module when it is wrapped."""
    return getattr(attn, "original", attn)

def _unwrap_mlp(mlp):
    """Return the original HF MLP module when it is wrapped."""
    return getattr(mlp, "original", mlp)

def test_quant_math():
    """Check CPU-safe INT8 quantize/dequantize arithmetic and its error bound."""
    torch.manual_seed(0)
    results = []

    # Real gate/up, down, and K/V projection shapes, plus a vocab-sized
    # stand-in (20k rows instead of the full 151936) to keep this CPU-safe.
    for n, k in [(8960, 1536), (1536, 8960), (256, 1536), (20000, 1536)]:
        weight = torch.randn(n, k) * 0.02
        quantized_weight, scale = quantize_int8_per_row(weight)

        assert quantized_weight.dtype == torch.int8
        assert quantized_weight.shape == (n, k)
        assert scale.shape == (n,)
        assert quantized_weight.min().item() >= -127
        assert quantized_weight.max().item() <= 127

        max_err, mean_err, max_rel_err = quantization_error(
            weight, quantized_weight, scale
        )
        passed = max_err <= scale.max().item() * 0.51
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
    quantized_weight = QuantizedWeightINT8(weight)
    y_int8 = kernels_module.int8_gemv(
        x, quantized_weight.qweight, quantized_weight.scale
    )
    y_ref_int8_math = (
        dequantize_int8_per_row(quantized_weight.qweight, quantized_weight.scale)
        @ x.to(torch.float32)
    )

    diff_vs_kernel_math = (y_int8.to(torch.float32) - y_ref_int8_math).abs()
    diff_vs_fp16 = (y_int8.to(torch.float32) - y_ref_fp16.to(torch.float32)).abs()
    return {
        "kernel_vs_int8_reference_max_err": diff_vs_kernel_math.max().item(),
        "kernel_vs_fp16_max_err": diff_vs_fp16.max().item(),
        "kernel_vs_fp16_mean_err": diff_vs_fp16.mean().item(),
    }

def test_int8_gemv_kernel(model):
    """Check CUDA Triton INT8 GEMV results against a PyTorch INT8 reference."""
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
        x = torch.randn(weight.shape[1], device=device, dtype=torch.float16)
        stats = _cuda_gemv_correctness(kernels, weight, x)
        results.append(
            {
                "name": f"int8_gemv_kernel[{name}]",
                "passed": bool(stats["kernel_vs_int8_reference_max_err"] < 0.5),
                "details": stats,
            }
        )

    return results

def test_fused_kernels(model):
    """Check fused INT8 gate/up kernel arithmetic on CUDA."""
    import kernels
    from quant import QuantizedWeightINT8

    device = next(model.parameters()).device
    mlp = _unwrap_mlp(model.model.layers[0].mlp)
    gate_weight = mlp.gate_proj.weight
    up_weight = mlp.up_proj.weight
    torch.manual_seed(0)
    x = torch.randn(gate_weight.shape[1], device=device, dtype=torch.float16)

    fused_ref = F.silu(F.linear(x, gate_weight).to(torch.float32)) * F.linear(
        x, up_weight
    ).to(torch.float32)
    gate_q = QuantizedWeightINT8(gate_weight)
    up_q = QuantizedWeightINT8(up_weight)
    fused_int8 = kernels.fused_int8_gate_up(
        x, gate_q.qweight, gate_q.scale, up_q.qweight, up_q.scale
    )
    diff = (fused_int8.to(torch.float32) - fused_ref).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    return [{
        "name": "fused_int8_gate_up_kernel",
        "passed": bool(max_err < 2.0),
        "details": {"max_abs_error": max_err, "mean_abs_error": mean_err},
    }]

def test_manual_loop_matches_generate(model, tokenizer, max_new_tokens=32):
    """Ensure the manual dynamic/static greedy loops match neutral generate()."""
    from decode_loop import greedy_decode_dynamic, greedy_decode_static, make_static_cache
    from transformers import GenerationConfig

    device = next(model.parameters()).device
    messages = [{"role": "user", "content": QUALITY_PROMPT}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)
    greedy_cfg = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        repetition_penalty=1.0,
        temperature=None,
        top_p=None,
        top_k=None,
        eos_token_id=None,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )
    with torch.inference_mode():
        reference = model.generate(**inputs, generation_config=greedy_cfg)

    dynamic = greedy_decode_dynamic(model, inputs["input_ids"], max_new_tokens)
    prompt_len = inputs["input_ids"].shape[1]
    static_cache = make_static_cache(model, prompt_len + max_new_tokens)
    static = greedy_decode_static(
        model, inputs["input_ids"], max_new_tokens, static_cache
    )
    dynamic_match = torch.equal(reference, dynamic)
    static_match = torch.equal(reference, static)
    return [
        {
            "name": "manual_loop_dynamic_matches_generate",
            "passed": bool(dynamic_match),
            "details": {"exact_match": dynamic_match},
        },
        {
            "name": "manual_loop_static_matches_generate",
            "passed": bool(static_match),
            "details": {"exact_match": static_match},
        },
    ]

@torch.inference_mode()
def _teacher_forced_nll(model, token_ids):
    """Compute decode-path teacher-forced negative log likelihood."""
    from transformers import DynamicCache

    past_key_values = DynamicCache()
    out = model(
        input_ids=token_ids[:, :1], past_key_values=past_key_values, use_cache=True
    )
    past_key_values = out.past_key_values
    total_nll = 0.0
    count = 0

    for position in range(1, token_ids.shape[1]):
        target = token_ids[:, position]
        log_probs = F.log_softmax(out.logits[:, -1, :].to(torch.float32), dim=-1)
        total_nll -= log_probs[0, target.item()].item()
        count += 1
        out = model(
            input_ids=token_ids[:, position : position + 1],
            past_key_values=past_key_values,
            use_cache=True,
            cache_position=torch.tensor(
                [position], device=token_ids.device, dtype=torch.long
            ),
        )
        past_key_values = out.past_key_values

    return total_nll / count

@torch.inference_mode()
def _teacher_forced_preds(model, full_ids, prompt_len, n):
    """Return decode-path teacher-forced argmax predictions on a fixed context."""
    from transformers import DynamicCache

    past_key_values = DynamicCache()
    device = full_ids.device
    out = model(
        input_ids=full_ids[:, :prompt_len],
        past_key_values=past_key_values,
        use_cache=True,
        cache_position=torch.arange(prompt_len, device=device),
        num_logits_to_keep=1,
    )
    past_key_values = out.past_key_values
    preds = [out.logits[:, -1, :].argmax(dim=-1)]

    for step in range(1, n):
        position = prompt_len + step - 1
        out = model(
            input_ids=full_ids[:, position : position + 1],
            past_key_values=past_key_values,
            use_cache=True,
            cache_position=torch.tensor([position], device=device, dtype=torch.long),
            num_logits_to_keep=1,
        )
        past_key_values = out.past_key_values
        preds.append(out.logits[:, -1, :].argmax(dim=-1))

    return torch.cat(preds).cpu()

def test_int8_quality(model, tokenizer, set_int8_fn, continuation_tokens=64):
    """Gate INT8 quality by perplexity ratio and teacher-forced top-1 overlap."""
    from decode_loop import greedy_decode_dynamic

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
    messages = [{"role": "user", "content": QUALITY_PROMPT}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]

    fp16_out = greedy_decode_dynamic(model, inputs["input_ids"], continuation_tokens)
    ref_tokens = fp16_out[0, prompt_len:].cpu()
    fp16_preds = _teacher_forced_preds(
        model, fp16_out, prompt_len, continuation_tokens
    )
    set_int8_fn(model, True)
    int8_preds = _teacher_forced_preds(
        model, fp16_out, prompt_len, continuation_tokens
    )
    set_int8_fn(model, False)
    fp16_self_agreement_pct = (fp16_preds == ref_tokens).float().mean().item() * 100.0
    tf_agreement_pct = (int8_preds == ref_tokens).float().mean().item() * 100.0
    first_divergence = next(
        (index for index, same in enumerate((int8_preds == ref_tokens).tolist()) if not same),
        None,
    )

    set_int8_fn(model, True)
    int8_out = greedy_decode_dynamic(model, inputs["input_ids"], continuation_tokens)
    set_int8_fn(model, False)
    int8_cont = int8_out[0, prompt_len:].cpu()
    free_running_overlap_pct = (ref_tokens == int8_cont).float().mean().item() * 100.0
    passed = (
        fp16_self_agreement_pct > 99.0
        and ppl_ratio < 1.15
        and tf_agreement_pct >= 90.0
    )
    return [{
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
            "fp16_continuation": tokenizer.decode(ref_tokens, skip_special_tokens=True),
            "int8_continuation": tokenizer.decode(int8_cont, skip_special_tokens=True),
        },
    }]

def run_all(model, tokenizer, set_int8_fn):
    """Run every check, recording each exception as a failed result."""
    all_results = []

    def safe_run(fn, *args):
        try:
            all_results.extend(fn(*args))
        except Exception as error:  # noqa: BLE001 - record every test failure
            all_results.append(
                {
                    "name": fn.__name__,
                    "passed": False,
                    "details": {"exception": f"{type(error).__name__}: {error}"},
                }
            )

    safe_run(test_quant_math)
    safe_run(test_int8_gemv_kernel, model)
    safe_run(test_fused_kernels, model)
    safe_run(test_manual_loop_matches_generate, model, tokenizer)
    safe_run(test_int8_quality, model, tokenizer, set_int8_fn)
    return all_results

if __name__ == "__main__":
    results = test_quant_math()
    for result in results:
        status = "PASS" if result["passed"] else "FAIL"
        print(f"[{status}] {result['name']}: {result['details']}")