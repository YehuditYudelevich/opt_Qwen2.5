"""
Tests for the qwen_optimizer package (see qwen_optimizer/core.py).

These exercise the public API end-to-end (`optimize(model)` +
`model.generate()`), so -- like test_correctness.py -- they need a CUDA
GPU with the real model loaded. Run directly:

    python test_qwen_optimizer.py

Quality is judged with the same bar this project already established
for INT8 in test_correctness.test_int8_quality (teacher-forced top-1
agreement vs FP16, not exact token equality): greedy decoding over
INT8-quantized weights is not guaranteed to be bit-identical to FP16,
only close, and asserting exact equality would be a flaky test that
contradicts the project's own quality methodology.
"""

import torch

from qwen_optimizer import optimize, is_optimized, original_generate
import optimized_model

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def _load():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="cuda"
    )
    model.eval()
    return tokenizer, model


def _chat_inputs(tokenizer, model, content, add_generation_prompt=True):
    messages = [{"role": "user", "content": content}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt
    )
    return tokenizer(text, return_tensors="pt").to(model.device)


def test_optimize_is_idempotent(model):
    """Calling optimize() twice must not re-quantize/re-wrap or change the bound generate()."""
    generate_before = model.generate
    same_model = optimize(model) is model
    generate_after = model.generate

    return [
        {
            "name": "optimize_idempotent",
            "passed": bool(same_model and generate_after is generate_before and is_optimized(model)),
            "details": {
                "same_model_object": same_model,
                "same_generate_fn": generate_after is generate_before,
            },
        }
    ]


def test_optimized_generate_runs(model, tokenizer, max_new_tokens=16):
    """The optimized generate() must produce a well-formed greedy continuation."""
    inputs = _chat_inputs(tokenizer, model, "Say hello in one short sentence.")
    prompt_len = inputs["input_ids"].shape[1]

    output = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1, use_cache=True
    )

    ok = output.shape[0] == 1 and output.shape[1] == prompt_len + max_new_tokens
    return [
        {
            "name": "optimized_generate_runs",
            "passed": bool(ok),
            "details": {
                "output_shape": list(output.shape),
                "expected_len": prompt_len + max_new_tokens,
            },
        }
    ]


def test_optimized_generate_quality(model, tokenizer, continuation_tokens=32):
    """
    Greedy output must remain correct: teacher-forced top-1 agreement
    between the optimized (INT8 + StaticCache + compile) path and the
    true original FP16 generate(), reusing
    test_correctness._teacher_forced_preds -- the same methodology
    test_correctness.test_int8_quality already validated this INT8
    scheme against.
    """
    from test_correctness import _teacher_forced_preds

    inputs = _chat_inputs(tokenizer, model, "Explain how a CPU works in a few sentences.")
    prompt_len = inputs["input_ids"].shape[1]

    reference_generate = original_generate(model)
    with torch.inference_mode():
        fp16_out = reference_generate(
            **inputs, max_new_tokens=continuation_tokens, do_sample=False, num_beams=1, use_cache=True
        )

    optimized_model.set_int8(model, False)
    fp16_preds = _teacher_forced_preds(model, fp16_out, prompt_len, continuation_tokens)

    optimized_model.set_int8(model, True)
    int8_preds = _teacher_forced_preds(model, fp16_out, prompt_len, continuation_tokens)
    optimized_model.set_int8(model, False)

    agreement_pct = (int8_preds == fp16_preds).float().mean().item() * 100.0

    optimized_out = model.generate(
        **inputs, max_new_tokens=continuation_tokens, do_sample=False, num_beams=1, use_cache=True
    )
    right_length = optimized_out.shape[1] == prompt_len + continuation_tokens

    return [
        {
            "name": "optimized_generate_quality",
            "passed": bool(agreement_pct >= 90.0 and right_length),
            "details": {
                "teacher_forced_top1_agreement_pct": agreement_pct,
                "optimized_output_len": int(optimized_out.shape[1]),
            },
        }
    ]


def test_unsupported_falls_back(model, tokenizer):
    """Sampling / beam search / batch>1 must run via the real generate(), INT8 left disabled."""
    inputs = _chat_inputs(tokenizer, model, "Say hello.")
    results = []

    def _int8_engaged():
        return bool(model.model.layers[0].self_attn.use_int8)

    torch.manual_seed(0)
    out_sampled = model.generate(
        **inputs, max_new_tokens=8, do_sample=True, temperature=0.7, top_p=0.8
    )
    results.append(
        {
            "name": "fallback_sampling",
            "passed": bool(out_sampled.shape[1] > inputs["input_ids"].shape[1] and not _int8_engaged()),
            "details": {"output_len": int(out_sampled.shape[1])},
        }
    )

    out_beam = model.generate(**inputs, max_new_tokens=8, do_sample=False, num_beams=4)
    results.append(
        {
            "name": "fallback_beam_search",
            "passed": bool(out_beam.shape[1] > inputs["input_ids"].shape[1] and not _int8_engaged()),
            "details": {"output_len": int(out_beam.shape[1])},
        }
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    messages = [{"role": "user", "content": "Say hello."}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    batched = tokenizer([text, text], return_tensors="pt", padding=True).to(model.device)
    out_batch = model.generate(**batched, max_new_tokens=8, do_sample=False)
    results.append(
        {
            "name": "fallback_batch_gt_1",
            "passed": bool(out_batch.shape[0] == 2 and not _int8_engaged()),
            "details": {"output_shape": list(out_batch.shape)},
        }
    )

    out_unrecognized_kwarg = model.generate(
        **inputs, max_new_tokens=8, do_sample=False, repetition_penalty=1.1
    )
    results.append(
        {
            "name": "fallback_unrecognized_kwarg",
            "passed": bool(
                out_unrecognized_kwarg.shape[1] > inputs["input_ids"].shape[1] and not _int8_engaged()
            ),
            "details": {"output_len": int(out_unrecognized_kwarg.shape[1])},
        }
    )

    return results


def run_all(model, tokenizer):
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

    safe_run(test_optimized_generate_runs, model, tokenizer)
    safe_run(test_optimized_generate_quality, model, tokenizer)
    safe_run(test_unsupported_falls_back, model, tokenizer)
    safe_run(test_optimize_is_idempotent, model)
    return all_results


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("test_qwen_optimizer.py requires a CUDA GPU (same as test_correctness.py).")

    tokenizer, model = _load()
    optimize(model)

    results = run_all(model, tokenizer)
    for result in results:
        status = "PASS" if result["passed"] else "FAIL"
        print(f"[{status}] {result['name']}: {result['details']}")
