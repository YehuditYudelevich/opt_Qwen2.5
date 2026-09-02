"""
Diagnostic: locate exactly where Inf/NaN first appears in the INT8
decode path, module by module, during a real forward pass -- instead of
only seeing a downstream symptom like `perplexity_int8 = NaN` in
test_correctness.test_int8_quality.

    python debug_int8_nan.py

This is what actually found/confirmed the fp16-multiply overflow bug in
kernels.py (see RESULTS.md "Batch 2 regression"): `int8_gemv_kernel`/
`fused_int8_gate_up_kernel` multiplied an activation by a RAW (unscaled,
up to +-127) int8 weight byte in fp16 before applying the dequant scale
-- any activation channel with |x| > ~515 overflows to inf the instant
it hits a near-+-127 weight byte, in fp16 (max ~65504) but not fp32
(~3.4e38). The two decode-path activations that are NOT freshly
RMSNorm'd -- attention output before o_proj, and SiLU(gate)*up before
down_proj -- are exactly where that's most likely to happen, since
RMSNorm bounds the other two GEMV inputs (QKV's input, lm_head's input).

Two independent checks:
  1. Forward hooks on every layer's self_attn/mlp/lm_head, reporting the
     first module whose INT8-path *output* contains inf/nan.
  2. optimized_model.DEBUG_ACTIVATION_STATS instrumentation, reporting
     the max-abs *input* to o_proj/down_proj per layer -- useful even
     when nothing is currently broken, since it's exactly the quantity
     that would need to exceed ~515 to risk the kind of overflow this
     found in a from-scratch fp16-multiply kernel.
"""

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

import optimized_model
from decode_loop import greedy_decode_dynamic

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
OVERFLOW_THRESHOLD = 515.0  # fp16 max (~65504) / 127 (max int8 weight byte)


def _load():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="cuda"
    )
    model.eval()
    return tokenizer, model


def _install_output_hooks(model, first_hit):
    handles = []

    def make_hook(name):
        def hook(module, inputs, output):
            if first_hit["found"]:
                return
            tensors = output if isinstance(output, tuple) else (output,)
            for t in tensors:
                if not torch.is_tensor(t):
                    continue
                if torch.isinf(t).any() or torch.isnan(t).any():
                    first_hit["found"] = True
                    first_hit["name"] = name
                    return

        return hook

    for i, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.register_forward_hook(make_hook(f"layer{i}.self_attn")))
        handles.append(layer.mlp.register_forward_hook(make_hook(f"layer{i}.mlp")))
    handles.append(model.lm_head.register_forward_hook(make_hook("lm_head")))

    return handles


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("debug_int8_nan.py requires a CUDA GPU.")

    tokenizer, model = _load()
    optimized_model.install_int8_quant(model)
    optimized_model.set_int8(model, True)
    optimized_model.DEBUG_ACTIVATION_STATS = True

    messages = [{"role": "user", "content": "Explain how a CPU works in a few sentences."}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    first_hit = {"found": False, "name": None}
    handles = _install_output_hooks(model, first_hit)

    try:
        greedy_decode_dynamic(model, inputs["input_ids"], max_new_tokens=48)
    finally:
        for h in handles:
            h.remove()
        optimized_model.set_int8(model, False)
        optimized_model.DEBUG_ACTIVATION_STATS = False

    print("=" * 70)
    if first_hit["found"]:
        print(f"FIRST Inf/NaN detected at module output: {first_hit['name']}")
    else:
        print("No Inf/NaN detected in any layer's self_attn/mlp/lm_head output.")

    print("\nMax |activation| into o_proj / down_proj, per layer:")
    print(f"{'layer':<10}{'o_proj input':>16}{'down_proj input':>18}")
    worst_o, worst_down = 0.0, 0.0
    for i, layer in enumerate(model.model.layers):
        o_val = getattr(layer.self_attn, "last_o_proj_input_max_abs", float("nan"))
        d_val = getattr(layer.mlp, "last_down_proj_input_max_abs", float("nan"))
        worst_o = max(worst_o, o_val)
        worst_down = max(worst_down, d_val)
        flag = " <-- near/over overflow threshold" if max(o_val, d_val) > OVERFLOW_THRESHOLD * 0.5 else ""
        print(f"{i:<10}{o_val:16.2f}{d_val:18.2f}{flag}")

    print(f"\nOverflow threshold for a RAW-int8-byte * fp16-activation multiply: ~{OVERFLOW_THRESHOLD:.0f}")
    print(f"Worst o_proj input seen:    {worst_o:.2f}")
    print(f"Worst down_proj input seen: {worst_down:.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
