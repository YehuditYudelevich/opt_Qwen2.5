"""
Microbenchmark + parameter sweep for the INT8 Triton kernels
(int8_gemv_kernel, fused_int8_gate_up_kernel) against the real model's
weights and shapes.

    python microbench_kernels.py

What it does, for every shape this model actually issues these kernels
against (fused QKV, O, down_proj, lm_head, fused gate+up):

  1. Benchmarks the CURRENT kernel (kernels.py: tiled K-loop, fp16
     multiply / fp32 accumulate) against a frozen, unmodified copy of
     the PRE-optimization kernel (single next_pow2(K)-wide masked load,
     fp32 multiply) -- an honest before/after number on real hardware.
     The v1 copy lives only in this file and is not used anywhere else.
  2. Sweeps BLOCK_K x num_warps for the current kernel and reports the
     fastest combination per shape, so kernels.DEFAULT_BLOCK_K /
     DEFAULT_NUM_WARPS (chosen by inspection, not measurement) can be
     confirmed or corrected from real data.
  3. Checks correctness against the same dequantized-INT8 reference
     test_correctness.py uses for every config before trusting its
     timing -- a fast but wrong config is never reported as "best".

This is a KERNEL microbenchmark only. Per this project's own
methodology (CLAUDE.md / instructions.md), a microkernel win does not
by itself establish an end-to-end improvement -- run
run_t4_experiments.py (or benchmark.py) same-session, A/B/A, before
keeping any BLOCK_K/num_warps change that this script suggests.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from transformers import AutoModelForCausalLM

import kernels
from quant import QuantizedWeightINT8, dequantize_int8_per_row

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

SWEEP_BLOCK_K = [128, 256, 512, 1024]
SWEEP_NUM_WARPS = [2, 4, 8]


# ---------------------------------------------------------------------
# Frozen copy of the PRE-optimization kernels (single next_pow2(K)-wide
# masked load, fp32 multiply), kept ONLY so this script can report a
# real before/after number. Not used anywhere else in the project --
# kernels.py has already moved on to the tiled/fp16-multiply design.
# ---------------------------------------------------------------------
@triton.jit
def _int8_gemv_kernel_v1(x_ptr, qw_ptr, scale_ptr, y_ptr, K: tl.constexpr, BLOCK_K: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    mask = offsets < K
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(qw_ptr + row * K + offsets, mask=mask, other=0).to(tl.float32)
    scale = tl.load(scale_ptr + row)
    acc = tl.sum(x * w, axis=0) * scale
    tl.store(y_ptr + row, acc.to(tl.float16))


def _int8_gemv_v1(x, qweight, scale):
    x = x.reshape(-1)
    n, k = qweight.shape
    y = torch.empty(n, device=x.device, dtype=torch.float16)
    block_k = triton.next_power_of_2(k)
    _int8_gemv_kernel_v1[(n,)](x, qweight, scale, y, K=k, BLOCK_K=block_k)
    return y


@triton.jit
def _fused_int8_gate_up_kernel_v1(
    x_ptr, gate_qw_ptr, gate_scale_ptr, up_qw_ptr, up_scale_ptr, out_ptr,
    N: tl.constexpr, K: tl.constexpr, BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    k = tl.arange(0, BLOCK_K)
    mask = k < K
    x = tl.load(x_ptr + k, mask=mask, other=0.0).to(tl.float32)
    gate_w = tl.load(gate_qw_ptr + row * K + k, mask=mask, other=0).to(tl.float32)
    up_w = tl.load(up_qw_ptr + row * K + k, mask=mask, other=0).to(tl.float32)
    gate_scale = tl.load(gate_scale_ptr + row)
    up_scale = tl.load(up_scale_ptr + row)
    gate = tl.sum(x * gate_w, axis=0) * gate_scale
    up = tl.sum(x * up_w, axis=0) * up_scale
    silu_gate = gate * tl.sigmoid(gate)
    result = silu_gate * up
    tl.store(out_ptr + row, result.to(tl.float16))


def _fused_int8_gate_up_v1(x, gate_qweight, gate_scale, up_qweight, up_scale):
    x = x.reshape(-1)
    n, k = gate_qweight.shape
    out = torch.empty(n, device=x.device, dtype=torch.float16)
    block_k = triton.next_power_of_2(k)
    _fused_int8_gate_up_kernel_v1[(n,)](
        x, gate_qweight, gate_scale, up_qweight, up_scale, out, N=n, K=k, BLOCK_K=block_k
    )
    return out


def _bench_us(fn, *args, **kwargs):
    return triton.testing.do_bench(lambda: fn(*args, **kwargs)) * 1000


def _gemv_max_err(qweight, scale, x, y):
    ref = dequantize_int8_per_row(qweight, scale) @ x.to(torch.float32)
    return (y.to(torch.float32) - ref).abs().max().item()


def bench_gemv_shape(name, weight, device):
    torch.manual_seed(0)
    n, k = weight.shape
    x = torch.randn(k, device=device, dtype=torch.float16)
    qw = QuantizedWeightINT8(weight)

    print(f"\n--- int8_gemv: {name}  (K={k}, N={n}) ---")

    y_v1 = _int8_gemv_v1(x, qw.qweight, qw.scale)
    err_v1 = _gemv_max_err(qw.qweight, qw.scale, x, y_v1)
    us_v1 = _bench_us(_int8_gemv_v1, x, qw.qweight, qw.scale)
    print(f"  v1 (old, masked next_pow2={triton.next_power_of_2(k)}): {us_v1:8.2f} us  (max_err vs int8 ref: {err_v1:.4f})")

    best = None
    for block_k in SWEEP_BLOCK_K:
        for num_warps in SWEEP_NUM_WARPS:
            try:
                y = kernels.int8_gemv(x, qw.qweight, qw.scale, block_k=block_k, num_warps=num_warps)
                err = _gemv_max_err(qw.qweight, qw.scale, x, y)
                if err > 1.0:
                    print(f"  BLOCK_K={block_k:5d} num_warps={num_warps}: CORRECTNESS FAILED (err={err:.4f}), skipped")
                    continue
                us = _bench_us(kernels.int8_gemv, x, qw.qweight, qw.scale, block_k=block_k, num_warps=num_warps)
                print(f"  BLOCK_K={block_k:5d} num_warps={num_warps}: {us:8.2f} us  (max_err vs int8 ref: {err:.4f})")
                if best is None or us < best[0]:
                    best = (us, block_k, num_warps)
            except Exception as e:
                print(f"  BLOCK_K={block_k:5d} num_warps={num_warps}: FAILED ({type(e).__name__}: {e})")

    if best is not None:
        us, block_k, num_warps = best
        print(f"  BEST: BLOCK_K={block_k} num_warps={num_warps} -> {us:.2f} us ({us_v1 / us:.2f}x vs v1)")


def bench_fused_gate_up(gate_weight, up_weight, device):
    torch.manual_seed(0)
    n, k = gate_weight.shape
    x = torch.randn(k, device=device, dtype=torch.float16)
    gate_q = QuantizedWeightINT8(gate_weight)
    up_q = QuantizedWeightINT8(up_weight)

    gate_ref = F.linear(x, gate_weight)
    up_ref = F.linear(x, up_weight)
    reference = F.silu(gate_ref.to(torch.float32)) * up_ref.to(torch.float32)

    print(f"\n--- fused_int8_gate_up  (K={k}, N={n}) ---")

    y_v1 = _fused_int8_gate_up_v1(x, gate_q.qweight, gate_q.scale, up_q.qweight, up_q.scale)
    err_v1 = (y_v1.to(torch.float32) - reference).abs().max().item()
    us_v1 = _bench_us(_fused_int8_gate_up_v1, x, gate_q.qweight, gate_q.scale, up_q.qweight, up_q.scale)
    print(f"  v1 (old, masked next_pow2={triton.next_power_of_2(k)}): {us_v1:8.2f} us  (max_err vs fp16 ref: {err_v1:.4f})")

    best = None
    for block_k in SWEEP_BLOCK_K:
        for num_warps in SWEEP_NUM_WARPS:
            try:
                y = kernels.fused_int8_gate_up(
                    x, gate_q.qweight, gate_q.scale, up_q.qweight, up_q.scale,
                    block_k=block_k, num_warps=num_warps,
                )
                err = (y.to(torch.float32) - reference).abs().max().item()
                if err > 2.0:
                    print(f"  BLOCK_K={block_k:5d} num_warps={num_warps}: CORRECTNESS FAILED (err={err:.4f}), skipped")
                    continue
                us = _bench_us(
                    kernels.fused_int8_gate_up, x, gate_q.qweight, gate_q.scale, up_q.qweight, up_q.scale,
                    block_k=block_k, num_warps=num_warps,
                )
                print(f"  BLOCK_K={block_k:5d} num_warps={num_warps}: {us:8.2f} us  (max_err vs fp16 ref: {err:.4f})")
                if best is None or us < best[0]:
                    best = (us, block_k, num_warps)
            except Exception as e:
                print(f"  BLOCK_K={block_k:5d} num_warps={num_warps}: FAILED ({type(e).__name__}: {e})")

    if best is not None:
        us, block_k, num_warps = best
        print(f"  BEST: BLOCK_K={block_k} num_warps={num_warps} -> {us:.2f} us ({us_v1 / us:.2f}x vs v1)")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("microbench_kernels.py requires a CUDA GPU.")

    print("GPU:", torch.cuda.get_device_name(0))
    print("Loading model for real weight shapes/values...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="cuda")
    model.eval()
    device = next(model.parameters()).device

    layer = model.model.layers[0]
    q_w = layer.self_attn.q_proj.weight
    k_w = layer.self_attn.k_proj.weight
    v_w = layer.self_attn.v_proj.weight
    qkv_w = torch.cat([q_w, k_w, v_w], dim=0).contiguous()
    o_w = layer.self_attn.o_proj.weight
    down_w = layer.mlp.down_proj.weight
    gate_w = layer.mlp.gate_proj.weight
    up_w = layer.mlp.up_proj.weight
    lm_head_w = model.lm_head.weight

    shapes = {
        "qkv_fused": qkv_w,
        "o_proj": o_w,
        "down_proj": down_w,
        "lm_head": lm_head_w,
    }
    for name, weight in shapes.items():
        bench_gemv_shape(name, weight, device)

    bench_fused_gate_up(gate_w, up_w, device)

    print(
        "\nNOTE: kernel microbenchmark only. Per this project's own "
        "methodology, do not infer an end-to-end result from this -- "
        "run run_t4_experiments.py same-session, A/B/A, before keeping "
        "any config change this script suggests."
    )


if __name__ == "__main__":
    main()
