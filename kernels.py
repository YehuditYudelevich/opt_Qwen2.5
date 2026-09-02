import torch
import triton
import triton.language as tl


@triton.jit
def gemv_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Specialized FP16 GEMV for decode-time Linear:
      x: [K]
      W: [N, K] contiguous
      y: [N]

    One Triton program computes one output row.
    """
    row = tl.program_id(0)

    offsets = tl.arange(0, BLOCK_K)
    mask = offsets < K

    x = tl.load(
        x_ptr + offsets,
        mask=mask,
        other=0.0,
    )

    w = tl.load(
        w_ptr + row * K + offsets,
        mask=mask,
        other=0.0,
    )

    result = tl.sum(x * w, axis=0)
    tl.store(y_ptr + row, result)


def triton_gemv(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    x: [K] or any tensor with exactly K elements
    weight: [N, K], contiguous
    returns: [N]
    """
    x = x.reshape(-1)

    if weight.ndim != 2:
        raise ValueError(f"weight must be rank-2, got {weight.shape}")

    n, k = weight.shape
    if x.numel() != k:
        raise ValueError(f"x has {x.numel()} elements but K={k}")

    y = torch.empty(
        n,
        device=x.device,
        dtype=x.dtype,
    )

    block_k = triton.next_power_of_2(k)

    gemv_kernel[(n,)](
        x,
        weight,
        y,
        K=k,
        BLOCK_K=block_k,
    )

    return y


@triton.jit
def fused_gate_up_kernel(
    x_ptr,
    gate_w_ptr,
    up_w_ptr,
    out_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fuses:
      gate = x @ W_gate.T
      up   = x @ W_up.T
      out  = silu(gate) * up

    Intended for Qwen decode at batch=1, seq_len=1.
    """
    row = tl.program_id(0)

    k = tl.arange(0, BLOCK_K)
    mask = k < K

    x = tl.load(
        x_ptr + k,
        mask=mask,
        other=0.0,
    )

    gate_w = tl.load(
        gate_w_ptr + row * K + k,
        mask=mask,
        other=0.0,
    )

    up_w = tl.load(
        up_w_ptr + row * K + k,
        mask=mask,
        other=0.0,
    )

    gate = tl.sum(x * gate_w, axis=0)
    up = tl.sum(x * up_w, axis=0)

    gate_fp32 = gate.to(tl.float32)
    up_fp32 = up.to(tl.float32)

    silu_gate = gate_fp32 * tl.sigmoid(gate_fp32)
    result = silu_gate * up_fp32

    tl.store(
        out_ptr + row,
        result.to(tl.float16),
    )


def fused_gate_up(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
) -> torch.Tensor:
    """
    x: [K] or tensor with K elements
    gate_weight/up_weight: [N, K]
    returns: [N]
    """
    x = x.reshape(-1)

    if gate_weight.shape != up_weight.shape:
        raise ValueError(
            f"gate/up weight shape mismatch: "
            f"{gate_weight.shape} vs {up_weight.shape}"
        )

    n, k = gate_weight.shape
    if x.numel() != k:
        raise ValueError(f"x has {x.numel()} elements but K={k}")

    out = torch.empty(
        n,
        device=x.device,
        dtype=x.dtype,
    )

    block_k = triton.next_power_of_2(k)

    fused_gate_up_kernel[(n,)](
        x,
        gate_weight,
        up_weight,
        out,
        N=n,
        K=k,
        BLOCK_K=block_k,
    )

    return out


# Default tile width for the K-loop in int8_gemv_kernel /
# fused_int8_gate_up_kernel. Chosen because it evenly divides BOTH K
# values this model actually uses (hidden=1536=256*6, intermediate=
# 8960=256*35), which means the compiled kernel emits zero masked/
# padded lanes for this model -- the previous BLOCK_K=next_pow2(K)
# design wasted 25% of every QKV/O/lm_head GEMV (2048 vs 1536) and
# 45.3% of every down_proj GEMV (16384 vs 8960) on provably-zero
# padding, plus held that whole padded width as live registers per
# program. This value was chosen by inspection (largest power-of-two
# divisor common to 1536 and 8960), not by on-GPU measurement -- see
# microbench_kernels.py to sweep it and confirm/override on real
# hardware.
DEFAULT_BLOCK_K = 256
DEFAULT_NUM_WARPS = 4


@triton.jit
def int8_gemv_kernel(
    x_ptr,
    qw_ptr,
    scale_ptr,
    y_ptr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Weight-only INT8 GEMV for decode-time Linear:
      x: [K] fp16
      qW: [N, K] int8, per-row symmetric quantized
      scale: [N] fp32, scale[n] = amax(|W[n,:]|) / 127
      y: [N] fp16

    K is walked in BLOCK_K-sized tiles instead of one big
    BLOCK_K=next_pow2(K)-wide masked load. K and BLOCK_K are
    tl.constexpr, so `num_full_tiles`/`remainder` are resolved at
    trace time: when BLOCK_K evenly divides K (true for this model's
    actual K values with the default BLOCK_K) the remainder branch
    generates no code at all, i.e. no masked loads anywhere. Arbitrary
    K still works correctly via the masked remainder tile.

    The multiply is done with the weight dequantized to fp16 (not
    fp32), accumulating the per-tile products in fp32 -- this matches
    the "INT8 v2" kernel already validated in this project's history
    (instructions.md: fp16-multiply/fp32-accumulate measured 75.2us ->
    71.1us over an all-fp32 version).
    """
    row = tl.program_id(0)
    offs_base = tl.arange(0, BLOCK_K)

    num_full_tiles = K // BLOCK_K
    remainder = K - num_full_tiles * BLOCK_K

    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)

    for tile in range(num_full_tiles):
        offs = tile * BLOCK_K + offs_base
        x = tl.load(x_ptr + offs)
        w = tl.load(qw_ptr + row * K + offs).to(tl.float16)
        acc += (x * w).to(tl.float32)

    if remainder > 0:
        offs = num_full_tiles * BLOCK_K + offs_base
        mask = offs_base < remainder
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        w = tl.load(qw_ptr + row * K + offs, mask=mask, other=0).to(tl.float16)
        acc += (x * w).to(tl.float32)

    scale = tl.load(scale_ptr + row)
    result = tl.sum(acc, axis=0) * scale
    tl.store(y_ptr + row, result.to(tl.float16))


def int8_gemv(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scale: torch.Tensor,
    block_k: int = DEFAULT_BLOCK_K,
    num_warps: int = DEFAULT_NUM_WARPS,
) -> torch.Tensor:
    """
    x: [K] or tensor with K elements, fp16
    qweight: [N, K] int8, contiguous
    scale: [N] fp32
    returns: [N] fp16

    block_k must be a power of 2 (tl.arange requirement); it need not
    divide K evenly (see int8_gemv_kernel), but for this model's shapes
    the default does, which is the point. block_k/num_warps are exposed
    for offline tuning via microbench_kernels.py.
    """
    x = x.reshape(-1)

    if qweight.ndim != 2:
        raise ValueError(f"qweight must be rank-2, got {qweight.shape}")
    if qweight.dtype != torch.int8:
        raise ValueError(f"qweight must be int8, got {qweight.dtype}")

    n, k = qweight.shape
    if x.numel() != k:
        raise ValueError(f"x has {x.numel()} elements but K={k}")
    if scale.numel() != n:
        raise ValueError(f"scale has {scale.numel()} elements but N={n}")

    y = torch.empty(
        n,
        device=x.device,
        dtype=torch.float16,
    )

    int8_gemv_kernel[(n,)](
        x,
        qweight,
        scale,
        y,
        K=k,
        BLOCK_K=block_k,
        num_warps=num_warps,
    )

    return y


@triton.jit
def fused_int8_gate_up_kernel(
    x_ptr,
    gate_qw_ptr,
    gate_scale_ptr,
    up_qw_ptr,
    up_scale_ptr,
    out_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fuses, with INT8 weight-only quantized gate/up matrices:
      gate = dequant(qW_gate) @ x
      up   = dequant(qW_up)   @ x
      out  = silu(gate) * up

    x is read from HBM once and reused for both projections; both weight
    matrices are read as INT8 (half the bytes of the FP16 fused kernel).

    Same K-tiling rework as int8_gemv_kernel (see its docstring): walks
    K in BLOCK_K-sized tiles instead of one BLOCK_K=next_pow2(K)-wide
    masked load, and multiplies in fp16 with fp32 accumulation. For
    this model's K=1536, the default BLOCK_K=256 divides evenly (6
    tiles, no masking) versus the old BLOCK_K=2048 which wasted 25% of
    every call on masked-zero lanes -- doubled here since gate and up
    are both affected per call.
    """
    row = tl.program_id(0)
    offs_base = tl.arange(0, BLOCK_K)

    num_full_tiles = K // BLOCK_K
    remainder = K - num_full_tiles * BLOCK_K

    gate_acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
    up_acc = tl.zeros((BLOCK_K,), dtype=tl.float32)

    for tile in range(num_full_tiles):
        offs = tile * BLOCK_K + offs_base
        x = tl.load(x_ptr + offs)
        gate_w = tl.load(gate_qw_ptr + row * K + offs).to(tl.float16)
        up_w = tl.load(up_qw_ptr + row * K + offs).to(tl.float16)
        gate_acc += (x * gate_w).to(tl.float32)
        up_acc += (x * up_w).to(tl.float32)

    if remainder > 0:
        offs = num_full_tiles * BLOCK_K + offs_base
        mask = offs_base < remainder
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        gate_w = tl.load(gate_qw_ptr + row * K + offs, mask=mask, other=0).to(tl.float16)
        up_w = tl.load(up_qw_ptr + row * K + offs, mask=mask, other=0).to(tl.float16)
        gate_acc += (x * gate_w).to(tl.float32)
        up_acc += (x * up_w).to(tl.float32)

    gate_scale = tl.load(gate_scale_ptr + row)
    up_scale = tl.load(up_scale_ptr + row)

    gate = tl.sum(gate_acc, axis=0) * gate_scale
    up = tl.sum(up_acc, axis=0) * up_scale

    silu_gate = gate * tl.sigmoid(gate)
    result = silu_gate * up

    tl.store(
        out_ptr + row,
        result.to(tl.float16),
    )


def fused_int8_gate_up(
    x: torch.Tensor,
    gate_qweight: torch.Tensor,
    gate_scale: torch.Tensor,
    up_qweight: torch.Tensor,
    up_scale: torch.Tensor,
    block_k: int = DEFAULT_BLOCK_K,
    num_warps: int = DEFAULT_NUM_WARPS,
) -> torch.Tensor:
    """
    x: [K] or tensor with K elements, fp16
    gate_qweight/up_qweight: [N, K] int8
    gate_scale/up_scale: [N] fp32
    returns: [N] fp16
    """
    x = x.reshape(-1)

    if gate_qweight.shape != up_qweight.shape:
        raise ValueError(
            f"gate/up qweight shape mismatch: "
            f"{gate_qweight.shape} vs {up_qweight.shape}"
        )

    n, k = gate_qweight.shape
    if x.numel() != k:
        raise ValueError(f"x has {x.numel()} elements but K={k}")

    out = torch.empty(
        n,
        device=x.device,
        dtype=torch.float16,
    )

    fused_int8_gate_up_kernel[(n,)](
        x,
        gate_qweight,
        gate_scale,
        up_qweight,
        up_scale,
        out,
        N=n,
        K=k,
        BLOCK_K=block_k,
        num_warps=num_warps,
    )

    return out
