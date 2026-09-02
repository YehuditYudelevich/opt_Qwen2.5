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

    Dequantization happens in registers; INT8 bytes are the only weight
    traffic read from HBM (half the bytes of the FP16 kernel).
    """
    row = tl.program_id(0)

    offsets = tl.arange(0, BLOCK_K)
    mask = offsets < K

    x = tl.load(
        x_ptr + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    w_i8 = tl.load(
        qw_ptr + row * K + offsets,
        mask=mask,
        other=0,
    )
    w = w_i8.to(tl.float32)

    scale = tl.load(scale_ptr + row)

    acc = tl.sum(x * w, axis=0) * scale
    tl.store(y_ptr + row, acc.to(tl.float16))


def int8_gemv(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """
    x: [K] or tensor with K elements, fp16
    qweight: [N, K] int8, contiguous
    scale: [N] fp32
    returns: [N] fp16
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

    block_k = triton.next_power_of_2(k)

    int8_gemv_kernel[(n,)](
        x,
        qweight,
        scale,
        y,
        K=k,
        BLOCK_K=block_k,
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
    """
    row = tl.program_id(0)

    k = tl.arange(0, BLOCK_K)
    mask = k < K

    x = tl.load(
        x_ptr + k,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    gate_w = tl.load(
        gate_qw_ptr + row * K + k,
        mask=mask,
        other=0,
    ).to(tl.float32)

    up_w = tl.load(
        up_qw_ptr + row * K + k,
        mask=mask,
        other=0,
    ).to(tl.float32)

    gate_scale = tl.load(gate_scale_ptr + row)
    up_scale = tl.load(up_scale_ptr + row)

    gate = tl.sum(x * gate_w, axis=0) * gate_scale
    up = tl.sum(x * up_w, axis=0) * up_scale

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

    block_k = triton.next_power_of_2(k)

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
    )

    return out
