"""
Weight-only INT8 quantization utilities.

Scheme: per-output-row (per-channel) symmetric INT8.

    scale[n]   = max(|W[n, :]|) / 127
    qW[n, k]   = round(W[n, k] / scale[n]).clamp(-127, 127)   (int8)
    deq[n, k]  = qW[n, k] * scale[n]

Dequantization happens inside the Triton GEMV kernel (kernels.py), never
materialized as a full FP16 tensor on the optimized path, so weight bytes
actually moved through HBM are halved relative to FP16.

All functions here are pure PyTorch tensor math with no CUDA/Triton
dependency, so quantization correctness can be checked on CPU.
"""

import torch


def quantize_int8_per_row(weight: torch.Tensor):
    """
    weight: [N, K] any float dtype, any device.

    Returns:
        qweight: [N, K] int8
        scale:   [N]     float32
    """
    if weight.ndim != 2:
        raise ValueError(f"weight must be rank-2, got {weight.shape}")

    w = weight.detach().to(torch.float32)

    amax = w.abs().amax(dim=1)
    amax = amax.clamp(min=1e-8)
    scale = amax / 127.0

    qweight = torch.round(w / scale[:, None]).clamp(-127, 127).to(torch.int8)

    return qweight.contiguous(), scale.contiguous()


def dequantize_int8_per_row(qweight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Reference dequantization (used only for correctness tests, never on the
    hot path).
    """
    return qweight.to(torch.float32) * scale[:, None]


def quantization_error(weight: torch.Tensor, qweight: torch.Tensor, scale: torch.Tensor):
    """
    Returns (max_abs_error, mean_abs_error, max_relative_error) between the
    original weight and its dequantized INT8 approximation.
    """
    w = weight.detach().to(torch.float32)
    deq = dequantize_int8_per_row(qweight, scale)
    diff = (w - deq).abs()

    max_err = diff.max().item()
    mean_err = diff.mean().item()

    denom = w.abs().clamp(min=1e-6)
    max_rel_err = (diff / denom).max().item()

    return max_err, mean_err, max_rel_err


class QuantizedWeightINT8:
    """
    Container for a per-row-quantized weight, holding the int8 payload and
    fp32 scales on the same device as the source weight.

    Not an nn.Module by design: it should be created once at model-patch
    time and referenced from lightweight wrapper modules
    (see optimized_model.py), not recreated per forward call.
    """

    __slots__ = ("qweight", "scale", "shape", "orig_dtype")

    def __init__(self, weight: torch.Tensor):
        self.shape = tuple(weight.shape)
        self.orig_dtype = weight.dtype
        qweight, scale = quantize_int8_per_row(weight)
        self.qweight = qweight.to(weight.device)
        self.scale = scale.to(weight.device)

    @property
    def n(self):
        return self.shape[0]

    @property
    def k(self):
        return self.shape[1]

    def memory_bytes(self) -> int:
        return self.qweight.numel() + self.scale.numel() * 4

    @staticmethod
    def fp16_memory_bytes(weight_shape) -> int:
        n, k = weight_shape
        return n * k * 2


def concat_quantized_rows(quantized_list):
    """
    Concatenate several QuantizedWeightINT8 objects along the row (N)
    dimension into a single logical weight, so multiple independent
    projections sharing the same input (e.g. Q/K/V) can be computed with
    ONE kernel launch instead of one launch each.

    All inputs must share K and device/dtype lineage.

    Returns a new QuantizedWeightINT8-like object (constructed directly
    from pre-quantized parts, no re-quantization) plus a list of
    (start_row, num_rows) slices in the same order as the input list, so
    callers can slice the fused output back into individual tensors.
    """
    ks = {q.k for q in quantized_list}
    if len(ks) != 1:
        raise ValueError(f"K mismatch across concatenated weights: {ks}")

    fused = object.__new__(QuantizedWeightINT8)
    fused.qweight = torch.cat([q.qweight for q in quantized_list], dim=0).contiguous()
    fused.scale = torch.cat([q.scale for q in quantized_list], dim=0).contiguous()
    fused.shape = (fused.qweight.shape[0], fused.qweight.shape[1])
    fused.orig_dtype = quantized_list[0].orig_dtype

    slices = []
    start = 0
    for q in quantized_list:
        slices.append((start, q.n))
        start += q.n

    return fused, slices
