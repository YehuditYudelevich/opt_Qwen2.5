"""
INT8 weight-only quantized decode path for Qwen2 (Qwen2.5-1.5B-Instruct).

Design principle (see instructions.md "problem you must solve"): earlier
experiments replaced individual nn.Linear modules independently and the
speedup disappeared end-to-end because Python/launch overhead dominated.
This module instead:

  1. Quantizes EVERY decode-heavy matrix at once (q/k/v/o/gate/up/down/
     lm_head), not just gate/up, so the memory-traffic reduction is
     coarse-grained across the whole layer instead of a lone kernel win
     buried in an otherwise-FP16 stack.
  2. Fuses Q+K+V into a single kernel launch (they share the same input
     activation, so their INT8 weights are concatenated once at install
     time and read with ONE Triton GEMV call instead of three).
  3. Fuses gate+up+SiLU+mul into a single kernel launch (extends the
     existing FP16 fused kernel to INT8 weights).
  4. Reuses HF's own `apply_rotary_pos_emb` / `repeat_kv` helpers for the
     attention math that is NOT being optimized this round, so the only
     new/untested surface area is the projection replacement itself.

Every wrapper only takes the fast path for the exact decode shape
(batch=1, seq_len=1); prefill and any other shape fall back to the
original, unmodified HF submodule.
"""

import torch
import torch.nn as nn

from kernels import int8_gemv, fused_int8_gate_up
from quant import QuantizedWeightINT8, concat_quantized_rows

try:
    from transformers.models.qwen2.modeling_qwen2 import (
        apply_rotary_pos_emb,
        repeat_kv,
    )
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "Could not import Qwen2 internals (apply_rotary_pos_emb, repeat_kv) "
        "from transformers.models.qwen2.modeling_qwen2. The installed "
        "transformers version may have refactored the Qwen2 modeling file; "
        "pin a known-good version or update these imports."
    ) from e


class Int8QwenAttention(nn.Module):
    """
    Drop-in replacement for a Qwen2 layer's `self_attn`.

    Decode path (bsz=1, q_len=1):
        fused INT8 GEMV -> [Q | K | V] in one kernel launch
        + fused bias add
        RoPE, cache update, SDPA, INT8 o_proj  (same math as HF)

    Anything else (prefill, batch>1) delegates entirely to the original
    module.
    """

    def __init__(self, original_self_attn):
        super().__init__()
        self.original = original_self_attn
        self.use_int8 = False

        q_w = QuantizedWeightINT8(original_self_attn.q_proj.weight)
        k_w = QuantizedWeightINT8(original_self_attn.k_proj.weight)
        v_w = QuantizedWeightINT8(original_self_attn.v_proj.weight)

        fused_qkv, slices = concat_quantized_rows([q_w, k_w, v_w])
        self.qkv_qweight = fused_qkv.qweight
        self.qkv_scale = fused_qkv.scale
        (self.q_start, self.q_n), (self.k_start, self.k_n), (self.v_start, self.v_n) = slices

        q_bias = original_self_attn.q_proj.bias
        k_bias = original_self_attn.k_proj.bias
        v_bias = original_self_attn.v_proj.bias
        if q_bias is None or k_bias is None or v_bias is None:
            raise RuntimeError(
                "Expected q_proj/k_proj/v_proj to have bias=True for Qwen2 "
                "attention; got a module without bias."
            )
        self.qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=0).contiguous()

        self.o_weight = QuantizedWeightINT8(original_self_attn.o_proj.weight)
        if original_self_attn.o_proj.bias is not None:
            raise RuntimeError("Expected o_proj bias=False for Qwen2 attention.")

        self.hidden_size = original_self_attn.hidden_size
        self.num_heads = original_self_attn.num_heads
        self.head_dim = original_self_attn.head_dim
        self.num_key_value_heads = original_self_attn.num_key_value_heads
        self.num_key_value_groups = original_self_attn.num_key_value_groups

    def _fused_qkv(self, hidden_states_flat):
        fused_out = int8_gemv(hidden_states_flat, self.qkv_qweight, self.qkv_scale)
        fused_out = fused_out + self.qkv_bias

        q = fused_out[self.q_start : self.q_start + self.q_n]
        k = fused_out[self.k_start : self.k_start + self.k_n]
        v = fused_out[self.v_start : self.v_start + self.v_n]
        return q, k, v

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        bsz, q_len, hidden = hidden_states.shape

        eligible = (
            self.use_int8
            and bsz == 1
            and q_len == 1
            and hidden == self.hidden_size
            and not output_attentions
        )

        if not eligible:
            return self.original(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        q, k, v = self._fused_qkv(hidden_states.reshape(-1))

        query_states = q.view(1, 1, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = k.view(1, 1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = v.view(1, 1, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            cos, sin = self.original.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.original.layer_idx, cache_kwargs
            )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if causal_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=0.0,
            is_causal=False,
        )

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(-1)

        out = int8_gemv(attn_output, self.o_weight.qweight, self.o_weight.scale)
        out = out.view(1, 1, self.hidden_size)

        return out, None, past_key_value


class Int8QwenMLP(nn.Module):
    """
    Drop-in replacement for a Qwen2 layer's `mlp`.

    Decode path: fused INT8 gate+up+SiLU+mul in one kernel, then INT8
    down_proj in a second kernel. Prefill/other shapes fall back to the
    original module.
    """

    def __init__(self, original_mlp):
        super().__init__()
        self.original = original_mlp
        self.use_int8 = False
        self.hidden_size = original_mlp.hidden_size

        self.gate_weight = QuantizedWeightINT8(original_mlp.gate_proj.weight)
        self.up_weight = QuantizedWeightINT8(original_mlp.up_proj.weight)
        self.down_weight = QuantizedWeightINT8(original_mlp.down_proj.weight)

        for name, mod in (("gate_proj", original_mlp.gate_proj), ("up_proj", original_mlp.up_proj), ("down_proj", original_mlp.down_proj)):
            if mod.bias is not None:
                raise RuntimeError(f"Expected {name} bias=False for Qwen2 MLP.")

    def forward(self, x):
        eligible = (
            self.use_int8
            and x.ndim == 3
            and x.shape[0] == 1
            and x.shape[1] == 1
            and x.shape[2] == self.hidden_size
        )

        if not eligible:
            return self.original(x)

        hidden = fused_int8_gate_up(
            x.reshape(-1),
            self.gate_weight.qweight,
            self.gate_weight.scale,
            self.up_weight.qweight,
            self.up_weight.scale,
        )

        out = int8_gemv(hidden, self.down_weight.qweight, self.down_weight.scale)
        return out.view(1, 1, self.hidden_size)


class Int8LMHead(nn.Module):
    """
    Drop-in replacement for `model.lm_head`.

    Decode path (single-token logits, the only case that matters for
    autoregressive decode): INT8 GEMV over the full ~152k-row vocabulary
    matrix, halving the weight bytes read for the single most expensive
    GEMV in the model (~16% of CUDA time per prior profiling).
    """

    def __init__(self, original_lm_head):
        super().__init__()
        self.original = original_lm_head
        self.use_int8 = False
        self.hidden_size = original_lm_head.in_features

        if original_lm_head.bias is not None:
            raise RuntimeError("Expected lm_head bias=False for Qwen2.")

        self.weight_q = QuantizedWeightINT8(original_lm_head.weight)

    def forward(self, x):
        eligible = (
            self.use_int8
            and x.ndim == 3
            and x.shape[0] == 1
            and x.shape[1] == 1
            and x.shape[2] == self.hidden_size
        )

        if not eligible:
            return self.original(x)

        logits = int8_gemv(x.reshape(-1), self.weight_q.qweight, self.weight_q.scale)
        return logits.view(1, 1, -1)


def install_int8_quant(model):
    """
    Wrap every layer's self_attn/mlp and the model's lm_head with their
    INT8 counterparts. Safe to call only once on an unwrapped model.
    """
    for layer in model.model.layers:
        if not isinstance(layer.self_attn, Int8QwenAttention):
            layer.self_attn = Int8QwenAttention(layer.self_attn)
        if not isinstance(layer.mlp, Int8QwenMLP):
            layer.mlp = Int8QwenMLP(layer.mlp)

    if not isinstance(model.lm_head, Int8LMHead):
        model.lm_head = Int8LMHead(model.lm_head)


def set_int8(model, enabled: bool):
    for layer in model.model.layers:
        if not isinstance(layer.self_attn, Int8QwenAttention):
            raise RuntimeError("Int8QwenAttention is not installed. Call install_int8_quant(model) first.")
        if not isinstance(layer.mlp, Int8QwenMLP):
            raise RuntimeError("Int8QwenMLP is not installed. Call install_int8_quant(model) first.")
        layer.self_attn.use_int8 = enabled
        layer.mlp.use_int8 = enabled

    if not isinstance(model.lm_head, Int8LMHead):
        raise RuntimeError("Int8LMHead is not installed. Call install_int8_quant(model) first.")
    model.lm_head.use_int8 = enabled


def quantization_report(model):
    """
    Returns a summary dict of FP16-equivalent vs INT8 weight bytes across
    the installed quantized modules, for sanity-checking the expected
    memory-traffic reduction before running any benchmark.
    """
    fp16_bytes = 0
    int8_bytes = 0

    for layer in model.model.layers:
        attn = layer.self_attn
        fp16_bytes += QuantizedWeightINT8.fp16_memory_bytes((attn.qkv_qweight.shape[0], attn.qkv_qweight.shape[1]))
        int8_bytes += attn.qkv_qweight.numel() + attn.qkv_scale.numel() * 4

        fp16_bytes += QuantizedWeightINT8.fp16_memory_bytes(attn.o_weight.shape)
        int8_bytes += attn.o_weight.memory_bytes()

        mlp = layer.mlp
        for w in (mlp.gate_weight, mlp.up_weight, mlp.down_weight):
            fp16_bytes += QuantizedWeightINT8.fp16_memory_bytes(w.shape)
            int8_bytes += w.memory_bytes()

    lm_head = model.lm_head
    fp16_bytes += QuantizedWeightINT8.fp16_memory_bytes(lm_head.weight_q.shape)
    int8_bytes += lm_head.weight_q.memory_bytes()

    return {
        "fp16_bytes": fp16_bytes,
        "int8_bytes": int8_bytes,
        "fp16_gb": fp16_bytes / 1e9,
        "int8_gb": int8_bytes / 1e9,
        "ratio": fp16_bytes / int8_bytes,
    }
