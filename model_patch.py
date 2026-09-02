import torch
import torch.nn as nn
import torch.nn.functional as F

from kernels import triton_gemv, fused_gate_up


class SwitchableLinear1536to8960(nn.Module):
    """
    Experimental wrapper used during the first integration test.

    Kept for reproducibility of the experiment where gate_proj and up_proj
    were replaced independently. That integration was slower end-to-end.
    """
    def __init__(self, old_module):
        super().__init__()
        self.weight = old_module.weight
        self.use_triton = False

    def forward(self, x):
        if self.use_triton and x.numel() == 1536:
            y = triton_gemv(
                x.reshape(-1),
                self.weight,
            )
            return y.reshape(*x.shape[:-1], self.weight.shape[0])

        return F.linear(x, self.weight)


class FusedQwenMLP(nn.Module):
    """
    Replaces Qwen MLP's:
      gate_proj + up_proj + SiLU + multiply
    with one fused Triton kernel during decode.

    down_proj intentionally remains PyTorch for now.

    Prefill and unsupported shapes fall back to the original implementation.
    """
    def __init__(self, original_mlp):
        super().__init__()
        self.original_mlp = original_mlp
        self.use_fused = False

    def forward(self, x):
        if (
            self.use_fused
            and x.ndim == 3
            and x.shape[0] == 1
            and x.shape[1] == 1
            and x.shape[2] == 1536
        ):
            hidden = fused_gate_up(
                x.reshape(-1),
                self.original_mlp.gate_proj.weight,
                self.original_mlp.up_proj.weight,
            )

            hidden = hidden.reshape(1, 1, -1)
            return self.original_mlp.down_proj(hidden)

        return self.original_mlp(x)


def install_fused_mlp(model):
    """
    Wrap each Qwen layer MLP once.
    Safe to call only on an unwrapped model.
    """
    for layer in model.model.layers:
        if isinstance(layer.mlp, FusedQwenMLP):
            continue
        layer.mlp = FusedQwenMLP(layer.mlp)


def set_fused_mlp(model, enabled: bool):
    for layer in model.model.layers:
        if not isinstance(layer.mlp, FusedQwenMLP):
            raise RuntimeError(
                "FusedQwenMLP is not installed. Call install_fused_mlp(model) first."
            )
        layer.mlp.use_fused = enabled
