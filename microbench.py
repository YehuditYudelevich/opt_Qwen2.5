import torch
import torch.nn.functional as F
import triton
from transformers import AutoModelForCausalLM

from kernels import triton_gemv, fused_gate_up


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def main():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        device_map="cuda",
    )
    model.eval()

    layer = model.model.layers[0]
    gate_proj = layer.mlp.gate_proj
    up_proj = layer.mlp.up_proj
    down_proj = layer.mlp.down_proj

    x = torch.randn(
        1536,
        device="cuda",
        dtype=torch.float16,
    )

    # ---- GEMV correctness / benchmark ----
    y_ref = F.linear(x, gate_proj.weight)
    y_tri = triton_gemv(x, gate_proj.weight)

    diff = (y_ref - y_tri).abs()
    print("GEMV correctness:")
    print("  Max error:", diff.max().item())
    print("  Mean error:", diff.mean().item())
    print(
        "  Close:",
        torch.allclose(y_ref, y_tri, atol=1e-2, rtol=1e-2),
    )

    torch_us = triton.testing.do_bench(
        lambda: F.linear(x, gate_proj.weight)
    ) * 1000

    tri_us = triton.testing.do_bench(
        lambda: triton_gemv(x, gate_proj.weight)
    ) * 1000

    print("\nGate projection:")
    print(f"  PyTorch: {torch_us:.2f} us")
    print(f"  Triton:  {tri_us:.2f} us")
    print(f"  Speedup: {torch_us / tri_us:.2f}x")

    # ---- Fused gate+up+SiLU+mul ----
    def pytorch_gate_up():
        gate = F.linear(x, gate_proj.weight)
        up = F.linear(x, up_proj.weight)
        return F.silu(gate) * up

    fused_ref = pytorch_gate_up()
    fused_tri = fused_gate_up(
        x,
        gate_proj.weight,
        up_proj.weight,
    )

    diff = (fused_ref - fused_tri).abs()
    print("\nFused correctness:")
    print("  Max error:", diff.max().item())
    print("  Mean error:", diff.mean().item())
    print(
        "  Close:",
        torch.allclose(
            fused_ref,
            fused_tri,
            atol=2e-2,
            rtol=2e-2,
        ),
    )

    torch_fused_us = triton.testing.do_bench(
        pytorch_gate_up
    ) * 1000

    triton_fused_us = triton.testing.do_bench(
        lambda: fused_gate_up(
            x,
            gate_proj.weight,
            up_proj.weight,
        )
    ) * 1000

    print("\nGate + up + SiLU + multiply:")
    print(f"  PyTorch: {torch_fused_us:.2f} us")
    print(f"  Triton:  {triton_fused_us:.2f} us")
    print(f"  Speedup: {torch_fused_us / triton_fused_us:.2f}x")

    # ---- down_proj baseline ----
    x_down = torch.randn(
        8960,
        device="cuda",
        dtype=torch.float16,
    )

    down_ref = F.linear(x_down, down_proj.weight)
    down_tri = triton_gemv(x_down, down_proj.weight)

    diff = (down_ref - down_tri).abs()
    print("\nDown projection correctness:")
    print("  Max error:", diff.max().item())
    print("  Mean error:", diff.mean().item())
    print(
        "  Close:",
        torch.allclose(
            down_ref,
            down_tri,
            atol=1e-2,
            rtol=1e-2,
        ),
    )

    down_torch_us = triton.testing.do_bench(
        lambda: F.linear(x_down, down_proj.weight)
    ) * 1000

    down_tri_us = triton.testing.do_bench(
        lambda: triton_gemv(x_down, down_proj.weight)
    ) * 1000

    print("\nDown projection:")
    print(f"  PyTorch: {down_torch_us:.2f} us")
    print(f"  Triton:  {down_tri_us:.2f} us")
    print(f"  Speedup: {down_torch_us / down_tri_us:.2f}x")


if __name__ == "__main__":
    main()
