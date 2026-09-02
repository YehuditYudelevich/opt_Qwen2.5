import argparse
import statistics
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model_patch import install_fused_mlp, set_fused_mlp


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        device_map="cuda",
    )
    model.eval()
    return tokenizer, model


def make_inputs(tokenizer):
    messages = [
        {
            "role": "user",
            "content": "Explain how a CPU works in a few sentences.",
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    return tokenizer(
        text,
        return_tensors="pt",
    ).to("cuda")


def benchmark_model(
    model,
    inputs,
    label,
    runs=5,
    new_tokens=128,
):
    vals = []

    with torch.inference_mode():
        for _ in range(2):
            _ = model.generate(
                **inputs,
                max_new_tokens=32,
                min_new_tokens=32,
                do_sample=False,
                use_cache=True,
            )

    torch.cuda.synchronize()

    for i in range(runs):
        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=new_tokens,
                min_new_tokens=new_tokens,
                do_sample=False,
                use_cache=True,
            )

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        generated = output.shape[1] - inputs["input_ids"].shape[1]
        tok_s = generated / elapsed
        vals.append(tok_s)

        print(
            f"{label} run {i + 1}: "
            f"{tok_s:.2f} tok/s | "
            f"{1000.0 / tok_s:.2f} ms/token"
        )

    median = statistics.median(vals)
    mean = statistics.mean(vals)

    print(f"{label} mean:   {mean:.2f} tok/s")
    print(f"{label} median: {median:.2f} tok/s\n")

    return median


def run_aba(model, inputs, runs=5):
    set_fused_mlp(model, False)
    a = benchmark_model(model, inputs, "PyTorch A", runs=runs)

    set_fused_mlp(model, True)
    b = benchmark_model(model, inputs, "Fused MLP", runs=runs)

    set_fused_mlp(model, False)
    c = benchmark_model(model, inputs, "PyTorch B", runs=runs)

    baseline = (a + c) / 2.0

    print("===== RESULTS =====")
    print(f"PyTorch A:      {a:.2f} tok/s")
    print(f"Fused MLP:      {b:.2f} tok/s")
    print(f"PyTorch B:      {c:.2f} tok/s")
    print(f"Baseline avg:   {baseline:.2f} tok/s")
    print(f"Speedup:        {b / baseline:.3f}x")
    print(f"Improvement:    {(b / baseline - 1) * 100:.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required.")

    print("GPU:", torch.cuda.get_device_name(0))
    print("Compute capability:", torch.cuda.get_device_capability(0))
    print("PyTorch:", torch.__version__)

    tokenizer, model = load_model()
    inputs = make_inputs(tokenizer)

    install_fused_mlp(model)
    run_aba(model, inputs, runs=args.runs)


if __name__ == "__main__":
    main()
