import argparse

import torch
from torch.profiler import profile, ProfilerActivity
from transformers import AutoModelForCausalLM, AutoTokenizer

from model_patch import install_fused_mlp, set_fused_mlp


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fused",
        action="store_true",
        help="Profile the fused MLP path instead of PyTorch baseline.",
    )
    parser.add_argument("--tokens", type=int, default=32)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        device_map="cuda",
    )
    model.eval()

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

    inputs = tokenizer(
        text,
        return_tensors="pt",
    ).to("cuda")

    install_fused_mlp(model)
    set_fused_mlp(model, args.fused)

    with torch.inference_mode():
        _ = model.generate(
            **inputs,
            max_new_tokens=16,
            min_new_tokens=16,
            do_sample=False,
            use_cache=True,
        )

    torch.cuda.synchronize()

    with profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        with torch.inference_mode():
            _ = model.generate(
                **inputs,
                max_new_tokens=args.tokens,
                min_new_tokens=args.tokens,
                do_sample=False,
                use_cache=True,
            )

    torch.cuda.synchronize()

    print("\n===== SORTED BY CUDA TIME =====")
    print(
        prof.key_averages().table(
            sort_by="cuda_time_total",
            row_limit=40,
        )
    )

    print("\n===== SORTED BY CPU TIME =====")
    print(
        prof.key_averages().table(
            sort_by="cpu_time_total",
            row_limit=40,
        )
    )


if __name__ == "__main__":
    main()
