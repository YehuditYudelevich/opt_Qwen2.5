"""
Apples-to-apples benchmark: our optimized Qwen2.5-1.5B decode path
(qwen_optimizer.optimize(), unmodified) vs vLLM, same model, same GPU,
same prompt, batch size 1, 96 generated tokens.

Run as TWO SEPARATE PROCESSES (recommended: two separate Colab
runtimes, see "T4/vLLM gotchas" below) so neither engine's dependencies
or GPU memory footprint can interfere with the other's measurement:

    python bench_vs_vllm.py --engine ours
    python bench_vs_vllm.py --engine vllm

Each saves a JSON to results/bench_vs_vllm_<engine>.json. Once both
exist (copy them into the same place if run in separate runtimes):

    python bench_vs_vllm.py --engine compare

prints the side-by-side table. This mode does not touch the GPU or
import either engine's dependencies -- it only reads the two JSON files.

--------------------------------------------------------------------
Methodology (identical for both engines, read before trusting numbers)
--------------------------------------------------------------------

TTFT (time to first token) is measured with a SEPARATE generation call
capped at exactly 1 output token. For our engine, this is exact: with
max_new_tokens=1, decode_loop.greedy_decode_static's decode loop runs
zero iterations, so the timed call is pure prefill. For vLLM, a
max_tokens=1 request is the standard way to isolate prefill+first-token
latency in non-streaming benchmarks. Both are timed after identical
warmup so neither measurement accidentally includes one-time
compilation (ours) or CUDA graph capture (vLLM) overhead.

Time per output token (TPOT) is NOT measured directly (neither engine's
simple offline API gives per-token timestamps without deeper
instrumentation than is worth adding here). It is derived:

    TPOT = (total_time_for_96_tokens - TTFT) / (96 - 1)

using the SAME formula for both engines, from the SAME two calls. This
is a standard approximation (used by many public LLM-serving
benchmarks) but it IS an approximation, not a direct per-token trace:
it assumes prefill cost is ~constant between the two calls (true here,
same prompt, same cache setup) and folds any per-call fixed overhead
into TTFT rather than TPOT.

tokens/sec is reported two ways: end-to-end (96 / total_time, includes
prefill) and decode-only / steady-state (95 / (total_time - TTFT), the
reciprocal of TPOT).

VRAM is read via `nvidia-smi --query-gpu=memory.used` (system-wide,
works identically for any engine/process) at four points: idle
(process start, before loading anything), loaded (model/engine
constructed), post-warmup (after a couple of throwaway generations --
this is where both engines' memory pools actually materialize: vLLM
pre-allocates its KV cache block pool up front based on
--vllm-gpu-memory-utilization, and our StaticCache/torch.compile
CUDA-graph pools are built lazily on first call), and post-run. No
background polling thread is used: for a single-request, 96-token,
batch=1 workload, both engines' peak memory is reached at pool
allocation time, not during the decode loop itself (KV writes go into
already-allocated buffers), so a polling thread would add sampling
jitter/CPU contention for no real precision gain here.

vLLM runs the plain, unquantized FP16 checkpoint through its own
standard (CUDA-graph-capturing) serving path -- i.e. vLLM's own
out-of-the-box best, not handicapped. Ours runs the INT8 weight-only +
fused-kernel + StaticCache + torch.compile path via the packaged
`qwen_optimizer.optimize()` API, also unmodified. This is intentionally
an "our best vs vLLM's best-out-of-the-box default" comparison, NOT a
controlled-for-quantization one -- if you want INT8-vs-INT8, vLLM would
need an AWQ/GPTQ-quantized checkpoint of this model, which is a
separate, larger effort not attempted here.

--------------------------------------------------------------------
T4 / vLLM gotchas (read before running --engine vllm)
--------------------------------------------------------------------

- vLLM pins its own torch (and sometimes triton) version per release.
  Installing it into the same environment as this project's
  transformers==4.46.3 / triton stack risks pip silently upgrading or
  downgrading packages this project depends on. RECOMMENDED: run
  `--engine vllm` in a fresh Colab runtime with only `pip install vllm`
  (plus whatever it pulls in), and run `--engine ours` in this
  project's normal environment. Bring the two JSON files together
  afterward for `--engine compare`.
- Some recent vLLM versions default to a FlashAttention-2 backend that
  requires compute capability >= 8.0 (Ampere+). Tesla T4 is compute
  capability 7.5 (Turing) and does NOT support it. vLLM is generally
  supposed to auto-detect and fall back, but if `--engine vllm` errors
  out mentioning FlashAttention or compute capability, try:
      VLLM_ATTENTION_BACKEND=XFORMERS
  (or consult that vLLM version's docs for the Turing-compatible
  backend name) and/or a slightly older vLLM release known to support
  Turing well.
- `--vllm-gpu-memory-utilization` defaults to 0.5 here (NOT vLLM's own
  default of 0.9) specifically so the VRAM comparison isn't trivially
  dominated by "vLLM pre-allocated most of the GPU because it was told
  it could" rather than anything about this workload. This is a real,
  documented vLLM serving knob, not a handicap -- feel free to rerun
  with a different value, but report which value was used.
"""

import argparse
import json
import os
import statistics
import subprocess
import time

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_PROMPT_CONTENT = "Explain how a CPU works in a few sentences."
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def render_prompt(content):
    """
    Renders the chat prompt via the model's own tokenizer ONCE, so both
    engines are handed the exact same text string -- "same prompt" is
    then literally true regardless of which engine tokenizes it.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    messages = [{"role": "user", "content": content}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_tokens = len(tokenizer(text)["input_ids"])
    return text, prompt_tokens


def gpu_memory_used_mb(gpu_index=0):
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--id={gpu_index}", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return int(out.stdout.strip().splitlines()[0])
    except Exception as e:
        log(f"WARNING: could not read GPU memory via nvidia-smi ({type(e).__name__}: {e}); reporting -1")
        return -1


def get_environment_info():
    info = {}
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        info["nvidia_smi"] = out.stdout.strip()
    except Exception as e:
        info["nvidia_smi"] = f"unavailable ({type(e).__name__})"
    return info


def summarize(name, values):
    return {
        f"{name}_median": statistics.median(values),
        f"{name}_mean": statistics.mean(values),
        f"{name}_values": values,
    }


def compute_metrics(ttft_samples, total_samples, tokens):
    tpot_samples = [(t - ttft) / (tokens - 1) for t, ttft in zip(total_samples, ttft_samples)]
    e2e_tps_samples = [tokens / t for t in total_samples]
    decode_tps_samples = [1.0 / tpot for tpot in tpot_samples]

    result = {}
    result.update(summarize("ttft_s", ttft_samples))
    result.update(summarize("total_s", total_samples))
    result.update(summarize("tpot_s", tpot_samples))
    result.update(summarize("tokens_per_sec_e2e", e2e_tps_samples))
    result.update(summarize("tokens_per_sec_decode", decode_tps_samples))
    return result


def save_result(engine, payload, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"bench_vs_vllm_{engine}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    log(f"Saved {path}")
    return path


# ---------------------------------------------------------------------
# Our engine
# ---------------------------------------------------------------------
def run_ours(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from qwen_optimizer import optimize

    if not torch.cuda.is_available():
        raise RuntimeError("--engine ours requires a CUDA GPU.")

    vram_idle = gpu_memory_used_mb()

    prompt_text, prompt_tokens = render_prompt(args.prompt)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    log("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="cuda")
    model.eval()
    model = optimize(model)  # unmodified public API; no internals touched

    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    vram_loaded = gpu_memory_used_mb()

    log("Warming up (also materializes StaticCache + torch.compile CUDA graphs)...")
    for _ in range(2):
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=8, min_new_tokens=8, do_sample=False, use_cache=True)
    torch.cuda.synchronize()
    vram_after_warmup = gpu_memory_used_mb()

    log(f"Measuring TTFT over {args.runs} runs (max_new_tokens=1)...")
    ttft_samples = []
    for _ in range(args.runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=1, min_new_tokens=1, do_sample=False, use_cache=True)
        torch.cuda.synchronize()
        ttft_samples.append(time.perf_counter() - t0)

    log(f"Measuring total time for {args.tokens} tokens over {args.runs} runs...")
    total_samples = []
    vram_after_run = vram_after_warmup
    for _ in range(args.runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.tokens,
                min_new_tokens=args.tokens,
                do_sample=False,
                use_cache=True,
            )
        torch.cuda.synchronize()
        total_samples.append(time.perf_counter() - t0)
        vram_after_run = max(vram_after_run, gpu_memory_used_mb())

        generated = out.shape[1] - inputs["input_ids"].shape[1]
        if generated != args.tokens:
            raise RuntimeError(f"Expected exactly {args.tokens} generated tokens, got {generated}")

    metrics = compute_metrics(ttft_samples, total_samples, args.tokens)
    payload = {
        "engine": "ours",
        "model": MODEL_NAME,
        "prompt": prompt_text,
        "prompt_tokens": prompt_tokens,
        "requested_tokens": args.tokens,
        "runs": args.runs,
        "vram_idle_mb": vram_idle,
        "vram_loaded_mb": vram_loaded,
        "vram_after_warmup_mb": vram_after_warmup,
        "vram_peak_mb": vram_after_run,
        "environment": get_environment_info(),
        "metrics": metrics,
    }
    return payload


# ---------------------------------------------------------------------
# vLLM engine
# ---------------------------------------------------------------------
def run_vllm(args):
    vram_idle = gpu_memory_used_mb()

    prompt_text, prompt_tokens = render_prompt(args.prompt)

    from vllm import LLM, SamplingParams

    log("Constructing vLLM engine (this pre-allocates its KV cache pool)...")
    llm = LLM(
        model=MODEL_NAME,
        dtype="float16",  # T4 (compute capability 7.5) predates good bf16 support; match our fp16 baseline
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=False,  # allow vLLM's own CUDA graph capture -- don't handicap it
    )
    vram_loaded = gpu_memory_used_mb()

    warmup_params = SamplingParams(temperature=0, top_p=1.0, max_tokens=8, min_tokens=8, ignore_eos=True)
    ttft_params = SamplingParams(temperature=0, top_p=1.0, max_tokens=1, min_tokens=1, ignore_eos=True)
    full_params = SamplingParams(temperature=0, top_p=1.0, max_tokens=args.tokens, min_tokens=args.tokens, ignore_eos=True)

    log("Warming up...")
    for _ in range(2):
        llm.generate([prompt_text], warmup_params, use_tqdm=False)
    vram_after_warmup = gpu_memory_used_mb()

    log(f"Measuring TTFT over {args.runs} runs (max_tokens=1)...")
    ttft_samples = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        llm.generate([prompt_text], ttft_params, use_tqdm=False)
        ttft_samples.append(time.perf_counter() - t0)

    log(f"Measuring total time for {args.tokens} tokens over {args.runs} runs...")
    total_samples = []
    vram_after_run = vram_after_warmup
    for _ in range(args.runs):
        t0 = time.perf_counter()
        outputs = llm.generate([prompt_text], full_params, use_tqdm=False)
        total_samples.append(time.perf_counter() - t0)
        vram_after_run = max(vram_after_run, gpu_memory_used_mb())

        generated = len(outputs[0].outputs[0].token_ids)
        if generated != args.tokens:
            raise RuntimeError(f"Expected exactly {args.tokens} generated tokens, got {generated}")

    metrics = compute_metrics(ttft_samples, total_samples, args.tokens)
    payload = {
        "engine": "vllm",
        "model": MODEL_NAME,
        "prompt": prompt_text,
        "prompt_tokens": prompt_tokens,
        "requested_tokens": args.tokens,
        "runs": args.runs,
        "vllm_gpu_memory_utilization": args.vllm_gpu_memory_utilization,
        "vram_idle_mb": vram_idle,
        "vram_loaded_mb": vram_loaded,
        "vram_after_warmup_mb": vram_after_warmup,
        "vram_peak_mb": vram_after_run,
        "environment": get_environment_info(),
        "metrics": metrics,
    }
    return payload


# ---------------------------------------------------------------------
# Compare (no GPU / engine imports needed)
# ---------------------------------------------------------------------
def run_compare(args):
    path_ours = os.path.join(args.out_dir, "bench_vs_vllm_ours.json")
    path_vllm = os.path.join(args.out_dir, "bench_vs_vllm_vllm.json")

    for p in (path_ours, path_vllm):
        if not os.path.exists(p):
            raise RuntimeError(f"Missing {p}. Run --engine ours and --engine vllm first.")

    with open(path_ours) as f:
        ours = json.load(f)
    with open(path_vllm) as f:
        vllm = json.load(f)

    if ours["prompt"] != vllm["prompt"]:
        print("WARNING: prompts differ between the two result files!")
    if ours["prompt_tokens"] != vllm["prompt_tokens"]:
        print(
            f"WARNING: prompt_tokens differ (ours={ours['prompt_tokens']}, "
            f"vllm={vllm['prompt_tokens']}) -- tokenization may not match."
        )
    if ours["requested_tokens"] != vllm["requested_tokens"]:
        raise RuntimeError("requested_tokens differ between the two runs; not comparable.")

    om, vm = ours["metrics"], vllm["metrics"]

    def row(label, key, unit="", fmt="{:.4f}"):
        o, v = om[key], vm[key]
        ratio = o / v if v else float("nan")
        print(f"{label:<28}{fmt.format(o):>14}{fmt.format(v):>14}{unit:>6}{ratio:>11.2f}x")

    print("\n" + "=" * 76)
    print(f"COMPARISON: ours vs vLLM  ({ours['requested_tokens']} generated tokens, batch=1)")
    print(f"model: {ours['model']}")
    print(f"prompt_tokens: ours={ours['prompt_tokens']}  vllm={vllm['prompt_tokens']}")
    print("=" * 76)
    print(f"{'metric':<28}{'ours':>14}{'vllm':>14}{'unit':>6}{'ratio(o/v)':>12}")
    print("-" * 76)
    row("TTFT (median)", "ttft_s_median", "s")
    row("TPOT (median)", "tpot_s_median", "s")
    row("tok/s end-to-end (median)", "tokens_per_sec_e2e_median", "")
    row("tok/s decode-only (median)", "tokens_per_sec_decode_median", "")
    print("-" * 76)
    print(f"{'VRAM loaded (MB)':<28}{ours['vram_loaded_mb']:>14}{vllm['vram_loaded_mb']:>14}")
    print(f"{'VRAM peak (MB)':<28}{ours['vram_peak_mb']:>14}{vllm['vram_peak_mb']:>14}")
    print("=" * 76)
    print(
        "\nNote: vLLM ran unquantized FP16 through its own standard serving path "
        "(not handicapped); ours ran the INT8+StaticCache+compile path. This is "
        "'our best vs vLLM's best-out-of-the-box default', not a controlled-for-"
        f"quantization comparison. vLLM's gpu_memory_utilization was set to "
        f"{vllm.get('vllm_gpu_memory_utilization', 'unknown')} for this run -- VRAM "
        "numbers are not meaningful without knowing that value, since vLLM "
        "pre-allocates its KV pool according to it up front."
    )
    print("=" * 76)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True, choices=["ours", "vllm", "compare"])
    parser.add_argument("--tokens", type=int, default=96)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT_CONTENT)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--out-dir", type=str, default=RESULTS_DIR)
    args = parser.parse_args()

    if args.engine == "compare":
        run_compare(args)
        return

    fn = run_ours if args.engine == "ours" else run_vllm
    payload = fn(args)

    m = payload["metrics"]
    print("\n" + "=" * 60)
    print(f"RESULT: engine={payload['engine']}")
    print(f"  TTFT (median):              {m['ttft_s_median']:.4f} s")
    print(f"  TPOT (median):              {m['tpot_s_median']:.4f} s")
    print(f"  tok/s end-to-end (median):  {m['tokens_per_sec_e2e_median']:.2f}")
    print(f"  tok/s decode-only (median): {m['tokens_per_sec_decode_median']:.2f}")
    print(f"  VRAM loaded / peak (MB):    {payload['vram_loaded_mb']} / {payload['vram_peak_mb']}")
    print("=" * 60)

    save_result(payload["engine"], payload, args.out_dir)


if __name__ == "__main__":
    main()
