"""
Apples-to-apples benchmark: our optimized Qwen2.5-1.5B decode path
(qwen_optimizer.optimize(), unmodified) vs vLLM, same model, same GPU,
same prompt, same 96 generated tokens, swept across batch sizes
1, 2, 4, 8, 16, 32.

Run as TWO SEPARATE PROCESSES (recommended: two separate Colab
runtimes, see "T4/vLLM gotchas" below) so neither engine's dependencies
or GPU memory footprint can interfere with the other's measurement:

    python bench_vs_vllm.py --engine ours
    python bench_vs_vllm.py --engine vllm

Each saves one JSON per batch size to
results/bench_vs_vllm_<engine>_bs<N>.json. Once both engines' files
exist (copy them into the same place if run in separate runtimes):

    python bench_vs_vllm.py --engine compare

prints the side-by-side tables, one row per batch size per metric, so
you can see exactly where (if anywhere) vLLM starts winning. This mode
does not touch the GPU or import either engine's dependencies -- it
only reads the JSON files.

--------------------------------------------------------------------
IMPORTANT: what "ours" actually measures above batch_size=1
--------------------------------------------------------------------

qwen_optimizer.optimize() (unmodified here, per instructions) only
engages its INT8 + StaticCache + torch.compile fast path for batch=1
requests -- this is an existing, deliberate design choice
(qwen_optimizer/core.py's `_resolve_supported_call` allowlist requires
`input_ids.shape[0] == 1`; CLAUDE.md's own constraint is "batch=1
autoregressive decode is the primary target"). For batch_size in
{2,4,8,16,32}, our wrapped generate() transparently falls back to the
real, unmodified `model.generate()` -- plain PyTorch FP16, DynamicCache,
no compile, no INT8. So the "ours" curve across batch sizes is: our
specialized batch=1 path at bs=1, then the vanilla PyTorch baseline at
every larger batch size. That is not a bug in this benchmark and was
not changed to "help" or "hurt" either engine -- it is exactly what
calling qwen_optimizer.optimize() already does, unmodified. Each saved
JSON records `used_optimized_path: bool` so this is explicit in the
data, not just this docstring.

--------------------------------------------------------------------
Methodology (identical for both engines at every batch size)
--------------------------------------------------------------------

Each batch size uses `batch_size` COPIES of the exact same rendered
prompt (same content -> automatically same length, avoiding any need
to hand-construct distinct equal-length prompts). vLLM's automatic
prefix caching is explicitly disabled (`enable_prefix_caching=False`)
specifically because the copies are identical -- with prefix caching
on, vLLM could compute the shared prefill ONCE and reuse it across all
copies, which no realistic multi-request workload (distinct users)
would get, and our engine has no equivalent optimization to compare
against. Disabling it keeps both engines doing the same amount of real
work: full independent prefill for every sequence in the batch.

TTFT (time to first token): a separate call capped at exactly 1 output
token per sequence in the batch. For our engine at batch=1 this is
exact (decode_loop.greedy_decode_static's loop runs zero iterations at
max_new_tokens=1, so the timed call is pure prefill); at batch>1 it's
the fallback HF generate()'s prefill + one decode step. For vLLM,
max_tokens=1 across the batch is the standard way to isolate
prefill+first-token latency for a non-streaming batch request. This is
a per-BATCH-CALL TTFT (when the whole synchronous call returns), not a
per-individual-request streaming timestamp -- neither engine's simple
offline/blocking API exposes the latter without deeper instrumentation.

TPOT (time per output token) is derived, not measured directly:

    TPOT = (total_time_for_96 - TTFT) / (96 - 1)

same formula both engines, same two calls. This is a per-DECODE-STEP
latency for the whole batch (i.e. time to advance every sequence in the
batch by one token), the natural definition for batched serving.

Per-request throughput = 96 / total_time_for_96 (what one user in that
batch actually experiences).
Total throughput = batch_size * per-request throughput (aggregate
tokens/sec across all requests in the batch -- what a server operator
cares about for capacity).

VRAM is read via `nvidia-smi --query-gpu=memory.used` (system-wide, any
process) at four points per batch size: idle, loaded, post-warmup,
post-run. For "ours", `torch.cuda.empty_cache()` is called between
batch sizes so each size's reading isn't inflated by a previous,
larger, allocator-reserved-but-now-unused pool. IMPORTANT CAVEAT for
vLLM: it pre-allocates ONE fixed KV-cache block pool at engine
construction, sized from --vllm-gpu-memory-utilization -- NOT from
however many requests you actually send. So vLLM's vram_loaded/peak
will likely look nearly IDENTICAL across every batch size in this
sweep; that is expected and is itself a real difference between the
two engines' memory models (ours grows organically with actual usage,
vLLM pre-commits a budget upfront), not a measurement bug.

vLLM runs the plain, unquantized FP16 checkpoint through its own
standard (CUDA-graph-capturing) serving path -- i.e. vLLM's own
out-of-the-box best, not handicapped. This is intentionally an
"our best vs vLLM's best-out-of-the-box default" comparison, NOT a
controlled-for-quantization one.

--------------------------------------------------------------------
T4 / vLLM gotchas (read before running --engine vllm)
--------------------------------------------------------------------

- vLLM pins its own torch (and sometimes triton) version per release.
  Installing it into the same environment as this project's
  transformers==4.46.3 / triton stack risks pip silently upgrading or
  downgrading packages this project depends on. RECOMMENDED: run
  `--engine vllm` in a fresh Colab runtime with only `pip install vllm`
  (plus whatever it pulls in), and run `--engine ours` in this
  project's normal environment. Bring the JSON files together
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
  default of 0.9). At batch_size=32 with only 96 tokens per sequence
  the actual KV-cache need is tiny for a 1.5B model, so this should
  still comfortably fit; raise it if vLLM's scheduler complains it
  can't fit the batch.
"""

import argparse
import json
import os
import statistics
import subprocess
import time

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_PROMPT_CONTENT = "Explain how a CPU works in a few sentences."
DEFAULT_BATCH_SIZES = [1, 2, 4, 8, 16, 32]
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


def parse_batch_sizes(spec):
    return [int(x) for x in spec.split(",") if x.strip()]


def summarize(name, values):
    return {
        f"{name}_median": statistics.median(values),
        f"{name}_mean": statistics.mean(values),
        f"{name}_values": values,
    }


def compute_metrics(ttft_samples, total_samples, tokens, batch_size):
    tpot_samples = [(t - ttft) / (tokens - 1) for t, ttft in zip(total_samples, ttft_samples)]
    per_req_tps_e2e = [tokens / t for t in total_samples]
    per_req_tps_decode = [1.0 / tpot for tpot in tpot_samples]
    total_tps_e2e = [batch_size * v for v in per_req_tps_e2e]
    total_tps_decode = [batch_size * v for v in per_req_tps_decode]

    result = {}
    result.update(summarize("ttft_s", ttft_samples))
    result.update(summarize("total_s", total_samples))
    result.update(summarize("tpot_s", tpot_samples))
    result.update(summarize("per_request_tokens_per_sec_e2e", per_req_tps_e2e))
    result.update(summarize("per_request_tokens_per_sec_decode", per_req_tps_decode))
    result.update(summarize("total_tokens_per_sec_e2e", total_tps_e2e))
    result.update(summarize("total_tokens_per_sec_decode", total_tps_decode))
    return result


def save_result(engine, batch_size, payload, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"bench_vs_vllm_{engine}_bs{batch_size}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    log(f"Saved {path}")
    return path


def print_single_result(payload):
    m = payload["metrics"]
    print("\n" + "=" * 64)
    print(f"RESULT: engine={payload['engine']}  batch_size={payload['batch_size']}"
          f"  used_optimized_path={payload['used_optimized_path']}")
    print(f"  TTFT (median):                    {m['ttft_s_median']:.4f} s")
    print(f"  TPOT (median):                    {m['tpot_s_median']:.4f} s")
    print(f"  per-request tok/s e2e (median):   {m['per_request_tokens_per_sec_e2e_median']:.2f}")
    print(f"  total tok/s e2e (median):         {m['total_tokens_per_sec_e2e_median']:.2f}")
    print(f"  VRAM loaded / peak (MB):          {payload['vram_loaded_mb']} / {payload['vram_peak_mb']}")
    print("=" * 64)


# ---------------------------------------------------------------------
# Our engine
# ---------------------------------------------------------------------
def bench_ours_at_batch_size(model, tokenizer, batch_size, prompt_text, prompt_tokens, args):
    import torch

    batch_prompts = [prompt_text] * batch_size
    tokenizer.padding_side = "left"  # required for correct batched decoder-only generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(model.device)

    vram_loaded = gpu_memory_used_mb()

    log(f"[ours bs={batch_size}] Warming up...")
    for _ in range(2):
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=8, min_new_tokens=8, do_sample=False, use_cache=True)
    torch.cuda.synchronize()
    vram_after_warmup = gpu_memory_used_mb()

    log(f"[ours bs={batch_size}] Measuring TTFT over {args.runs} runs (max_new_tokens=1)...")
    ttft_samples = []
    for _ in range(args.runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=1, min_new_tokens=1, do_sample=False, use_cache=True)
        torch.cuda.synchronize()
        ttft_samples.append(time.perf_counter() - t0)

    log(f"[ours bs={batch_size}] Measuring total time for {args.tokens} tokens over {args.runs} runs...")
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
        if out.shape[0] != batch_size:
            raise RuntimeError(f"Expected batch_size={batch_size} outputs, got {out.shape[0]}")

    metrics = compute_metrics(ttft_samples, total_samples, args.tokens, batch_size)
    payload = {
        "engine": "ours",
        "model": MODEL_NAME,
        "batch_size": batch_size,
        "used_optimized_path": (batch_size == 1),  # qwen_optimizer's own allowlist; see module docstring
        "prompt": prompt_text,
        "prompt_tokens": prompt_tokens,
        "requested_tokens": args.tokens,
        "runs": args.runs,
        "vram_loaded_mb": vram_loaded,
        "vram_after_warmup_mb": vram_after_warmup,
        "vram_peak_mb": vram_after_run,
        "metrics": metrics,
    }

    torch.cuda.empty_cache()  # avoid inflating the NEXT (larger) batch size's "loaded" reading
    return payload


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

    env = get_environment_info()
    results = []
    for batch_size in args.batch_sizes:
        payload = bench_ours_at_batch_size(model, tokenizer, batch_size, prompt_text, prompt_tokens, args)
        payload["vram_idle_mb"] = vram_idle
        payload["environment"] = env
        print_single_result(payload)
        save_result("ours", batch_size, payload, args.out_dir)
        results.append(payload)

    return results


# ---------------------------------------------------------------------
# vLLM engine
# ---------------------------------------------------------------------
def bench_vllm_at_batch_size(llm, batch_size, prompt_text, prompt_tokens, args):
    from vllm import SamplingParams

    batch_prompts = [prompt_text] * batch_size

    warmup_params = SamplingParams(temperature=0, top_p=1.0, max_tokens=8, min_tokens=8, ignore_eos=True)
    ttft_params = SamplingParams(temperature=0, top_p=1.0, max_tokens=1, min_tokens=1, ignore_eos=True)
    full_params = SamplingParams(
        temperature=0, top_p=1.0, max_tokens=args.tokens, min_tokens=args.tokens, ignore_eos=True
    )

    log(f"[vllm bs={batch_size}] Warming up...")
    for _ in range(2):
        llm.generate(batch_prompts, warmup_params, use_tqdm=False)
    vram_after_warmup = gpu_memory_used_mb()

    log(f"[vllm bs={batch_size}] Measuring TTFT over {args.runs} runs (max_tokens=1)...")
    ttft_samples = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        llm.generate(batch_prompts, ttft_params, use_tqdm=False)
        ttft_samples.append(time.perf_counter() - t0)

    log(f"[vllm bs={batch_size}] Measuring total time for {args.tokens} tokens over {args.runs} runs...")
    total_samples = []
    vram_after_run = vram_after_warmup
    for _ in range(args.runs):
        t0 = time.perf_counter()
        outputs = llm.generate(batch_prompts, full_params, use_tqdm=False)
        total_samples.append(time.perf_counter() - t0)
        vram_after_run = max(vram_after_run, gpu_memory_used_mb())

        if len(outputs) != batch_size:
            raise RuntimeError(f"Expected batch_size={batch_size} outputs, got {len(outputs)}")
        for o in outputs:
            generated = len(o.outputs[0].token_ids)
            if generated != args.tokens:
                raise RuntimeError(f"Expected exactly {args.tokens} generated tokens, got {generated}")

    metrics = compute_metrics(ttft_samples, total_samples, args.tokens, batch_size)
    return {
        "engine": "vllm",
        "model": MODEL_NAME,
        "batch_size": batch_size,
        "used_optimized_path": True,  # vLLM always uses its own standard serving path regardless of batch size
        "prompt": prompt_text,
        "prompt_tokens": prompt_tokens,
        "requested_tokens": args.tokens,
        "runs": args.runs,
        "vllm_gpu_memory_utilization": args.vllm_gpu_memory_utilization,
        "vram_after_warmup_mb": vram_after_warmup,
        "vram_peak_mb": vram_after_run,
        "metrics": metrics,
    }


def run_vllm(args):
    vram_idle = gpu_memory_used_mb()
    prompt_text, prompt_tokens = render_prompt(args.prompt)

    from vllm import LLM

    log("Constructing vLLM engine (this pre-allocates its KV cache pool ONCE for the whole sweep)...")
    llm = LLM(
        model=MODEL_NAME,
        dtype="float16",  # T4 (compute capability 7.5) predates good bf16 support; match our fp16 baseline
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=max(args.batch_sizes + [256]),
        enforce_eager=False,  # allow vLLM's own CUDA graph capture -- don't handicap it
        enable_prefix_caching=False,  # batch members are identical copies; see module docstring
    )
    vram_loaded = gpu_memory_used_mb()

    env = get_environment_info()
    results = []
    for batch_size in args.batch_sizes:
        payload = bench_vllm_at_batch_size(llm, batch_size, prompt_text, prompt_tokens, args)
        payload["vram_idle_mb"] = vram_idle
        payload["vram_loaded_mb"] = vram_loaded
        payload["environment"] = env
        print_single_result(payload)
        save_result("vllm", batch_size, payload, args.out_dir)
        results.append(payload)

    return results


# ---------------------------------------------------------------------
# Compare (no GPU / engine imports needed)
# ---------------------------------------------------------------------
def _load_pair(batch_size, out_dir):
    path_ours = os.path.join(out_dir, f"bench_vs_vllm_ours_bs{batch_size}.json")
    path_vllm = os.path.join(out_dir, f"bench_vs_vllm_vllm_bs{batch_size}.json")
    missing = [p for p in (path_ours, path_vllm) if not os.path.exists(p)]
    if missing:
        log(f"SKIPPING batch_size={batch_size}: missing {missing}")
        return None
    with open(path_ours) as f:
        ours = json.load(f)
    with open(path_vllm) as f:
        vllm = json.load(f)
    if ours["requested_tokens"] != vllm["requested_tokens"]:
        raise RuntimeError(f"requested_tokens differ at batch_size={batch_size}; not comparable.")
    return ours, vllm


def _print_metric_table(title, key, pairs, fmt="{:.4f}"):
    print(f"\n{title}")
    print(f"{'batch_size':<12}{'ours':>14}{'vllm':>14}{'ratio(o/v)':>14}")
    print("-" * 54)
    for batch_size, (ours, vllm) in pairs.items():
        o = ours["metrics"][key]
        v = vllm["metrics"][key]
        ratio = o / v if v else float("nan")
        flag = "  <-- vLLM wins" if ratio < 1.0 else ""
        print(f"{batch_size:<12}{fmt.format(o):>14}{fmt.format(v):>14}{ratio:>13.2f}x{flag}")


def run_compare(args):
    pairs = {}
    for batch_size in args.batch_sizes:
        pair = _load_pair(batch_size, args.out_dir)
        if pair is not None:
            pairs[batch_size] = pair

    if not pairs:
        raise RuntimeError("No matching (ours, vllm) result pairs found for any requested batch size.")

    any_ours, any_vllm = next(iter(pairs.values()))
    print("\n" + "=" * 78)
    print(f"COMPARISON: ours vs vLLM across batch sizes {list(pairs.keys())}")
    print(f"model: {any_ours['model']}  |  requested_tokens: {any_ours['requested_tokens']}")
    print("=" * 78)

    _print_metric_table("Total throughput (tok/s, end-to-end, median)", "total_tokens_per_sec_e2e_median", pairs, "{:.1f}")
    _print_metric_table("Per-request throughput (tok/s, end-to-end, median)", "per_request_tokens_per_sec_e2e_median", pairs, "{:.1f}")
    _print_metric_table("TTFT (s, median)", "ttft_s_median", pairs)
    _print_metric_table("TPOT (s, median)", "tpot_s_median", pairs)

    print("\nVRAM peak (MB)")
    print(f"{'batch_size':<12}{'ours':>14}{'vllm':>14}")
    print("-" * 40)
    for batch_size, (ours, vllm) in pairs.items():
        print(f"{batch_size:<12}{ours['vram_peak_mb']:>14}{vllm['vram_peak_mb']:>14}")

    print("\n" + "=" * 78)
    used_optimized = {bs: ours["used_optimized_path"] for bs, (ours, _) in pairs.items()}
    print(f"ours used_optimized_path per batch size: {used_optimized}")
    print(
        "For batch_size > 1 where used_optimized_path=False, 'ours' is the plain "
        "unmodified HF generate() fallback (qwen_optimizer's own existing "
        "behavior, not a change made for this benchmark) -- see module docstring."
    )
    print(
        "vLLM's VRAM figures reflect its fixed pre-allocated KV pool "
        f"(gpu_memory_utilization={any_vllm.get('vllm_gpu_memory_utilization', 'unknown')}), "
        "not per-batch-size actual need -- expect it to look roughly constant "
        "across batch sizes."
    )
    print("=" * 78)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True, choices=["ours", "vllm", "compare"])
    parser.add_argument("--tokens", type=int, default=96)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT_CONTENT)
    parser.add_argument("--batch-sizes", type=str, default=",".join(str(b) for b in DEFAULT_BATCH_SIZES))
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--out-dir", type=str, default=RESULTS_DIR)
    args = parser.parse_args()
    args.batch_sizes = parse_batch_sizes(args.batch_sizes)

    if args.engine == "compare":
        run_compare(args)
        return

    fn = run_ours if args.engine == "ours" else run_vllm
    fn(args)


if __name__ == "__main__":
    main()
