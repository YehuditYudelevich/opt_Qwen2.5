"""
Single entry point for the batch-1 experiment set described in
RESULTS.md / NEXT_STEPS.md.

    python run_t4_experiments.py
    python run_t4_experiments.py --quick      # fast smoke test, ~2 min
    python run_t4_experiments.py --runs 10 --tokens 128

What it does, in order, all in one GPU session:

  1. Print environment info (GPU, compute capability, library versions,
     best-effort clocks/utilization).
  2. Load the model once.
  3. Run the correctness + quality gate suite (test_correctness.py).
     Anything that fails gates OFF the variants that depend on it --
     broken code never gets benchmarked for speed.
  4. Benchmark, same-session, interleaved as
     A0, B, A1, C, A2, D, A3, E, A4, F, A5
     so every candidate gets a baseline measured immediately before and
     after it (baseline pairs are shared between neighbors to avoid
     wasting time on redundant measurements).
  5. Profile the baseline and the best surviving candidate with
     torch.profiler, CUDA-time sorted.
  6. Save every raw number to results/run_<timestamp>.json.
  7. Print one concise summary table with a >=2x verdict.

Variants:
  A  baseline: unmodified model.generate()
  B  manual greedy loop + DynamicCache            (isolates generate() overhead)
  C  manual greedy loop + StaticCache             (isolates cache-format overhead)
  D  C + torch.compile(mode="reduce-overhead")    (CUDA graphs)
  E  INT8 weight-only (fused QKV, fused gate/up, o/down/lm_head) + DynamicCache
  F  D + E combined                                (main >=2x candidate)
"""

import argparse
import json
import os
import statistics
import subprocess
import time
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import decode_loop
import optimized_model
import test_correctness

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def get_environment_info():
    info = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["compute_capability"] = torch.cuda.get_device_capability(0)

    try:
        import triton
        info["triton"] = triton.__version__
    except ImportError:
        info["triton"] = "not installed"

    try:
        import transformers
        info["transformers"] = transformers.__version__
    except ImportError:
        info["transformers"] = "not installed"

    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=clocks.sm,clocks.mem,temperature.gpu,utilization.gpu,power.draw",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        info["nvidia_smi"] = out.stdout.strip()
    except Exception as e:
        info["nvidia_smi"] = f"unavailable ({type(e).__name__})"

    return info


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    model.eval()
    return tokenizer, model


def make_inputs(tokenizer):
    messages = [{"role": "user", "content": "Explain how a CPU works in a few sentences."}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer(text, return_tensors="pt").to("cuda")


def benchmark_variant(gen_fn, runs, new_tokens, warmup=2):
    """
    gen_fn(new_tokens) -> output_ids. Returns list of tok/s, one per run.
    Same warmup/sync/timing discipline as the existing benchmark.py.
    """
    with torch.no_grad():
        for _ in range(warmup):
            _ = gen_fn(32)
    torch.cuda.synchronize()

    vals = []
    for _ in range(runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            out = gen_fn(new_tokens)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        tok_s = new_tokens / elapsed
        vals.append(tok_s)

    return vals


def compile_with_fallback(model, static_cache, warmup_input_ids, tag):
    for fullgraph in (True, False):
        try:
            step = decode_loop.make_compiled_decode_step(model, fullgraph=fullgraph)
            static_cache.reset()
            _ = decode_loop.greedy_decode_static(model, warmup_input_ids, 6, static_cache, decode_step_fn=step)
            torch.cuda.synchronize()
            log(f"[{tag}] torch.compile succeeded with fullgraph={fullgraph}")
            return step, f"reduce-overhead(fullgraph={fullgraph})"
        except Exception as e:
            log(f"[{tag}] torch.compile fullgraph={fullgraph} failed: {type(e).__name__}: {e}")
    log(f"[{tag}] torch.compile unavailable, falling back to eager StaticCache (no CUDA graph)")
    return None, "eager_fallback (compile failed)"


def profile_variant(model, gen_fn, tag, tokens=24):
    from torch.profiler import profile, ProfilerActivity

    with torch.no_grad():
        _ = gen_fn(16)
    torch.cuda.synchronize()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        with torch.no_grad():
            _ = gen_fn(tokens)
    torch.cuda.synchronize()

    table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=25)
    log(f"\n===== PROFILE: {tag} (sorted by CUDA time) =====\n{table}")
    return str(table)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--tokens", type=int, default=96)
    parser.add_argument("--quick", action="store_true", help="Fast smoke test: runs=2, tokens=24")
    args = parser.parse_args()

    if args.quick:
        args.runs = 2
        args.tokens = 24

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required. This must be run on the Tesla T4 Colab session.")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    env = get_environment_info()
    log("Environment:")
    for k, v in env.items():
        log(f"  {k}: {v}")

    log("Loading model...")
    tokenizer, model = load_model()
    inputs = make_inputs(tokenizer)
    prompt_len = inputs["input_ids"].shape[1]

    log("Installing INT8 quantized modules (toggle-able, does not affect FP16 path when disabled)...")
    optimized_model.install_int8_quant(model)
    report = optimized_model.quantization_report(model)
    log(
        f"Quantization footprint: FP16={report['fp16_gb']:.3f} GB, "
        f"INT8={report['int8_gb']:.3f} GB, ratio={report['ratio']:.2f}x"
    )

    # ---------------- Correctness + quality gate ----------------
    log("\n===== CORRECTNESS + QUALITY GATE =====")
    correctness_results = test_correctness.run_all(model, tokenizer, optimized_model.set_int8)
    for r in correctness_results:
        status = "PASS" if r["passed"] else "FAIL"
        log(f"[{status}] {r['name']}: {r['details']}")

    def _passed(name_prefix):
        matches = [r for r in correctness_results if r["name"].startswith(name_prefix)]
        return len(matches) > 0 and all(r["passed"] for r in matches)

    dynamic_loop_ok = _passed("manual_loop_dynamic_matches_generate")
    static_loop_ok = _passed("manual_loop_static_matches_generate")
    manual_loop_ok = dynamic_loop_ok and static_loop_ok
    int8_kernel_ok = _passed("int8_gemv_kernel") and _passed("fused_int8_gate_up_kernel")
    int8_quality_ok = _passed("int8_quality")
    int8_ok = int8_kernel_ok and int8_quality_ok

    log(
        f"\nGate summary: dynamic_loop_ok={dynamic_loop_ok}, static_loop_ok={static_loop_ok}, "
        f"int8_kernel_ok={int8_kernel_ok}, int8_quality_ok={int8_quality_ok}"
    )

    optimized_model.set_int8(model, False)

    # ---------------- Build variant callables ----------------
    max_cache_len = prompt_len + max(args.tokens, 32) + 8

    static_cache_fp16 = decode_loop.make_static_cache(model, max_cache_len)
    static_cache_int8 = decode_loop.make_static_cache(model, max_cache_len)

    compiled_step_fp16 = None
    compiled_mode_fp16 = "not attempted"
    compiled_step_int8 = None
    compiled_mode_int8 = "not attempted"

    if static_loop_ok:
        optimized_model.set_int8(model, False)
        log("\nCompiling decode step (FP16 path) for variant D...")
        compiled_step_fp16, compiled_mode_fp16 = compile_with_fallback(
            model, static_cache_fp16, inputs["input_ids"], "D-fp16"
        )

        if int8_ok:
            optimized_model.set_int8(model, True)
            log("\nCompiling decode step (INT8 path) for variant F...")
            compiled_step_int8, compiled_mode_int8 = compile_with_fallback(
                model, static_cache_int8, inputs["input_ids"], "F-int8"
            )
            optimized_model.set_int8(model, False)

    # NOTE: each gen_* function assumes the caller has already put the
    # model in the right INT8 on/off state (see `set_state` below). This
    # keeps the toggle out of the timed region and makes each variant's
    # required state explicit and auditable in one place.
    def gen_A(n):
        return model.generate(**inputs, max_new_tokens=n, min_new_tokens=n, do_sample=False, use_cache=True)

    def gen_B(n):
        return decode_loop.greedy_decode_dynamic(model, inputs["input_ids"], n)

    def gen_C(n):
        static_cache_fp16.reset()
        return decode_loop.greedy_decode_static(model, inputs["input_ids"], n, static_cache_fp16)

    def gen_D(n):
        static_cache_fp16.reset()
        return decode_loop.greedy_decode_static(
            model, inputs["input_ids"], n, static_cache_fp16, decode_step_fn=compiled_step_fp16
        )

    def gen_E(n):
        return decode_loop.greedy_decode_dynamic(model, inputs["input_ids"], n)

    def gen_F(n):
        static_cache_int8.reset()
        return decode_loop.greedy_decode_static(
            model, inputs["input_ids"], n, static_cache_int8, decode_step_fn=compiled_step_int8
        )

    # variant name -> (gen_fn, whether INT8 must be enabled for this variant)
    VARIANT_INT8_STATE = {
        "A_baseline": False,
        "B_manual_dynamic_fp16": False,
        "C_manual_static_fp16": False,
        "D_compiled_static_fp16": False,
        "E_int8_manual_dynamic": True,
        "F_compiled_static_int8": True,
    }

    def set_state(name):
        optimized_model.set_int8(model, VARIANT_INT8_STATE[name])

    candidates = []
    if dynamic_loop_ok:
        candidates.append(("B_manual_dynamic_fp16", gen_B))
    else:
        log("SKIPPING variant B: manual-loop-dynamic-matches-generate correctness gate failed.")

    if static_loop_ok:
        candidates.append(("C_manual_static_fp16", gen_C))
        candidates.append(("D_compiled_static_fp16", gen_D))
    else:
        log("SKIPPING variants C/D: manual-loop-static-matches-generate correctness gate failed.")

    if int8_ok:
        if dynamic_loop_ok:
            candidates.append(("E_int8_manual_dynamic", gen_E))
        else:
            log("SKIPPING variant E: requires dynamic-loop correctness gate, which failed.")
        if static_loop_ok:
            candidates.append(("F_compiled_static_int8", gen_F))
        else:
            log("SKIPPING variant F: requires static-loop correctness gate, which failed.")
    else:
        log("SKIPPING variants E/F: INT8 correctness/quality gate failed.")

    # ---------------- Same-session interleaved A/B/A benchmark ----------------
    log(f"\n===== BENCHMARK (runs={args.runs}, tokens={args.tokens}) =====")

    baseline_runs = []
    bench_results = {}

    log("Measuring A0 (baseline)...")
    set_state("A_baseline")
    a_vals = benchmark_variant(gen_A, args.runs, args.tokens)
    baseline_runs.append(a_vals)
    log(f"  A0 median: {statistics.median(a_vals):.2f} tok/s")

    for name, fn in candidates:
        log(f"Measuring {name}...")
        set_state(name)
        try:
            x_vals = benchmark_variant(fn, args.runs, args.tokens)
            log(f"  {name} median: {statistics.median(x_vals):.2f} tok/s")
        except Exception as e:
            log(f"  {name} FAILED during benchmark: {type(e).__name__}: {e}")
            x_vals = None

        log(f"Measuring A (sandwich after {name})...")
        set_state("A_baseline")
        a_vals_next = benchmark_variant(gen_A, args.runs, args.tokens)
        baseline_runs.append(a_vals_next)
        log(f"  A median: {statistics.median(a_vals_next):.2f} tok/s")

        local_baseline_median = (statistics.median(baseline_runs[-2]) + statistics.median(baseline_runs[-1])) / 2.0
        bench_results[name] = {
            "values": x_vals,
            "median": statistics.median(x_vals) if x_vals else None,
            "local_baseline_median": local_baseline_median,
            "speedup_vs_local_baseline": (
                statistics.median(x_vals) / local_baseline_median if x_vals else None
            ),
        }

    set_state("A_baseline")

    # ---------------- Profiling ----------------
    log("\n===== PROFILING =====")
    profiles = {}
    set_state("A_baseline")
    profiles["A_baseline"] = profile_variant(model, gen_A, "A_baseline")

    best_name, best_speedup = None, -1.0
    for name, res in bench_results.items():
        if res["speedup_vs_local_baseline"] and res["speedup_vs_local_baseline"] > best_speedup:
            best_speedup = res["speedup_vs_local_baseline"]
            best_name = name

    if best_name is not None:
        fn_map = dict(candidates)
        set_state(best_name)
        profiles[best_name] = profile_variant(model, fn_map[best_name], best_name)

    set_state("A_baseline")

    # ---------------- Save + summarize ----------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(RESULTS_DIR, f"run_{timestamp}.json")

    payload = {
        "timestamp": timestamp,
        "environment": env,
        "args": vars(args),
        "quantization_report": report,
        "correctness_results": correctness_results,
        "gates": {
            "manual_loop_ok": manual_loop_ok,
            "int8_kernel_ok": int8_kernel_ok,
            "int8_quality_ok": int8_quality_ok,
        },
        "compile_modes": {"D_fp16": compiled_mode_fp16, "F_int8": compiled_mode_int8},
        "baseline_runs": baseline_runs,
        "benchmark_results": bench_results,
        "profiles": profiles,
    }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    log(f"\nRaw results saved to {out_path}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    overall_baseline = statistics.median([v for run in baseline_runs for v in run])
    print(f"Overall baseline (all A measurements): {overall_baseline:.2f} tok/s median")
    print(f"{'variant':<28}{'tok/s':>10}{'speedup':>12}{'2x?':>6}")
    for name, res in bench_results.items():
        if res["median"] is None:
            print(f"{name:<28}{'FAILED':>10}")
            continue
        speedup = res["speedup_vs_local_baseline"]
        flag = "YES" if speedup >= 2.0 else ""
        print(f"{name:<28}{res['median']:>10.2f}{speedup:>11.3f}x{flag:>6}")

    print("\nCorrectness/quality gates:")
    print(f"  manual_loop_ok:   {manual_loop_ok}")
    print(f"  int8_kernel_ok:   {int8_kernel_ok}")
    print(f"  int8_quality_ok:  {int8_quality_ok}")
    print(f"  compile D (fp16): {compiled_mode_fp16}")
    print(f"  compile F (int8): {compiled_mode_int8}")

    any_2x = any(
        res["speedup_vs_local_baseline"] is not None and res["speedup_vs_local_baseline"] >= 2.0
        for res in bench_results.values()
    )
    print(f"\n>=2x end-to-end achieved: {'YES' if any_2x else 'NOT YET'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
