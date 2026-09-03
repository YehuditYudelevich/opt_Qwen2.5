"""
Microbenchmark for the two CPU<->GPU synchronization fixes applied to
decode_loop.greedy_decode_static this batch (see RESULTS.md "Batch 3"):

  1. A persistent cache_position buffer updated via .fill_() instead of
     constructing a fresh torch.tensor([cur_pos], device=...) every
     decode step (allocation + host-to-device copy, every step).
  2. A configurable `eos_check_interval` instead of an unconditional
     `next_token.item()` (a blocking device->host read) every step --
     `qwen_optimizer.optimize()`'s wrapped generate() always resolves a
     real eos_token_id from the model's own config, so this ran on
     every single decode step for every call through the packaged API.

Both were found by applying the PROFILING METHODOLOGY from
https://github.com/vllm-project/vllm/issues/421 (look for small,
frequently-repeated CPU<->GPU round trips that stall the pipeline) to
this project's own decode loop -- not by copying that issue's actual
code, which fixes a hand-written attention CUDA kernel and batches
per-sequence torch.multinomial sampling, neither of which exists here
(we use PyTorch's SDPA for attention and are greedy/batch=1, so those
two specific bugs don't transfer; see RESULTS.md for the full
issue-421-checklist walkthrough, including what was checked and did
NOT apply).

Run:
    python bench_decode_overhead.py

Sections:
  A. Synthetic isolation (no model): raw cost of N
     torch.tensor([i], device=cuda) constructions vs N .fill_() calls
     on a reused buffer; raw cost of N blocking .item() calls vs N
     no-op GPU ops. Quantifies the per-operation overhead in isolation.
  B. Real decode loop, Fix 1 only: a FROZEN, verbatim copy of
     greedy_decode_static as it existed before this batch (embedded
     below, not imported -- decode_loop.py has already moved on) vs
     the current decode_loop.greedy_decode_static, same model/
     StaticCache/compiled-step, eos_token_ids=None in both so the ONLY
     difference is the cache_position construction. Asserts the two
     produce bit-identical output before trusting any timing (this fix
     is supposed to be a pure no-op on behavior).
  C. eos_check_interval sweep (Fix 2), current decode loop only,
     interval in {1, 4, 8, 16}, using the model's real eos_token_id and
     NOT forcing min_new_tokens up to the budget -- i.e. letting natural
     early stopping happen, since that's the behavior being measured.
     Reports wall time AND final sequence length for each interval, so
     the speed/precision tradeoff is visible together, not just speed.

This does not modify decode_loop.py or qwen_optimizer -- it only
benchmarks what's already there, plus one frozen historical copy kept
solely for section B's comparison.
"""

import time

import torch

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------
# Section A: synthetic isolation, no model needed
# ---------------------------------------------------------------------
def bench_tensor_construction_vs_fill(device, n=2000):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(n):
        _ = torch.tensor([i], device=device, dtype=torch.long)
    torch.cuda.synchronize()
    construct_time = time.perf_counter() - t0

    buf = torch.empty(1, device=device, dtype=torch.long)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(n):
        buf.fill_(i)
    torch.cuda.synchronize()
    fill_time = time.perf_counter() - t0

    return construct_time, fill_time


def bench_item_sync_cost(device, n=2000):
    x = torch.zeros(1, device=device, dtype=torch.long)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        _ = x.item()
    torch.cuda.synchronize()
    with_item = time.perf_counter() - t0

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        x.add_(0)
    torch.cuda.synchronize()
    without_item = time.perf_counter() - t0

    return with_item, without_item


def run_section_a(device, n=2000):
    print("\n" + "=" * 70)
    print(f"SECTION A: synthetic isolation, n={n} iterations")
    print("=" * 70)

    construct_time, fill_time = bench_tensor_construction_vs_fill(device, n)
    print(f"torch.tensor([i], device=cuda) x{n}:  {construct_time * 1e6 / n:.2f} us/iter  (total {construct_time:.4f}s)")
    print(f"buf.fill_(i) x{n}:                    {fill_time * 1e6 / n:.2f} us/iter  (total {fill_time:.4f}s)")
    print(f"  -> {construct_time / fill_time:.2f}x" if fill_time > 0 else "  -> n/a")

    with_item, without_item = bench_item_sync_cost(device, n)
    print(f"\nx.item() x{n}:                        {with_item * 1e6 / n:.2f} us/iter  (total {with_item:.4f}s)")
    print(f"x.add_(0) [no sync] x{n}:              {without_item * 1e6 / n:.2f} us/iter  (total {without_item:.4f}s)")
    print(f"  -> {with_item / without_item:.2f}x" if without_item > 0 else "  -> n/a")

    return {
        "construct_time_s": construct_time,
        "fill_time_s": fill_time,
        "item_time_s": with_item,
        "no_sync_time_s": without_item,
    }


# ---------------------------------------------------------------------
# Section B: frozen pre-batch-3 greedy_decode_static, verbatim
# (copied from git history at the commit before this batch's changes --
#  not imported from decode_loop.py, which has already moved on)
# ---------------------------------------------------------------------
@torch.no_grad()
def greedy_decode_static_before_batch3(
    model,
    input_ids,
    max_new_tokens,
    past_key_values,
    decode_step_fn=None,
    eos_token_ids=None,
    min_new_tokens=1,
):
    device = input_ids.device
    prompt_len = input_ids.shape[1]

    cache_position = torch.arange(prompt_len, device=device)
    out = model(
        input_ids=input_ids,
        past_key_values=past_key_values,
        use_cache=True,
        cache_position=cache_position,
        num_logits_to_keep=1,
    )
    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    generated = [input_ids, next_token]
    cur_pos = prompt_len
    generated_count = 1

    def _hit_eos():
        return (
            eos_token_ids is not None
            and generated_count >= min_new_tokens
            and next_token.item() in eos_token_ids
        )

    if not _hit_eos():
        for _ in range(max_new_tokens - 1):
            cache_position = torch.tensor([cur_pos], device=device, dtype=torch.long)

            if decode_step_fn is not None:
                logits = decode_step_fn(next_token, cache_position, past_key_values)
            else:
                out = model(
                    input_ids=next_token,
                    past_key_values=past_key_values,
                    use_cache=True,
                    cache_position=cache_position,
                    num_logits_to_keep=1,
                )
                logits = out.logits

            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated.append(next_token)
            cur_pos += 1
            generated_count += 1

            if _hit_eos():
                break

    return torch.cat(generated, dim=1)


def _time_calls(fn, runs, warmup=2):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples = []
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - t0)
    return samples


def run_section_b(model, input_ids, static_cache, compiled_step, tokens, runs):
    import statistics

    import decode_loop

    print("\n" + "=" * 70)
    print(f"SECTION B: Fix 1 only (cache_position construction), {tokens} tokens, {runs} runs")
    print("(eos_token_ids=None in both -- isolates ONLY the tensor-construction change)")
    print("=" * 70)

    def run_before():
        static_cache.reset()
        return greedy_decode_static_before_batch3(
            model, input_ids, tokens, static_cache, decode_step_fn=compiled_step
        )

    def run_after():
        static_cache.reset()
        return decode_loop.greedy_decode_static(
            model, input_ids, tokens, static_cache, decode_step_fn=compiled_step
        )

    out_before = run_before()
    out_after = run_after()
    identical = torch.equal(out_before, out_after)
    print(f"Output identical (before vs after): {identical}")
    if not identical:
        print("!! MISMATCH -- Fix 1 was supposed to be output-identical. Do not trust timing below; investigate.")

    before_samples = _time_calls(run_before, runs)
    after_samples = _time_calls(run_after, runs)

    before_med = statistics.median(before_samples)
    after_med = statistics.median(after_samples)
    print(f"before (per-step torch.tensor):  {before_med * 1000:.2f} ms/call  (median of {runs})")
    print(f"after  (persistent .fill_()):     {after_med * 1000:.2f} ms/call  (median of {runs})")
    print(f"speedup: {before_med / after_med:.4f}x" if after_med > 0 else "speedup: n/a")

    return {
        "output_identical": identical,
        "before_s": before_samples,
        "after_s": after_samples,
    }


# ---------------------------------------------------------------------
# Section C: eos_check_interval sweep (Fix 2), current decode loop only
# ---------------------------------------------------------------------
def run_section_c(model, tokenizer, input_ids, static_cache, compiled_step, max_tokens, runs, intervals=(1, 4, 8, 16)):
    import statistics

    import decode_loop
    import optimized_model

    print("\n" + "=" * 70)
    print(f"SECTION C: eos_check_interval sweep, up to {max_tokens} tokens, {runs} runs, natural stopping")
    print("=" * 70)

    eos_token_id = getattr(model.config, "eos_token_id", None)
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_ids = set(eos_token_id) if eos_token_id else None
    if eos_token_ids is None:
        print("Model has no eos_token_id configured; cannot exercise early stopping. Skipping section C.")
        return None

    results = {}
    for interval in intervals:
        def run():
            static_cache.reset()
            return decode_loop.greedy_decode_static(
                model,
                input_ids,
                max_tokens,
                static_cache,
                decode_step_fn=compiled_step,
                eos_token_ids=eos_token_ids,
                eos_check_interval=interval,
            )

        out = run()  # warmup + also captures the actual output for this interval
        samples = _time_calls(run, runs)
        med = statistics.median(samples)
        length = out.shape[1] - input_ids.shape[1]
        text = tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)
        print(f"interval={interval:<4} median={med * 1000:8.2f} ms   generated_tokens={length:<4} text={text[:60]!r}")
        results[interval] = {"median_s": med, "generated_tokens": length, "text": text}

    baseline_len = results[1]["generated_tokens"]
    print(f"\nReference (interval=1) generated {baseline_len} tokens before stopping.")
    for interval, r in results.items():
        if interval == 1:
            continue
        overshoot = r["generated_tokens"] - baseline_len
        print(f"interval={interval}: {overshoot:+d} tokens vs reference, {results[1]['median_s'] / r['median_s']:.4f}x time")

    return results


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("bench_decode_overhead.py requires a CUDA GPU.")

    device = "cuda"
    run_section_a(device)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    import decode_loop
    import optimized_model

    log("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="cuda")
    model.eval()
    optimized_model.install_int8_quant(model)
    optimized_model.set_int8(model, True)

    messages = [{"role": "user", "content": "Explain how a CPU works in a few sentences."}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]

    max_cache_len = input_ids.shape[1] + 256
    static_cache = decode_loop.make_static_cache(model, max_cache_len)

    log("Compiling decode step (shared by sections B and C)...")
    compiled_step, mode = decode_loop.compile_decode_step_with_fallback(
        model, static_cache, input_ids, tag="bench_decode_overhead", log=log
    )
    log(f"Compile mode: {mode}")

    run_section_b(model, input_ids, static_cache, compiled_step, tokens=96, runs=5)
    run_section_c(model, tokenizer, input_ids, static_cache, compiled_step, max_tokens=200, runs=5)

    optimized_model.set_int8(model, False)

    print(
        "\nNOTE: sections B/C are kernel/loop-level microbenchmarks. Per this "
        "project's own methodology, confirm any real gain with an end-to-end "
        "run through the actual qwen_optimizer.optimize()-wrapped generate() "
        "(e.g. bench_vs_vllm.py --engine ours) before keeping any change to "
        "the default eos_check_interval."
    )


if __name__ == "__main__":
    main()
