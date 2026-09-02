"""
Manual greedy autoregressive decode, bypassing `model.generate()`.

instructions.md explicitly calls out that HuggingFace `generate()` may
itself be part of the overhead problem (logits processors, stopping
criteria, sampling machinery, repeated Python-side bookkeeping) and asks
for a controlled manual decode loop with the *exact same model semantics*
(greedy, do_sample=False) so its overhead can be measured in isolation
from both the raw forward pass and any kernel-level optimization.

Two cache strategies are supported:

  - DynamicCache: HF's default, grows via torch.cat every step.
  - StaticCache:  preallocated fixed-size buffers, updated via
    index_copy_ at a fixed cache_position. Required for CUDA-graph
    capture (via torch.compile(mode="reduce-overhead")) since graph
    replay needs stable tensor addresses and static shapes.

Both paths call `model(...)` directly -- the real HF forward, with the
real weights (optionally patched by optimized_model.py). No decoder math
is reimplemented here.
"""

import torch
from transformers import DynamicCache, StaticCache


@torch.no_grad()
def greedy_decode_dynamic(model, input_ids, max_new_tokens):
    """
    Manual greedy loop using DynamicCache. Isolates the cost of
    `generate()`'s Python-side machinery from the raw per-token forward
    cost, while keeping the same cache growth behavior as the baseline.
    """
    device = input_ids.device
    past_key_values = DynamicCache()

    prompt_len = input_ids.shape[1]
    cache_position = torch.arange(prompt_len, device=device)

    out = model(
        input_ids=input_ids,
        past_key_values=past_key_values,
        use_cache=True,
        cache_position=cache_position,
        num_logits_to_keep=1,
    )
    past_key_values = out.past_key_values
    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    generated = [input_ids, next_token]
    cur_pos = prompt_len

    for _ in range(max_new_tokens - 1):
        cache_position = torch.tensor([cur_pos], device=device, dtype=torch.long)
        out = model(
            input_ids=next_token,
            past_key_values=past_key_values,
            use_cache=True,
            cache_position=cache_position,
            num_logits_to_keep=1,
        )
        past_key_values = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_token)
        cur_pos += 1

    return torch.cat(generated, dim=1)


def make_static_cache(model, max_cache_len, device=None, dtype=None):
    """
    Builds a StaticCache once. Callers should reuse the SAME instance
    across multiple generations (calling `.reset()` in between) rather
    than constructing a new one per call: torch.compile's
    reduce-overhead/cudagraph mode is keyed off tensor identity/shape, and
    `.reset()` zeroes the buffers in place without breaking their static
    addresses, which a fresh StaticCache() would not preserve.
    """
    return StaticCache(
        config=model.config,
        batch_size=1,
        max_cache_len=max_cache_len,
        device=device or next(model.parameters()).device,
        dtype=dtype or next(model.parameters()).dtype,
    )


@torch.no_grad()
def greedy_decode_static(
    model,
    input_ids,
    max_new_tokens,
    past_key_values,
    decode_step_fn=None,
):
    """
    Manual greedy loop using a caller-provided StaticCache (preallocated,
    fixed-address buffers; call `.reset()` on it before each fresh
    generation). `decode_step_fn`, if given, replaces the per-token
    forward call -- this is the hook used to swap in a torch.compile'd,
    CUDA-graph-capturing callable for the steady-state decode step while
    prefill still runs eagerly (varying prompt length would otherwise
    force recompilation every run).

    decode_step_fn(input_ids_1x1, cache_position_1, past_key_values) -> logits [1,1,V]
    """
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

    return torch.cat(generated, dim=1)


def make_compiled_decode_step(model, fullgraph=True):
    """
    Builds a torch.compile'd single-token decode step suitable for
    `greedy_decode_static`'s `decode_step_fn` hook.

    NOTE: compilation is lazy -- the returned callable must be exercised
    during a warmup phase (outside of timed benchmark iterations) before
    any timing measurement, and the caller is responsible for catching
    failures on that first call and retrying with fullgraph=False.
    """

    def _step(input_ids, cache_position, past_key_values):
        out = model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True,
            cache_position=cache_position,
            num_logits_to_keep=1,
        )
        return out.logits

    compiled = torch.compile(_step, mode="reduce-overhead", fullgraph=fullgraph)
    return compiled
