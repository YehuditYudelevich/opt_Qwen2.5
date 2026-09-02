"""
qwen_optimizer: a thin, reusable wrapper around this project's existing
INT8 + Triton + StaticCache + torch.compile decode path.

    from qwen_optimizer import optimize
    model = AutoModelForCausalLM.from_pretrained(...)
    model = optimize(model)
    output = model.generate(**inputs, do_sample=False, max_new_tokens=64)

This module does not implement any new optimization. It only wires
together the modules already validated elsewhere in this project:

  - optimized_model.py  (INT8 attention/MLP/lm_head wrappers, which in
    turn call the Triton kernels in kernels.py)
  - decode_loop.py       (StaticCache-based manual decode loop, and the
    torch.compile(mode="reduce-overhead") CUDA-graph decode step)

`optimize(model)` patches `model.generate()` so that calls it can
confidently route through that existing fast path do so automatically.
Anything it isn't sure about -- sampling, beam search, batch>1, custom
logits processors, unresolvable generation length, etc. -- transparently
falls back to the original, unmodified `model.generate()` with INT8
disabled, so behavior for unsupported calls is unchanged from a plain
HF model.
"""

import types

import torch

import decode_loop
import optimized_model

_STATE_ATTR = "_qwen_optimizer_state"
_ORIGINAL_GENERATE_ATTR = "_qwen_optimizer_original_generate"

SUPPORTED_MODEL_TYPES = {"qwen2"}

# Anything outside this set on a generate() call is unfamiliar enough
# (custom processors, streaming, assistant/speculative decoding, output
# objects we don't reproduce, ...) that we fall back rather than risk
# silently mishandling it.
_SUPPORTED_KWARGS = {
    "input_ids",
    "inputs",
    "attention_mask",
    "max_new_tokens",
    "min_new_tokens",
    "do_sample",
    "num_beams",
    "num_return_sequences",
    "use_cache",
    "pad_token_id",
    "eos_token_id",
    "generation_config",
}


def is_optimized(model) -> bool:
    """True once `optimize(model)` has installed the fast path on this instance."""
    return hasattr(model, _STATE_ATTR)


def original_generate(model):
    """
    Returns the model's original (pre-`optimize()`) bound `generate`
    method, useful for tests/callers that want to explicitly bypass the
    optimized path for one call. Returns the current `model.generate`
    unchanged if `optimize()` was never called.
    """
    return getattr(model, _ORIGINAL_GENERATE_ATTR, model.generate)


def _validate_architecture(model):
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(
            "qwen_optimizer.optimize() only supports Qwen2-family causal LMs "
            f"(model_type in {sorted(SUPPORTED_MODEL_TYPES)}); got "
            f"model_type={model_type!r}."
        )
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise ValueError(
            "Expected a Qwen2ForCausalLM-shaped model exposing `model.model.layers`."
        )
    if not hasattr(model, "lm_head"):
        raise ValueError("Expected the model to expose `lm_head`.")


def optimize(model, use_compile: bool = True):
    """
    Installs the existing INT8 Triton decode path onto `model` and wraps
    `model.generate()` so supported greedy, batch=1 calls automatically
    use it (StaticCache, optionally with torch.compile CUDA graphs).
    Unsupported calls transparently fall back to the original
    `model.generate()` with INT8 disabled.

    `use_compile=False` skips the torch.compile step (eager StaticCache
    only) -- useful if compilation is flaky in a given environment; the
    INT8/StaticCache speedup still applies.

    Safe to call more than once: later calls are a no-op and return the
    same model instance.
    """
    if is_optimized(model):
        return model

    _validate_architecture(model)

    if not torch.cuda.is_available() or not next(model.parameters()).is_cuda:
        raise RuntimeError(
            "qwen_optimizer.optimize() requires a CUDA model: the Triton "
            "kernels this project relies on are CUDA-only."
        )

    # Reuses the existing installer/toggle from optimized_model.py
    # verbatim -- both are already idempotent and this is the same call
    # test_correctness.py / run_t4_experiments.py make.
    optimized_model.install_int8_quant(model)
    optimized_model.set_int8(model, False)

    state = types.SimpleNamespace(
        static_cache=None,
        compiled_step=None,
        compiled_mode="not attempted",
        max_cache_len=0,
        use_compile=use_compile,
    )

    saved_original_generate = model.generate

    def generate(self, *args, **kwargs):
        return _route_generate(self, state, saved_original_generate, args, kwargs)

    setattr(model, _ORIGINAL_GENERATE_ATTR, saved_original_generate)
    setattr(model, _STATE_ATTR, state)
    model.generate = types.MethodType(generate, model)

    return model


def _resolve_supported_call(model, args, kwargs):
    """
    Conservative allowlist check: returns (True, max_new_tokens) only
    when we are confident the call is plain greedy, batch=1 decoding.
    Any doubt returns (False, None) so the caller falls back safely.
    """
    if len(args) > 1:
        return False, None
    if set(kwargs) - _SUPPORTED_KWARGS:
        return False, None

    input_ids = kwargs.get("input_ids", kwargs.get("inputs"))
    if input_ids is None and args:
        input_ids = args[0]
    if not torch.is_tensor(input_ids) or input_ids.ndim != 2 or input_ids.shape[0] != 1:
        return False, None

    attention_mask = kwargs.get("attention_mask")
    if attention_mask is not None and not bool(attention_mask.bool().all()):
        return False, None  # padding present; the manual loop assumes none

    base_gen_config = kwargs.get("generation_config") or model.generation_config

    def effective(field, default):
        if field in kwargs:
            return kwargs[field]
        value = getattr(base_gen_config, field, None)
        return default if value is None else value

    do_sample = effective("do_sample", False)
    num_beams = effective("num_beams", 1)
    num_return_sequences = effective("num_return_sequences", 1)
    if do_sample or num_beams != 1 or num_return_sequences != 1:
        return False, None

    max_new_tokens = effective("max_new_tokens", None)
    if not max_new_tokens or max_new_tokens <= 0:
        # We don't resolve `max_length`/implicit-length generation; that
        # case just falls back to the real generate().
        return False, None

    return True, int(max_new_tokens)


def _resolve_eos_and_min(model, kwargs):
    base_gen_config = kwargs.get("generation_config") or model.generation_config

    min_new_tokens = kwargs.get("min_new_tokens", getattr(base_gen_config, "min_new_tokens", None))
    min_new_tokens = int(min_new_tokens) if min_new_tokens else 1

    eos_token_id = kwargs.get("eos_token_id", getattr(base_gen_config, "eos_token_id", None))
    if eos_token_id is None:
        eos_token_id = getattr(model.config, "eos_token_id", None)
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_ids = set(eos_token_id) if eos_token_id else None

    return eos_token_ids, min_new_tokens


def _route_generate(model, state, saved_original_generate, args, kwargs):
    supported, max_new_tokens = _resolve_supported_call(model, args, kwargs)

    if not supported:
        optimized_model.set_int8(model, False)
        return saved_original_generate(*args, **kwargs)

    input_ids = kwargs.get("input_ids", kwargs.get("inputs"))
    if input_ids is None and args:
        input_ids = args[0]

    eos_token_ids, min_new_tokens = _resolve_eos_and_min(model, kwargs)

    needed_len = input_ids.shape[1] + max_new_tokens
    if state.static_cache is None or needed_len > state.max_cache_len:
        state.static_cache = decode_loop.make_static_cache(model, needed_len)
        state.max_cache_len = needed_len
        state.compiled_step = None
        state.compiled_mode = "not attempted"

    optimized_model.set_int8(model, True)

    if state.use_compile and state.compiled_step is None:
        state.compiled_step, state.compiled_mode = decode_loop.compile_decode_step_with_fallback(
            model, state.static_cache, input_ids, tag="qwen_optimizer"
        )

    state.static_cache.reset()
    try:
        output = decode_loop.greedy_decode_static(
            model,
            input_ids,
            max_new_tokens,
            state.static_cache,
            decode_step_fn=state.compiled_step,
            eos_token_ids=eos_token_ids,
            min_new_tokens=min_new_tokens,
        )
    finally:
        optimized_model.set_int8(model, False)

    return output
