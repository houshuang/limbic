"""Multi-provider LLM client (Gemini, Anthropic, OpenAI).

Usage:
    from amygdala.llm import generate, generate_structured
    text = await generate("What is the capital of France?")
    result, meta = await generate_structured(prompt="...", schema={...})
"""

import asyncio
import json
import logging
import os
import random
import time

log = logging.getLogger(__name__)

MODELS = {
    "gemini3-flash": {"provider": "gemini", "id": "gemini-3-flash-preview", "input_price": 0.10, "output_price": 0.40},
    "gemini25-flash": {"provider": "gemini", "id": "gemini-2.5-flash", "input_price": 0.15, "output_price": 0.60},
    "gemini25-pro": {"provider": "gemini", "id": "gemini-2.5-pro", "input_price": 1.25, "output_price": 10.0},
    "sonnet": {"provider": "anthropic", "id": "claude-sonnet-4-20250514", "input_price": 3.0, "output_price": 15.0},
    "haiku": {"provider": "anthropic", "id": "claude-haiku-4-5-20251001", "input_price": 1.0, "output_price": 5.0},
    "gpt41-mini": {"provider": "openai", "id": "gpt-4.1-mini", "input_price": 0.40, "output_price": 1.60},
    "gpt41-nano": {"provider": "openai", "id": "gpt-4.1-nano", "input_price": 0.10, "output_price": 0.40},
}
FALLBACK = {"gemini3-flash": "gemini25-flash"}
MAX_RETRIES, BACKOFF_BASE = 3, 2

# Cost logging (optional)
try:
    from limbic.cerebellum.cost_log import cost_log as _cost_log
except ImportError:
    _cost_log = None


def _calc_cost(key, inp, out):
    m = MODELS[key]
    return (inp * m["input_price"] + out * m["output_price"]) / 1_000_000


def _strip_gemini_schema(s):
    if not isinstance(s, dict): return s
    r = {}
    for k, v in s.items():
        if k == "type" and isinstance(v, list):
            r[k] = next((t for t in v if t != "null"), "string")
        elif isinstance(v, dict): r[k] = _strip_gemini_schema(v)
        elif isinstance(v, list): r[k] = [_strip_gemini_schema(i) if isinstance(i, dict) else i for i in v]
        else: r[k] = v
    return r


def _is_retryable(e):
    s = str(e).lower()
    return any(c in s for c in ("429", "500", "503", "resource_exhausted", "rate limit", "overloaded")) or \
           any(c in type(e).__name__.lower() for c in ("timeout", "connection", "transport"))


async def _call_gemini(model_id, sys, user, schema, max_tok, thinking_budget=None):
    from google import genai
    from google.genai import types as gt
    client = genai.Client(api_key=os.environ.get("GEMINI_KEY") or os.environ.get("GOOGLE_API_KEY"))
    t0 = time.time()
    cfg = dict(system_instruction=sys, max_output_tokens=max_tok)
    if schema:
        cfg["response_mime_type"] = "application/json"
        cfg["response_schema"] = _strip_gemini_schema(schema)
    if thinking_budget is not None:
        cfg["thinking_config"] = gt.ThinkingConfig(thinking_budget=thinking_budget)
    r = await client.aio.models.generate_content(model=model_id, contents=user, config=gt.GenerateContentConfig(**cfg))
    return {"text": r.text, "input_tokens": r.usage_metadata.prompt_token_count or 0,
            "output_tokens": r.usage_metadata.candidates_token_count or 0, "duration_s": time.time() - t0}


async def _call_anthropic(model_id, sys, user, schema, max_tok, **kw):
    import anthropic
    client = anthropic.AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_KEY") or os.environ.get("ANTHROPIC_API_KEY"))
    t0 = time.time()
    try:
        msgs = [{"role": "user", "content": user + ("\n\nRespond with valid JSON." if schema else "")}]
        if schema: msgs.append({"role": "assistant", "content": "{"})
        r = await client.messages.create(model=model_id, max_tokens=max_tok, system=sys, messages=msgs)
        text = ("{" + r.content[0].text) if schema else r.content[0].text
        return {"text": text, "input_tokens": r.usage.input_tokens,
                "output_tokens": r.usage.output_tokens, "duration_s": time.time() - t0}
    finally:
        await client.close()


async def _call_openai(model_id, sys, user, schema, max_tok, **kw):
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_KEY") or os.environ.get("OPENAI_API_KEY"))
    t0 = time.time()
    try:
        r = await client.chat.completions.create(
            model=model_id, messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
            response_format={"type": "json_object"} if schema else None, max_completion_tokens=max_tok)
        return {"text": r.choices[0].message.content, "input_tokens": r.usage.prompt_tokens,
                "output_tokens": r.usage.completion_tokens, "duration_s": time.time() - t0}
    finally:
        await client.close()


_PROVIDERS = {"anthropic": _call_anthropic, "gemini": _call_gemini, "openai": _call_openai}


async def _retry_call(fn, args, retries=MAX_RETRIES, **kwargs):
    for attempt in range(retries + 1):
        try:
            return await fn(*args, **kwargs)
        except Exception as e:
            if _is_retryable(e) and attempt < retries:
                await asyncio.sleep(BACKOFF_BASE * (4 ** attempt) + random.uniform(0, 1))
                continue
            raise


async def generate(prompt: str, system_prompt: str = "You are a helpful assistant.",
                   model: str = "gemini3-flash", max_tokens: int = 4096) -> str:
    """Generate text. Returns raw string."""
    m = MODELS[model]
    raw = await _retry_call(_PROVIDERS[m["provider"]], (m["id"], system_prompt, prompt, None, max_tokens))
    return raw["text"]


async def generate_structured(prompt: str, schema: dict, system_prompt: str = "You are a helpful assistant.",
                              model: str = "gemini3-flash", max_tokens: int = 8192,
                              thinking_budget: int | None = None) -> tuple[dict, dict]:
    """Generate structured JSON. Returns (result_dict, metadata with cost)."""
    m = MODELS[model]
    extra = {"thinking_budget": thinking_budget} if thinking_budget is not None and m["provider"] == "gemini" else {}
    t0 = time.time()
    try:
        raw = await _retry_call(_PROVIDERS[m["provider"]], (m["id"], system_prompt, prompt, schema, max_tokens), **extra)
        result = json.loads(raw["text"])
    except json.JSONDecodeError:
        fb = FALLBACK.get(model)
        if not fb: raise
        fm = MODELS[fb]
        raw = await _PROVIDERS[fm["provider"]](fm["id"], system_prompt, prompt, schema, max_tokens)
        result, model = json.loads(raw["text"]), fb
        m = fm
    cost = _calc_cost(model, raw["input_tokens"], raw["output_tokens"])
    if _cost_log:
        try:
            _cost_log.log(project=os.environ.get("COST_LOG_PROJECT", "limbic"),
                          model=f"{m['provider']}/{m['id']}",
                          prompt_tokens=raw["input_tokens"],
                          completion_tokens=raw["output_tokens"],
                          cost_usd=cost)
        except Exception:
            pass
    return result, {"total_cost_usd": cost, "input_tokens": raw["input_tokens"],
                    "output_tokens": raw["output_tokens"], "duration_s": raw["duration_s"],
                    "model": model, "provider": m["provider"]}


def _run_sync(coro):
    """Run an async coroutine synchronously, handling nested event loops."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is None:
        return asyncio.run(coro)
    # Already inside a running loop (Jupyter, ASGI, etc.)
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def generate_sync(prompt: str, **kwargs) -> str:
    return _run_sync(generate(prompt, **kwargs))


def generate_structured_sync(prompt: str, schema: dict, **kwargs) -> tuple[dict, dict]:
    return _run_sync(generate_structured(prompt, schema, **kwargs))
