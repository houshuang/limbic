"""Tests for limbic.amygdala.llm — model registry and per-provider call shaping."""

from __future__ import annotations

import json
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from limbic.amygdala import llm
from limbic.amygdala.llm import FALLBACK, MODELS, generate_structured

SCHEMA = {"type": "object", "properties": {"capital": {"type": "string"}}, "required": ["capital"]}


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_every_entry_is_well_formed(self):
        for key, m in MODELS.items():
            assert m["provider"] in llm._PROVIDERS, f"{key} has unknown provider {m['provider']}"
            assert m["id"], f"{key} has no wire id"
            assert m["input_price"] > 0 and m["output_price"] > 0, f"{key} has non-positive prices"

    def test_current_generation_models_registered(self):
        for key in ("luna", "sol", "terra", "opus", "sonnet", "haiku", "gemini38-flash"):
            assert key in MODELS

    def test_gpt56_tiers_map_to_wire_ids(self):
        assert MODELS["luna"]["id"] == "gpt-5.6-luna"
        assert MODELS["sol"]["id"] == "gpt-5.6-sol"
        assert MODELS["terra"]["id"] == "gpt-5.6-terra"

    def test_fallbacks_point_at_registered_models(self):
        for src, dst in FALLBACK.items():
            assert src in MODELS, f"fallback source {src} is not registered"
            assert dst in MODELS, f"fallback target {dst} is not registered"

    def test_cost_uses_registry_prices(self):
        # luna: $0.20/M in, $1.20/M out
        assert llm._calc_cost("luna", 1_000_000, 1_000_000) == pytest.approx(1.40)


# ---------------------------------------------------------------------------
# OpenAI call shaping
# ---------------------------------------------------------------------------


def _openai_response(content: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )


class TestOpenAICallShape:
    """json_object mode is rejected by the API unless the messages mention JSON."""

    def setup_method(self):
        pytest.importorskip("openai")

    def test_structured_call_mentions_json_and_schema(self):
        create = AsyncMock(return_value=_openai_response('{"capital": "Paris"}'))
        with patch("openai.AsyncOpenAI") as mk:
            mk.return_value = SimpleNamespace(
                chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
                close=AsyncMock(),
            )
            _run(llm._call_openai("gpt-5.6-luna", "terse", "Capital of France?", SCHEMA, 2000))

        kwargs = create.await_args.kwargs
        assert kwargs["response_format"] == {"type": "json_object"}
        system = kwargs["messages"][0]["content"]
        assert "json" in system.lower()
        assert json.dumps(SCHEMA) in system

    def test_unstructured_call_leaves_system_prompt_alone(self):
        create = AsyncMock(return_value=_openai_response("Paris"))
        with patch("openai.AsyncOpenAI") as mk:
            mk.return_value = SimpleNamespace(
                chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
                close=AsyncMock(),
            )
            _run(llm._call_openai("gpt-5.6-luna", "terse", "Capital of France?", None, 2000))

        kwargs = create.await_args.kwargs
        assert kwargs["response_format"] is None
        assert kwargs["messages"][0]["content"] == "terse"


# ---------------------------------------------------------------------------
# Gemini token accounting
# ---------------------------------------------------------------------------


class TestGeminiTokenAccounting:
    """Gemini bills thinking tokens as output but reports them in a separate counter."""

    def setup_method(self):
        pytest.importorskip("google.genai")

    def test_thinking_tokens_counted_as_output(self):
        response = SimpleNamespace(
            text='{"capital": "Paris"}',
            usage_metadata=SimpleNamespace(
                prompt_token_count=8, candidates_token_count=5, thoughts_token_count=136
            ),
        )
        with patch("google.genai.Client") as mk:
            mk.return_value = SimpleNamespace(
                aio=SimpleNamespace(
                    models=SimpleNamespace(generate_content=AsyncMock(return_value=response))
                )
            )
            raw = _run(llm._call_gemini("gemini-3.8-flash", "terse", "Capital of France?", SCHEMA, 2000))

        assert raw["output_tokens"] == 141

    def test_missing_thought_counter_is_tolerated(self):
        response = SimpleNamespace(
            text='{"capital": "Paris"}',
            usage_metadata=SimpleNamespace(
                prompt_token_count=8, candidates_token_count=11, thoughts_token_count=None
            ),
        )
        with patch("google.genai.Client") as mk:
            mk.return_value = SimpleNamespace(
                aio=SimpleNamespace(
                    models=SimpleNamespace(generate_content=AsyncMock(return_value=response))
                )
            )
            raw = _run(llm._call_gemini("gemini-3.1-flash-lite", "terse", "x", SCHEMA, 2000))

        assert raw["output_tokens"] == 11


# ---------------------------------------------------------------------------
# Fallback on unparseable JSON
# ---------------------------------------------------------------------------


class TestFallback:
    def test_empty_response_falls_back_to_configured_model(self):
        """A reasoning model that spends its whole budget thinking returns ''."""
        calls = []

        async def fake_openai(model_id, sys, user, schema, max_tok, **kw):
            calls.append(model_id)
            text = "" if model_id == "gpt-5.6-luna" else '{"capital": "Paris"}'
            return {"text": text, "input_tokens": 10, "output_tokens": 5, "duration_s": 0.1}

        with patch.dict(llm._PROVIDERS, {"openai": fake_openai}):
            result, meta = _run(generate_structured("Capital?", SCHEMA, model="luna"))

        assert result == {"capital": "Paris"}
        assert calls == ["gpt-5.6-luna", "gpt-5.6-terra"]
        assert meta["model"] == "terra"
