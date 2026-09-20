"""Tests for the built-in `openai` / `gemini` HTTP transports in calls.py.

All tests monkeypatch `calls._http_post_json` — no real network call is ever
made, and no SDK (`openai`, `google-genai`) is imported.
"""

from __future__ import annotations

import json

import pytest

from limbic.cerebellum import calls, claude_cli
from limbic.cerebellum.calls import TransportError, cached_call
from limbic.cerebellum.cost_log import CostLog, UnknownModelPriceError, price_for


@pytest.fixture
def tmp_cost_log(tmp_path, monkeypatch):
    fresh = CostLog(db_path=tmp_path / "costs.db")
    monkeypatch.setattr(calls, "cost_log", fresh)
    monkeypatch.setattr(claude_cli, "cost_log", fresh)
    return fresh


@pytest.fixture
def cache_db(tmp_path):
    return tmp_path / "cache.db"


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Belt and suspenders: any real HTTP attempt fails loudly instead of
    hitting the network, even if a test forgets to patch `_http_post_json`."""
    import urllib.request

    def _boom(*a, **kw):
        raise AssertionError("real network call attempted in test")

    monkeypatch.setattr(urllib.request, "urlopen", _boom)


# ---------------------------------------------------------------------------
# OpenAI Responses API transport
# ---------------------------------------------------------------------------


def _openai_response(text=None, structured=None, input_tokens=100, cached_tokens=0, output_tokens=20):
    resp = {
        "id": "resp_abc123",
        "usage": {
            "input_tokens": input_tokens,
            "input_tokens_details": {"cached_tokens": cached_tokens},
            "output_tokens": output_tokens,
        },
    }
    if structured is not None:
        resp["output"] = [{"type": "message", "content": [
            {"type": "output_text", "text": json.dumps(structured)},
        ]}]
    else:
        resp["output"] = [{"type": "message", "content": [{"type": "output_text", "text": text}]}]
    return resp


class TestOpenAITransport:
    def test_plain_text(self, tmp_cost_log, cache_db, monkeypatch):
        monkeypatch.setenv("OPENAI_KEY", "sk-test")
        captured = {}

        def _fake_post(url, payload, *, headers, timeout):
            captured["url"] = url
            captured["payload"] = payload
            captured["headers"] = headers
            return _openai_response(text="hello there")

        monkeypatch.setattr(calls, "_http_post_json", _fake_post)
        result, meta = cached_call(
            "say hi", project="p", purpose="greet", transport="openai",
            model="gpt-5.4-mini", cache_db_path=cache_db,
        )
        assert result == "hello there"
        assert captured["url"] == "https://api.openai.com/v1/responses"
        assert captured["headers"]["Authorization"] == "Bearer sk-test"
        assert captured["payload"]["model"] == "gpt-5.4-mini"
        assert "text" not in captured["payload"]  # no schema -> no json_schema format

        rows = tmp_cost_log.query()
        assert len(rows) == 1
        assert rows[0]["script"] == "cached_call.openai"
        assert rows[0]["prompt_tokens"] == 100
        assert meta.call_id == rows[0]["id"]

        # response_id lands in both the ledger row and meta.raw, so a lost
        # result can be fetched back by id instead of re-bought.
        assert json.loads(rows[0]["metadata"])["response_id"] == "resp_abc123"
        assert meta.raw["response_id"] == "resp_abc123"

    def test_response_id_also_lands_in_the_cache_entry(self, tmp_cost_log, cache_db, monkeypatch):
        monkeypatch.setenv("OPENAI_KEY", "sk-test")
        monkeypatch.setattr(calls, "_http_post_json", lambda *a, **kw: _openai_response(text="hello"))
        cached_call("say hi", project="p", purpose="greet", transport="openai", cache_db_path=cache_db)

        from limbic.amygdala import connect
        conn = connect(str(cache_db))
        row = conn.execute("SELECT meta_json FROM call_cache").fetchone()
        conn.close()
        assert json.loads(row["meta_json"])["response_id"] == "resp_abc123"

    def test_structured_output_with_schema(self, tmp_cost_log, cache_db, monkeypatch):
        monkeypatch.setenv("OPENAI_KEY", "sk-test")
        captured = {}
        schema = {"type": "object", "properties": {"label": {"type": "string"}}}

        def _fake_post(url, payload, *, headers, timeout):
            captured["payload"] = payload
            return _openai_response(structured={"label": "positive"})

        monkeypatch.setattr(calls, "_http_post_json", _fake_post)
        result, meta = cached_call(
            "classify", project="p", purpose="sentiment", transport="openai",
            schema=schema, cache_db_path=cache_db,
        )
        assert result == {"label": "positive"}
        assert captured["payload"]["text"]["format"]["type"] == "json_schema"
        assert captured["payload"]["text"]["format"]["strict"] is True
        assert captured["payload"]["text"]["format"]["schema"] == schema

    def test_system_prompt_becomes_instructions(self, tmp_cost_log, cache_db, monkeypatch):
        monkeypatch.setenv("OPENAI_KEY", "sk-test")
        captured = {}

        def _fake_post(url, payload, *, headers, timeout):
            captured["payload"] = payload
            return _openai_response(text="ok")

        monkeypatch.setattr(calls, "_http_post_json", _fake_post)
        cached_call("hi", project="p", purpose="x", transport="openai",
                    system="You are terse.", cache_db_path=cache_db)
        assert captured["payload"]["instructions"] == "You are terse."

    def test_missing_api_key_raises(self, tmp_cost_log, cache_db, monkeypatch):
        monkeypatch.delenv("OPENAI_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(TransportError, match="OPENAI_KEY"):
            cached_call("hi", project="p", purpose="x", transport="openai", cache_db_path=cache_db)

    def test_http_error_logs_a_failure_row_and_raises(self, tmp_cost_log, cache_db, monkeypatch):
        monkeypatch.setenv("OPENAI_KEY", "sk-test")

        def _fake_post(url, payload, *, headers, timeout):
            raise TransportError("HTTP 429: rate limited")

        monkeypatch.setattr(calls, "_http_post_json", _fake_post)
        with pytest.raises(TransportError, match="rate limited"):
            cached_call("hi", project="p", purpose="x", transport="openai", cache_db_path=cache_db)

        rows = tmp_cost_log.query()
        assert len(rows) == 1
        assert rows[0]["cost_usd"] == 0
        meta = json.loads(rows[0]["metadata"])
        assert meta["failed"] is True

    def test_cost_computed_via_price_for(self, tmp_cost_log, cache_db, monkeypatch):
        monkeypatch.setenv("OPENAI_KEY", "sk-test")
        monkeypatch.setattr(
            calls, "_http_post_json",
            lambda *a, **kw: _openai_response(text="x", input_tokens=1_000_000, output_tokens=1_000_000),
        )
        _, meta = cached_call(
            "hi", project="p", purpose="x", transport="openai",
            model="gpt-5.4-mini", cache_db_path=cache_db,
        )
        inp, out = price_for("gpt-5.4-mini")
        assert meta.cost_usd == pytest.approx(inp + out)

    def test_unpriced_model_does_not_fail_the_call(self, tmp_cost_log, cache_db, monkeypatch):
        monkeypatch.setenv("OPENAI_KEY", "sk-test")
        monkeypatch.setattr(calls, "_http_post_json", lambda *a, **kw: _openai_response(text="ok"))
        with pytest.raises(UnknownModelPriceError):
            price_for("no-such-model-xyz")  # sanity: strict=True would raise
        result, meta = cached_call(
            "hi", project="p", purpose="x", transport="openai",
            model="no-such-model-xyz", cache_db_path=cache_db,
        )
        assert result == "ok"
        assert meta.cost_usd == 0.0


# ---------------------------------------------------------------------------
# Gemini REST transport
# ---------------------------------------------------------------------------


def _gemini_response(text=None, prompt_tokens=50, candidates_tokens=10, cached_tokens=0, response_id="gresp_1"):
    return {
        "candidates": [{"content": {"parts": [{"text": text}]}}],
        "usageMetadata": {
            "promptTokenCount": prompt_tokens,
            "candidatesTokenCount": candidates_tokens,
            "cachedContentTokenCount": cached_tokens,
        },
        "responseId": response_id,
    }


class TestGeminiTransport:
    def test_plain_text(self, tmp_cost_log, cache_db, monkeypatch):
        monkeypatch.setenv("GEMINI_KEY", "gk-test")
        captured = {}

        def _fake_post(url, payload, *, headers, timeout):
            captured["url"] = url
            captured["payload"] = payload
            return _gemini_response(text="hello")

        monkeypatch.setattr(calls, "_http_post_json", _fake_post)
        result, meta = cached_call(
            "say hi", project="p", purpose="greet", transport="gemini",
            model="gemini-2.5-flash", cache_db_path=cache_db,
        )
        assert result == "hello"
        assert "gemini-2.5-flash:generateContent" in captured["url"]
        assert "key=gk-test" in captured["url"]

        rows = tmp_cost_log.query()
        assert len(rows) == 1
        assert rows[0]["script"] == "cached_call.gemini"
        assert meta.call_id == rows[0]["id"]
        assert json.loads(rows[0]["metadata"])["response_id"] == "gresp_1"
        assert meta.raw["response_id"] == "gresp_1"

    def test_schema_strips_null_union_and_sets_json_mime(self, tmp_cost_log, cache_db, monkeypatch):
        monkeypatch.setenv("GEMINI_KEY", "gk-test")
        captured = {}
        schema = {"type": "object", "properties": {"label": {"type": ["string", "null"]}}}

        def _fake_post(url, payload, *, headers, timeout):
            captured["payload"] = payload
            return _gemini_response(text='{"label": "pos"}')

        monkeypatch.setattr(calls, "_http_post_json", _fake_post)
        result, meta = cached_call(
            "classify", project="p", purpose="sentiment", transport="gemini",
            schema=schema, cache_db_path=cache_db,
        )
        assert result == {"label": "pos"}
        gen_cfg = captured["payload"]["generationConfig"]
        assert gen_cfg["responseMimeType"] == "application/json"
        assert gen_cfg["responseSchema"]["properties"]["label"]["type"] == "string"

    def test_system_prompt_becomes_system_instruction(self, tmp_cost_log, cache_db, monkeypatch):
        monkeypatch.setenv("GEMINI_KEY", "gk-test")
        captured = {}

        def _fake_post(url, payload, *, headers, timeout):
            captured["payload"] = payload
            return _gemini_response(text="ok")

        monkeypatch.setattr(calls, "_http_post_json", _fake_post)
        cached_call("hi", project="p", purpose="x", transport="gemini",
                    system="Be terse.", cache_db_path=cache_db)
        assert captured["payload"]["systemInstruction"]["parts"][0]["text"] == "Be terse."

    def test_missing_api_key_raises(self, tmp_cost_log, cache_db, monkeypatch):
        for var in ("GEMINI_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
            monkeypatch.delenv(var, raising=False)
        with pytest.raises(TransportError, match="GEMINI_KEY"):
            cached_call("hi", project="p", purpose="x", transport="gemini", cache_db_path=cache_db)

    def test_malformed_payload_raises_transport_error(self, tmp_cost_log, cache_db, monkeypatch):
        monkeypatch.setenv("GEMINI_KEY", "gk-test")
        monkeypatch.setattr(calls, "_http_post_json", lambda *a, **kw: {"candidates": []})
        with pytest.raises(TransportError, match="no text in Gemini payload"):
            cached_call("hi", project="p", purpose="x", transport="gemini", cache_db_path=cache_db)

    def test_calls_module_has_no_sdk_dependency(self):
        """Static check for the design goal: calls.py must not import
        `google.genai` or `openai` at all — the whole point of the built-in
        transports is stdlib `urllib` only, avoiding the arm64 google-genai
        import gotcha and an unnecessary dependency."""
        import ast

        import limbic.cerebellum.calls as calls_module

        tree = ast.parse(open(calls_module.__file__).read())
        imported_roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".")[0])
        assert "google" not in imported_roots
        assert "openai" not in imported_roots

    def test_second_identical_gemini_call_is_a_cache_hit(self, tmp_cost_log, cache_db, monkeypatch):
        calls_made = {"n": 0}

        def _fake_post(url, payload, *, headers, timeout):
            calls_made["n"] += 1
            return _gemini_response(text="cached-answer")

        monkeypatch.setenv("GEMINI_KEY", "gk-test")
        monkeypatch.setattr(calls, "_http_post_json", _fake_post)
        cached_call("x", project="p", purpose="y", transport="gemini", cache_db_path=cache_db)
        result, meta = cached_call("x", project="p", purpose="y", transport="gemini", cache_db_path=cache_db)
        assert result == "cached-answer"
        assert meta.cache_hit is True
        assert calls_made["n"] == 1
