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
        # OpenAI's strict mode rejects this schema as written; the transport closes it.
        assert captured["payload"]["text"]["format"]["schema"] == {
            "type": "object", "properties": {"label": {"type": ["string", "null"]}},
            "additionalProperties": False, "required": ["label"]}

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


# ---------------------------------------------------------------------------
# Raw-request passthrough
# ---------------------------------------------------------------------------


def _skard_style_request():
    return {
        "model": "gpt-5.4-mini",
        "reasoning": {"effort": "low"},
        "max_output_tokens": 6000,
        "prompt_cache_key": "v7:0123456789abcdef",
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": "Kodebok: læreplan — «ånd»"}]},
            {"role": "user", "content": [{"type": "input_text", "text": "side 12"}]},
        ],
        "text": {"format": {
            "type": "json_schema", "name": "skard_packet_result", "strict": True,
            "schema": {"type": "object", "properties": {"items": {"type": "array"}},
                       "required": ["items"], "additionalProperties": False},
        }},
    }


class _FakeHTTPResponse:
    def __init__(self, payload):
        self._body = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class TestRawRequestPassthrough:
    def _capture_urlopen(self, monkeypatch, response):
        import urllib.request

        posted = []

        def fake_urlopen(req, timeout=None, context=None):
            posted.append(req)
            return _FakeHTTPResponse(response)

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        return posted

    def test_bytes_in_equal_bytes_posted(self, tmp_cost_log, cache_db, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        response = _openai_response(structured={"items": []}, input_tokens=5000, cached_tokens=4096)
        posted = self._capture_urlopen(monkeypatch, response)
        body = calls.canonical_bytes(_skard_style_request())

        result, meta = cached_call(
            request=body, project="skard", purpose="packet-coding:v7",
            transport="openai", cache_db_path=cache_db,
        )

        assert len(posted) == 1
        assert posted[0].data == body
        assert posted[0].full_url == "https://api.openai.com/v1/responses"
        assert result == response
        assert meta.request_sha256 == calls.hashlib.sha256(body).hexdigest()
        assert meta.model == "gpt-5.4-mini"
        assert meta.raw["cached_tokens"] == 4096
        assert meta.raw["response_id"] == "resp_abc123"

    def test_dict_request_is_posted_as_canonical_bytes(self, tmp_cost_log, cache_db, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        posted = self._capture_urlopen(monkeypatch, _openai_response(structured={"items": []}))
        request = _skard_style_request()

        cached_call(request=request, project="skard", purpose="p", transport="openai", cache_db_path=cache_db)

        expected = json.dumps(request, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        assert posted[0].data == expected
        assert "«ånd»".encode("utf-8") in posted[0].data

    def test_ledger_row_and_cache_hit(self, tmp_cost_log, cache_db, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        response = _openai_response(structured={"items": []}, input_tokens=5000, cached_tokens=4096)
        posted = self._capture_urlopen(monkeypatch, response)
        kwargs = dict(
            request=_skard_style_request(), project="skard", purpose="packet-coding:v7",
            transport="openai", cache_db_path=cache_db, packet_id="src-1:p003",
            ledger_metadata={"source_id": "src-1", "replicate": 0},
        )

        _, first = cached_call(**kwargs)
        again, second = cached_call(**kwargs)

        assert len(posted) == 1
        assert again == response
        assert second.cache_hit and second.cache_key == first.cache_key
        rows = {r["id"]: r for r in tmp_cost_log.query()}
        billed = rows[first.call_id]
        assert billed["purpose"] == "packet-coding:v7"
        assert billed["packet_id"] == "src-1:p003"
        assert (billed["prompt_tokens"], billed["cached_tokens"]) == (5000, 4096)
        assert billed["cost_usd"] == pytest.approx((904 * 0.75 + 4096 * 0.075 + 20 * 4.50) / 1_000_000)
        billed_meta = json.loads(billed["metadata"])
        assert billed_meta["response_id"] == "resp_abc123"
        assert billed_meta["source_id"] == "src-1"
        hit = rows[second.call_id]
        assert hit["cache_hit"] == 1 and hit["cost_usd"] == 0.0
        assert hit["packet_id"] == "src-1:p003"

    def test_caller_supplied_cache_key(self, tmp_cost_log, cache_db, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        posted = self._capture_urlopen(monkeypatch, _openai_response(structured={"items": []}))
        common = dict(request=_skard_style_request(), project="skard", purpose="p",
                      transport="openai", cache_db_path=cache_db)

        _, r0 = cached_call(cache_key="input-sha:model:v7:0", **common)
        _, r1 = cached_call(cache_key="input-sha:model:v7:1", **common)
        _, r0_again = cached_call(cache_key="input-sha:model:v7:0", **common)

        assert len(posted) == 2
        assert (r0.cache_key, r1.cache_key) == ("input-sha:model:v7:0", "input-sha:model:v7:1")
        assert r0_again.cache_hit

    def test_failure_row_carries_error_outcome(self, tmp_cost_log, cache_db, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        def boom(url, body, *, headers, timeout):
            raise TransportError("HTTP 500: upstream")

        monkeypatch.setattr(calls, "_http_post_bytes", boom)
        with pytest.raises(TransportError):
            cached_call(request=_skard_style_request(), project="skard", purpose="p",
                        transport="openai", cache_db_path=cache_db, packet_id="src-1:p003")

        (row,) = tmp_cost_log.query()
        assert (row["outcome"], row["packet_id"], row["purpose"]) == ("error", "src-1:p003", "p")

    def test_truncated_response_is_returned_not_raised(self, tmp_cost_log, cache_db, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        response = {"id": "resp_cut", "status": "incomplete", "output": [],
                    "usage": {"input_tokens": 10, "output_tokens": 6000}}
        self._capture_urlopen(monkeypatch, response)

        result, meta = cached_call(request=_skard_style_request(), project="skard", purpose="p",
                                   transport="openai", cache=False)

        assert result["status"] == "incomplete"
        assert tmp_cost_log.query()[0]["completion_tokens"] == 6000

    def test_request_excludes_prompt_and_replicates(self, tmp_cost_log, cache_db):
        with pytest.raises(ValueError, match="replaces prompt"):
            cached_call("hi", request={"model": "m"}, project="p", purpose="p", transport="openai")
        with pytest.raises(ValueError, match="replicates"):
            cached_call(request={"model": "m"}, project="p", purpose="p", transport="openai", replicates=2)
        with pytest.raises(ValueError, match="prompt is required"):
            cached_call(project="p", purpose="p", transport="openai")

    def test_gemini_posts_request_bytes(self, tmp_cost_log, cache_db, monkeypatch):
        monkeypatch.setenv("GEMINI_KEY", "g-test")
        response = {"responseId": "g1", "candidates": [{"content": {"parts": [{"text": "ok"}]}}],
                    "usageMetadata": {"promptTokenCount": 9, "candidatesTokenCount": 2,
                                      "cachedContentTokenCount": 4}}
        posted = self._capture_urlopen(monkeypatch, response)
        body = calls.canonical_bytes({"contents": [{"role": "user", "parts": [{"text": "hei"}]}]})

        result, meta = cached_call(request=body, model="gemini-2.5-flash", project="kb", purpose="p",
                                   transport="gemini", cache_db_path=cache_db)

        assert posted[0].data == body
        assert "gemini-2.5-flash:generateContent" in posted[0].full_url
        assert result == response and meta.raw["cached_tokens"] == 4


class TestStrictOpenAISchema:
    def test_closes_objects_and_makes_optional_fields_nullable(self):
        from limbic.cerebellum.calls import _strict_openai_schema
        schema = {"type": "object", "properties": {
            "decisions": {"type": "array", "items": {"type": "object", "properties": {
                "slot": {"type": "string", "enum": ["i01"]},
                "note": {"type": "string"}}, "required": ["slot"]}},
            "label": {"type": "string", "enum": ["a", "b"]}},
            "required": ["decisions"]}
        strict = _strict_openai_schema(schema)
        assert strict["additionalProperties"] is False
        assert strict["required"] == ["decisions", "label"]
        assert strict["properties"]["label"]["type"] == ["string", "null"]
        assert strict["properties"]["label"]["enum"] == ["a", "b", None]
        item = strict["properties"]["decisions"]["items"]
        assert item["additionalProperties"] is False
        assert item["required"] == ["slot", "note"]
        assert item["properties"]["slot"]["type"] == "string"
        assert item["properties"]["note"]["type"] == ["string", "null"]
        assert "additionalProperties" not in schema
        assert schema["properties"]["label"]["type"] == "string"


class TestReasoningEffort:
    def test_effort_is_sent_and_keyed_separately(self, tmp_cost_log, cache_db, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test")
        sent = []

        def fake_post(url, payload, headers=None, timeout=None):
            sent.append(payload)
            return {"output": [{"type": "message", "content": [{"type": "output_text", "text": "hi"}]}],
                    "usage": {"input_tokens": 1, "output_tokens": 1}}

        monkeypatch.setattr(calls, "_http_post_json", fake_post)
        for effort in (None, "low", "low"):
            calls.cached_call("same prompt", project="p", purpose="t", transport="openai",
                              model="gpt-6-luna", cache_db_path=cache_db,
                              **({"reasoning_effort": effort} if effort else {}))
        assert len(sent) == 2
        assert "reasoning" not in sent[0]
        assert sent[1]["reasoning"] == {"effort": "low"}


# ---------------------------------------------------------------------------
# Image input
# ---------------------------------------------------------------------------

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
JPEG = b"\xff\xd8\xff\xe0" + b"\x01" * 16


class TestImages:
    def test_openai_sends_input_image_parts(self, tmp_cost_log, cache_db, monkeypatch):
        monkeypatch.setenv("OPENAI_KEY", "sk-test")
        captured = {}

        def _fake_post(url, payload, *, headers, timeout):
            captured["payload"] = payload
            return _openai_response(text="Norsk tekst")

        monkeypatch.setattr(calls, "_http_post_json", _fake_post)
        result, _ = cached_call("transcribe", project="p", purpose="ocr", transport="openai",
                                model="gpt-5.6-luna", images=[PNG], cache_db_path=cache_db)
        assert result == "Norsk tekst"
        content = captured["payload"]["input"][0]["content"]
        assert content[0] == {"type": "input_text", "text": "transcribe"}
        assert content[1]["type"] == "input_image"
        assert content[1]["image_url"].startswith("data:image/png;base64,")

    def test_gemini_sends_inline_data(self, tmp_cost_log, cache_db, monkeypatch):
        monkeypatch.setenv("GEMINI_KEY", "g-test")
        captured = {}

        def _fake_post(url, payload, *, headers, timeout):
            captured["payload"] = payload
            return {"candidates": [{"content": {"parts": [{"text": "ok"}]}}],
                    "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 2}}

        monkeypatch.setattr(calls, "_http_post_json", _fake_post)
        cached_call("read", project="p", purpose="ocr", transport="gemini", model="gemini-2.5-pro",
                    images=[("image/webp", b"x"), JPEG], cache_db_path=cache_db)
        parts = captured["payload"]["contents"][0]["parts"]
        assert parts[0] == {"text": "read"}
        assert [p["inline_data"]["mime_type"] for p in parts[1:]] == ["image/webp", "image/jpeg"]

    def test_images_are_part_of_the_cache_key(self, tmp_cost_log, cache_db, monkeypatch):
        monkeypatch.setenv("OPENAI_KEY", "sk-test")
        calls_made = []

        def _fake_post(url, payload, *, headers, timeout):
            calls_made.append(payload)
            return _openai_response(text=f"page {len(calls_made)}")

        monkeypatch.setattr(calls, "_http_post_json", _fake_post)
        kw = dict(project="p", purpose="ocr", transport="openai", model="gpt-5.6-luna", cache_db_path=cache_db)
        a, _ = cached_call("transcribe", images=[PNG], **kw)
        b, _ = cached_call("transcribe", images=[JPEG], **kw)
        a2, meta = cached_call("transcribe", images=[PNG], **kw)
        assert (a, b, a2) == ("page 1", "page 2", "page 1")
        assert meta.cache_hit and len(calls_made) == 2

    def test_text_only_cache_key_is_unchanged(self):
        before = calls._cache_key(model="m", system="s", prompt="p", schema=None, version="")
        assert before == calls._cache_key(model="m", system="s", prompt="p", schema=None, version="", images_sha="")

    def test_rejects_images_on_claude_cli_and_unknown_bytes(self, cache_db):
        with pytest.raises(ValueError, match="claude_cli"):
            cached_call("x", project="p", purpose="ocr", images=[PNG], cache_db_path=cache_db)
        with pytest.raises(ValueError, match="image type"):
            cached_call("x", project="p", purpose="ocr", transport="openai", images=[b"nope"],
                        cache_db_path=cache_db)


def test_gemini_thinking_tokens_are_billed_as_output(tmp_cost_log, cache_db, monkeypatch):
    monkeypatch.setenv("GEMINI_KEY", "g-test")
    monkeypatch.setattr(calls, "_http_post_json", lambda url, payload, *, headers, timeout: {
        "candidates": [{"content": {"parts": [{"text": "ok"}]}}],
        "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 20, "thoughtsTokenCount": 300}})
    _, meta = cached_call("x", project="p", purpose="t", transport="gemini", model="gemini-2.5-flash",
                          cache_db_path=cache_db)
    assert meta.raw["output_tokens"] == 320
