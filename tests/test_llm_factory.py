import pytest
from pydantic import BaseModel, ConfigDict, Field
from types import SimpleNamespace
from typing import Optional

from confirm.llm import (
    AnthropicClient,
    GoogleClient,
    OpenAIClient,
    OpenRouterClient,
    StandInClient,
    _create_chat_completion_with_param_fallback,
    _openai_strict_json_schema,
    make_llm,
)


def test_make_llm_parses_provider_model_specs():
    openai = make_llm("openai:gpt-5-mini")
    anthropic = make_llm("anthropic:claude-haiku-4-5")
    openrouter = make_llm("openrouter:deepseek/deepseek-chat")
    google = make_llm("google:gemini-3.5-flash")
    standin = make_llm("standin")

    assert isinstance(openai, OpenAIClient)
    assert openai.model == "gpt-5-mini"
    assert isinstance(anthropic, AnthropicClient)
    assert anthropic.model == "claude-haiku-4-5"
    assert isinstance(openrouter, OpenRouterClient)
    assert openrouter.model == "deepseek/deepseek-chat"
    assert isinstance(google, GoogleClient)
    assert google.model == "gemini-3.5-flash"
    assert isinstance(standin, StandInClient)


def test_make_llm_rejects_unknown_provider():
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        make_llm("unknown:model")


def test_openai_compatible_param_fallback_drops_brittle_params():
    calls = []

    def create(**kwargs):
        calls.append(dict(kwargs))
        if len(calls) == 1:
            raise RuntimeError("Unsupported parameter: temperature")
        if len(calls) == 2:
            raise RuntimeError("Unsupported parameter: max_tokens")
        return "ok"

    result = _create_chat_completion_with_param_fallback(
        create,
        model="m",
        messages=[],
        temperature=0,
        max_tokens=2048,
    )

    assert result == "ok"
    assert "temperature" in calls[0]
    assert "max_tokens" in calls[0]
    assert "temperature" not in calls[1]
    assert "max_tokens" in calls[1]
    assert "temperature" not in calls[2]
    assert "max_tokens" not in calls[2]


def test_openai_strict_json_schema_requires_defaulted_fields():
    class NestedPayload(BaseModel):
        model_config = ConfigDict(extra="forbid")

        name: str
        changed_fields: list[str] = Field(default_factory=list)

    class ResponsePayload(BaseModel):
        model_config = ConfigDict(extra="forbid")

        nested: NestedPayload
        optional_text: Optional[str] = None

    schema = _openai_strict_json_schema(ResponsePayload)

    assert set(schema["required"]) == {"nested", "optional_text"}
    assert schema["additionalProperties"] is False
    nested_schema = schema["$defs"]["NestedPayload"]
    assert set(nested_schema["required"]) == {"name", "changed_fields"}
    assert nested_schema["additionalProperties"] is False
    assert "default" not in schema["properties"]["optional_text"]
    assert "default" not in nested_schema["properties"]["changed_fields"]


def test_openrouter_structured_output_requires_supported_route(monkeypatch):
    class Payload(BaseModel):
        value: str
        items: list[str] = Field(max_length=3)

    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            id="response-1",
            model="anthropic/claude-opus-4.8",
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=4, total_tokens=14),
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"value":"ok","items":[]}'))],
        )

    client = OpenRouterClient("anthropic/claude-opus-4.8")
    monkeypatch.setattr(
        client,
        "_client",
        lambda: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
    )
    raw = client.complete_structured("system", "user", Payload)
    assert Payload.model_validate_json(raw).value == "ok"
    assert captured["response_format"]["type"] == "json_schema"
    assert "maxItems" not in str(captured["response_format"]["json_schema"]["schema"])
    assert captured["extra_body"]["provider"]["require_parameters"] is True
    assert client.last_call_metadata["usage"]["total_tokens"] == 14


def test_google_structured_output_uses_pydantic_schema(monkeypatch):
    class Payload(BaseModel):
        value: str
        items: list[str] = Field(max_length=12)

    captured = {}

    def generate_content(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            text='{"value":"ok","items":[]}',
            usage_metadata=SimpleNamespace(prompt_token_count=8, candidates_token_count=3, total_token_count=11),
        )

    client = GoogleClient("gemini-3.5-flash")
    monkeypatch.setattr(
        client,
        "_client",
        lambda: SimpleNamespace(models=SimpleNamespace(generate_content=generate_content)),
    )
    raw = client.complete_structured("system", "user", Payload)
    assert Payload.model_validate_json(raw).value == "ok"
    schema = captured["config"]["response_json_schema"]
    assert schema["properties"]["value"]["type"] == "string"
    assert "additionalProperties" not in str(schema)
    assert "maxItems" not in str(schema)
    assert "response_schema" not in captured["config"]
    assert client.last_call_metadata["usage"]["total_tokens"] == 11
