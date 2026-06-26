import pytest

from confirm.llm import (
    AnthropicClient,
    OpenAIClient,
    OpenRouterClient,
    StandInClient,
    _create_chat_completion_with_param_fallback,
    make_llm,
)


def test_make_llm_parses_provider_model_specs():
    openai = make_llm("openai:gpt-5-mini")
    anthropic = make_llm("anthropic:claude-haiku-4-5")
    openrouter = make_llm("openrouter:deepseek/deepseek-chat")
    standin = make_llm("standin")

    assert isinstance(openai, OpenAIClient)
    assert openai.model == "gpt-5-mini"
    assert isinstance(anthropic, AnthropicClient)
    assert anthropic.model == "claude-haiku-4-5"
    assert isinstance(openrouter, OpenRouterClient)
    assert openrouter.model == "deepseek/deepseek-chat"
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
