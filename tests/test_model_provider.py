"""Contracts for primary and escalation model-provider configuration."""

import pytest

from src.model_provider import (
    escalation_is_configured,
    get_model_config,
    redact_secrets,
    sanitize_provider_error,
)


def test_primary_config_pairs_generic_environment_only(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "primary-sentinel-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://primary.invalid/v1/")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("LLM_MODEL", "primary-model")

    config = get_model_config("primary")

    assert config.provider == "primary"
    assert config.model == "primary-model"
    assert config.base_url == "https://primary.invalid/v1"
    assert config.api_key.get_secret_value() == "primary-sentinel-key"


def test_primary_config_accepts_unambiguous_deprecated_openai_alias(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "primary-sentinel-key")
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://legacy.invalid/v1/")

    with pytest.warns(DeprecationWarning, match="OPENAI_BASE_URL"):
        config = get_model_config("primary")

    assert config.base_url == "https://legacy.invalid/v1"


def test_primary_config_rejects_conflicting_base_variables(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "primary-sentinel-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://primary.invalid/v1")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://other.invalid/v1")

    with pytest.raises(ValueError, match="conflicting"):
        get_model_config("primary")


def test_primary_config_ignores_cross_vendor_keys_and_requires_llm_key(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("LINOAPI_API_KEY", "lino-sentinel")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-sentinel")

    with pytest.raises(ValueError, match="LLM_API_KEY"):
        get_model_config("primary")


def test_escalation_config_uses_only_its_own_environment(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "primary-sentinel-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://primary.invalid/v1")
    monkeypatch.setenv("LLM_MODEL", "primary-model")
    monkeypatch.setenv("LLM_ESCALATION_API_KEY", "escalation-sentinel-key")
    monkeypatch.setenv("LLM_ESCALATION_BASE_URL", "https://escalation.invalid/v1/")
    monkeypatch.setenv("LLM_ESCALATION_MODEL", "escalation-model")

    config = get_model_config("escalation")

    assert config.provider == "escalation"
    assert config.model == "escalation-model"
    assert config.base_url == "https://escalation.invalid/v1"
    assert config.api_key.get_secret_value() == "escalation-sentinel-key"


def test_missing_escalation_key_never_falls_back_to_primary(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "primary-sentinel-key")
    monkeypatch.delenv("LLM_ESCALATION_API_KEY", raising=False)

    config = get_model_config("escalation")

    assert config.api_key.get_secret_value() == ""


def test_whitespace_escalation_key_is_not_configured(monkeypatch):
    monkeypatch.setenv("REPOPILOT_ESCALATION_ENABLED", "1")
    monkeypatch.setenv("LLM_ESCALATION_API_KEY", "   ")

    assert escalation_is_configured() is False


def test_escalation_requires_explicit_enablement_and_a_key(monkeypatch):
    monkeypatch.setenv("LLM_ESCALATION_API_KEY", "escalation-sentinel-key")
    monkeypatch.delenv("REPOPILOT_ESCALATION_ENABLED", raising=False)
    assert escalation_is_configured() is False

    monkeypatch.setenv("REPOPILOT_ESCALATION_ENABLED", "1")
    assert escalation_is_configured() is True

    monkeypatch.delenv("LLM_ESCALATION_API_KEY")
    assert escalation_is_configured() is False


@pytest.mark.parametrize(
    "base_url",
    [
        "http://provider.invalid/v1",
        "https://user:password@provider.invalid/v1",
        "https://provider.invalid/v1?api_key=secret",
        "https://provider.invalid/v1#fragment",
        "ftp://provider.invalid/v1",
        "http://localhost.evil.invalid/v1",
        "http://127.0.0.1.evil.invalid/v1",
        "http://127.1/v1",
        "http://2130706433/v1",
        "https:///missing-host",
    ],
)
def test_primary_config_rejects_unsafe_base_urls(monkeypatch, base_url):
    monkeypatch.setenv("LLM_API_KEY", "primary-sentinel-key")
    monkeypatch.setenv("LLM_BASE_URL", base_url)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    with pytest.raises(ValueError, match="base URL"):
        get_model_config("primary")


@pytest.mark.parametrize(
    "base_url",
    [
        "http://localhost:8000/v1",
        "http://127.0.0.1:8000/v1",
        "http://[::1]:8000/v1",
        "https://provider.invalid/v1",
    ],
)
def test_primary_config_allows_https_and_exact_loopback_http(monkeypatch, base_url):
    monkeypatch.setenv("LLM_API_KEY", "primary-sentinel-key")
    monkeypatch.setenv("LLM_BASE_URL", base_url)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    assert get_model_config("primary").base_url == base_url


def test_sanitize_provider_error_redacts_bearer_query_and_header_values():
    exc = RuntimeError(
        "provider failed: Authorization: Bearer bearer-sentinel-key; "
        "https://provider.invalid/v1?api_key=query-sentinel-key&mode=test; "
        "X-API-Key=header-sentinel-key"
    )

    message = sanitize_provider_error(exc)

    assert "bearer-sentinel-key" not in message
    assert "query-sentinel-key" not in message
    assert "header-sentinel-key" not in message
    assert "[REDACTED]" in message


def test_sanitize_provider_error_redacts_quoted_secret_fields():
    message = sanitize_provider_error(
        RuntimeError(
            "provider failed: {'api_key': 'single-quoted-sentinel'}; "
            '{"Authorization": "Bearer double-quoted-sentinel"}'
        )
    )

    assert "single-quoted-sentinel" not in message
    assert "double-quoted-sentinel" not in message
    assert "[REDACTED]" in message


def test_redact_secrets_catches_bare_sk_tokens_without_a_header():
    token = "sk-adversarial-sentinel-token"

    message = redact_secrets(f"runner printed {token} in plain text")

    assert token not in message
    assert "runner printed [REDACTED]" in message
