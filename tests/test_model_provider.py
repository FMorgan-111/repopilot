"""Contracts for primary and escalation model-provider configuration."""

from src.model_provider import (
    escalation_is_configured,
    get_model_config,
    sanitize_provider_error,
)


def test_primary_config_keeps_legacy_environment_fallbacks(monkeypatch):
    monkeypatch.delenv("LINOAPI_API_KEY", raising=False)
    monkeypatch.setenv("LLM_API_KEY", "primary-sentinel-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://primary.invalid/v1/")
    monkeypatch.setenv("LLM_MODEL", "primary-model")

    config = get_model_config("primary")

    assert config.provider == "primary"
    assert config.model == "primary-model"
    assert config.base_url == "https://primary.invalid/v1"
    assert config.api_key.get_secret_value() == "primary-sentinel-key"


def test_escalation_config_uses_only_its_own_environment(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "primary-sentinel-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://primary.invalid/v1")
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


def test_escalation_requires_explicit_enablement_and_a_key(monkeypatch):
    monkeypatch.setenv("LLM_ESCALATION_API_KEY", "escalation-sentinel-key")
    monkeypatch.delenv("REPOPILOT_ESCALATION_ENABLED", raising=False)
    assert escalation_is_configured() is False

    monkeypatch.setenv("REPOPILOT_ESCALATION_ENABLED", "1")
    assert escalation_is_configured() is True

    monkeypatch.delenv("LLM_ESCALATION_API_KEY")
    assert escalation_is_configured() is False


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
