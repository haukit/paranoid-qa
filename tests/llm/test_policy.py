import pytest

from paranoid_qa.config import settings
from paranoid_qa.llm.policy import validate_model_policy


def test_same_family_fails_with_actionable_message(monkeypatch):
    monkeypatch.setattr(settings, "gen_model_family", "gpt-4o")
    monkeypatch.setattr(settings, "critic_model_family", "gpt-4o")
    with pytest.raises(ValueError, match="different model family"):
        validate_model_policy()


def test_different_families_pass(monkeypatch):
    monkeypatch.setattr(settings, "gen_model_family", "gpt-4o")
    monkeypatch.setattr(settings, "critic_model_family", "gpt-5.4")
    validate_model_policy()  # does not raise


def test_empty_family_fails(monkeypatch):
    monkeypatch.setattr(settings, "gen_model_family", "")
    monkeypatch.setattr(settings, "critic_model_family", "gpt-5.4")
    with pytest.raises(ValueError):
        validate_model_policy()
