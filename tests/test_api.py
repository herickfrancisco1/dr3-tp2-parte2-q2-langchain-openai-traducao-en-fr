"""Testes sem chamar a OpenAI: o ChatOpenAI é substituído por um chat model falso do LangChain,
mas o prompt e o parser REAIS da cadeia são exercitados."""

import os

import httpx
import openai
import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.chain import TRANSLATION_PROMPT, build_translation_chain
from app.config import Settings
from app.main import app, get_chain


def use_chain(chain) -> TestClient:
    app.state.model_name = "fake-chat-model"
    app.dependency_overrides[get_chain] = lambda: chain
    return TestClient(app)


@pytest.fixture(autouse=True)
def _clear_overrides():
    yield
    app.dependency_overrides.clear()


def test_translate_returns_french_text():
    client = use_chain(build_translation_chain(FakeListChatModel(responses=["Bonjour le monde !\n"])))
    response = client.post("/translate", json={"text": "Hello world!"})

    assert response.status_code == 200
    body = response.json()
    assert body["translated_text"] == "Bonjour le monde !"
    assert (body["source_language"], body["target_language"]) == ("en", "fr")
    assert body["model"] == "fake-chat-model"


def test_stream_endpoint_returns_text():
    client = use_chain(build_translation_chain(FakeListChatModel(responses=["Bonjour"])))
    response = client.post("/translate/stream", json={"text": "Hello"})
    assert response.status_code == 200
    assert response.text == "Bonjour"


def test_prompt_contains_user_text_and_instructions():
    messages = TRANSLATION_PROMPT.format_messages(text="Good morning")
    assert "French" in messages[0].content
    assert messages[1].content == "Good morning"


@pytest.mark.parametrize("payload", [{}, {"text": ""}, {"text": "  "}, {"text": "a" * 10_001}])
def test_invalid_input_returns_422(payload):
    client = use_chain(build_translation_chain(FakeListChatModel(responses=["x"])))
    assert client.post("/translate", json=payload).status_code == 422


class _RaisingChain:
    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    async def ainvoke(self, *_args, **_kwargs):
        raise self.exc


_REQ = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")


@pytest.mark.parametrize(
    ("exc", "expected_status"),
    [
        (openai.RateLimitError("rate", response=httpx.Response(429, request=_REQ), body=None), 429),
        (openai.APITimeoutError(request=_REQ), 504),
        (openai.AuthenticationError("auth", response=httpx.Response(401, request=_REQ), body=None), 502),
    ],
)
def test_openai_errors_are_mapped_to_http_status(exc, expected_status):
    client = use_chain(_RaisingChain(exc))
    assert client.post("/translate", json={"text": "Hello"}).status_code == expected_status


def test_insufficient_quota_is_not_reported_as_retryable():
    # Corpo real devolvido pela OpenAI para uma conta sem créditos.
    body = {"message": "You have no credits remaining.", "type": "insufficient_quota", "param": None,
            "code": "credit_balance_exhausted"}
    exc = openai.RateLimitError("quota", response=httpx.Response(429, request=_REQ), body=body)
    client = use_chain(_RaisingChain(exc))
    response = client.post("/translate", json={"text": "Hello"})
    assert response.status_code == 503
    assert "sem créditos" in response.json()["detail"]


def test_missing_api_key_returns_503(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("app.main.get_settings", lambda: Settings(_env_file=None))
    with TestClient(app) as client:  # roda o lifespan real, que não consegue criar o ChatOpenAI
        assert client.post("/translate", json={"text": "Hello"}).status_code == 503
        assert client.get("/health").json()["api_key_configured"] is False


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="requer OPENAI_API_KEY (chamada real e paga)")
def test_translate_with_real_openai():
    with TestClient(app) as client:
        body = client.post("/translate", json={"text": "Good morning, my friend."}).json()
    assert "bonjour" in body["translated_text"].lower()
