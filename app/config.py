"""Configurações (variáveis de ambiente / arquivo .env)."""

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="APP_", extra="ignore")

    # Lida de OPENAI_API_KEY (sem o prefixo APP_). Nunca versione a chave: use o arquivo .env.
    openai_api_key: SecretStr | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    openai_model: str = "gpt-4o-mini"  # modelo pequeno/rápido; pode ser trocado sem mudar código
    request_timeout_s: float = 30.0
    max_retries: int = 2  # novas tentativas automáticas (backoff) em erros transitórios / 429
    enable_cache: bool = True  # cache em memória: textos repetidos não geram nova chamada


@lru_cache
def get_settings() -> Settings:
    return Settings()
