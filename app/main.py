"""API de tradução EN -> FR usando a OpenAI via LangChain.

Execução:  uvicorn app.main:app --reload     (requer OPENAI_API_KEY)
Docs:      http://127.0.0.1:8000/docs
"""

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import openai
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache
from langchain_core.runnables import Runnable

from app.chain import build_llm, build_translation_chain
from app.config import get_settings
from app.schemas import TranslationRequest, TranslationResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Cria o cliente/cadeia UMA vez (reaproveita conexões HTTP entre requisições)."""
    settings = get_settings()
    app.state.model_name = settings.openai_model
    if settings.enable_cache:
        set_llm_cache(InMemoryCache())
    try:
        app.state.chain = build_translation_chain(build_llm(settings))
    except RuntimeError as exc:  # sem chave: a API sobe, mas /translate responde 503
        logger.error("%s", exc)
        app.state.chain = None
    yield


app = FastAPI(
    title="EN -> FR Translation API (OpenAI + LangChain)",
    description="Traduz textos do inglês para o francês com um modelo da OpenAI orquestrado pelo LangChain.",
    version="1.0.0",
    lifespan=lifespan,
)


def get_chain(request: Request) -> Runnable:
    chain = getattr(request.app.state, "chain", None)
    if chain is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Serviço indisponível: OPENAI_API_KEY ausente.")
    return chain


# --- Tradução de erros da OpenAI em respostas HTTP adequadas -----------------------------------
@app.exception_handler(openai.RateLimitError)
async def rate_limit_handler(_: Request, exc: openai.RateLimitError) -> JSONResponse:
    # A OpenAI usa o mesmo HTTP 429 para "muitas requisições" e "conta sem créditos";
    # só o primeiro caso se resolve tentando de novo.
    if exc.type == "insufficient_quota" or exc.code in {"insufficient_quota", "credit_balance_exhausted"}:
        logger.error("Conta OpenAI sem créditos (insufficient_quota)")
        return JSONResponse(status_code=503, content={"detail": "Conta da OpenAI sem créditos. Adicione créditos para usar a API."})
    return JSONResponse(status_code=429, content={"detail": "Limite de requisições da OpenAI atingido. Tente novamente em instantes."})


@app.exception_handler(openai.APITimeoutError)
async def timeout_handler(_: Request, exc: openai.APITimeoutError) -> JSONResponse:
    return JSONResponse(status_code=504, content={"detail": "A OpenAI não respondeu a tempo."})


@app.exception_handler(openai.AuthenticationError)
async def auth_handler(_: Request, exc: openai.AuthenticationError) -> JSONResponse:
    logger.error("Falha de autenticação na OpenAI: verifique OPENAI_API_KEY")
    return JSONResponse(status_code=502, content={"detail": "Falha de autenticação com o provedor de LLM."})


@app.exception_handler(openai.APIError)
async def api_error_handler(_: Request, exc: openai.APIError) -> JSONResponse:
    logger.error("Erro da API OpenAI: %s", exc)
    return JSONResponse(status_code=502, content={"detail": "Erro no provedor de LLM."})


# --- Rotas ----------------------------------------------------------------------------------------
@app.get("/health", tags=["infra"])
def health(request: Request) -> dict:
    return {
        "status": "ok",
        "model": request.app.state.model_name,
        "api_key_configured": getattr(request.app.state, "chain", None) is not None,
    }


# Rota assíncrona: a chamada à OpenAI é I/O de rede; `ainvoke` libera o event loop enquanto
# espera, então um único processo atende muitas traduções simultâneas.
@app.post("/translate", response_model=TranslationResponse, tags=["tradução"])
async def translate(
    payload: TranslationRequest,
    request: Request,
    chain: Runnable = Depends(get_chain),
) -> TranslationResponse:
    start = time.perf_counter()
    translated = await chain.ainvoke({"text": payload.text})
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info("Tradução concluída em %.0f ms", elapsed_ms)
    return TranslationResponse(
        original_text=payload.text,
        translated_text=translated.strip(),
        model=request.app.state.model_name,
        elapsed_ms=round(elapsed_ms, 2),
    )


@app.post("/translate/stream", tags=["tradução"])
async def translate_stream(payload: TranslationRequest, chain: Runnable = Depends(get_chain)) -> StreamingResponse:
    """Devolve a tradução em streaming (texto puro): o cliente vê os primeiros tokens em ~centenas de ms."""
    return StreamingResponse(chain.astream({"text": payload.text}), media_type="text/plain; charset=utf-8")
