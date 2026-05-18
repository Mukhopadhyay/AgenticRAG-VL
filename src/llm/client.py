from langchain_litellm import ChatLiteLLM
from litellm.exceptions import RateLimitError as LiteLLMRateLimitError
from config import settings


def _build_llm() -> ChatLiteLLM:
    kwargs: dict = dict(
        model=settings.LLM_MODEL,
        temperature=settings.LLM_TEMPERATURE,
    )
    if settings.LLM_API_BASE:
        kwargs["api_base"] = settings.LLM_API_BASE
        kwargs["api_key"] = "lm-studio"
    if settings.LLM_API_KEY:
        kwargs["api_key"] = settings.LLM_API_KEY
    return ChatLiteLLM(**kwargs)


def get_raw_llm() -> ChatLiteLLM:
    """Return the bare ChatLiteLLM instance (no retry wrapper)."""
    return _build_llm()


def get_llm():
    llm = _build_llm()
    return llm.with_retry(
        retry_if_exception_type=(LiteLLMRateLimitError,),
        wait_exponential_jitter=True,
        stop_after_attempt=6,
    )
