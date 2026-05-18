from langchain_litellm import ChatLiteLLM
from litellm.exceptions import RateLimitError as LiteLLMRateLimitError
from config import settings


def get_llm():

    kwargs: dict = dict(
        model=settings.LLM_MODEL,
        temperature=settings.LLM_TEMPERATURE,
    )
    if settings.LLM_API_BASE:
        kwargs["api_base"] = settings.LLM_API_BASE
        kwargs["api_key"] = "lm-studio"
    if settings.LLM_API_KEY:
        kwargs["api_key"] = settings.LLM_API_KEY
    llm = ChatLiteLLM(**kwargs)
    return llm.with_retry(
        retry_if_exception_type=(LiteLLMRateLimitError,),
        wait_exponential_jitter=True,
        stop_after_attempt=6,
    )
