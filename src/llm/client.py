from langchain_litellm import ChatLiteLLM
from config import settings


def get_llm():
    return ChatLiteLLM(
        model=settings.LLM_MODEL,
        api_base=settings.LLM_API_BASE,
        temperature=settings.LLM_TEMPERATURE,
        api_key="test",
        custom_llm_provider="lm_studio",
    )
