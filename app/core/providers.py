from functools import lru_cache
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings


@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    """Khởi tạo và cache LLM client dựa theo LLM_PROVIDER trong config.

    Hỗ trợ: ollama (local), openai, groq, anthropic.
    Chỉ khởi tạo một lần nhờ lru_cache, tái sử dụng cho mọi request.
    """
    from app.core.config import settings

    provider = settings.LLM_PROVIDER.lower()

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=settings.OLLAMA_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=0,  # Deterministic output
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.OPENAI_MODEL,
            api_key=settings.OPENAI_API_KEY,
            temperature=0,
        )

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=settings.GROQ_MODEL,
            api_key=settings.GROQ_API_KEY,
            temperature=0,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=settings.ANTHROPIC_MODEL,
            api_key=settings.ANTHROPIC_API_KEY,
            temperature=0,
        )

    raise ValueError(f"Unsupported LLM_PROVIDER: {provider}. Choose: ollama, openai, groq, anthropic")


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """Khởi tạo và cache Embeddings model dựa theo EMBED_PROVIDER trong config.

    Hỗ trợ: ollama (local), openai, huggingface (local).
    Chỉ khởi tạo một lần nhờ lru_cache, tái sử dụng cho mọi request.
    """
    from app.core.config import settings

    provider = settings.EMBED_PROVIDER.lower()

    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(
            model=settings.OLLAMA_EMBED_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
        )

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=settings.OPENAI_EMBED_MODEL,
            api_key=settings.OPENAI_API_KEY,
        )

    if provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=settings.HF_EMBED_MODEL)

    raise ValueError(f"Unsupported EMBED_PROVIDER: {provider}. Choose: ollama, openai, huggingface")
