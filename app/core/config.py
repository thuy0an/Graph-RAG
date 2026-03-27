from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # --- Provider selection ---
    LLM_PROVIDER: str = "ollama"    # ollama | openai | groq | anthropic
    EMBED_PROVIDER: str = "ollama"  # ollama | openai | huggingface

    # --- Ollama ---
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2"
    OLLAMA_EMBED_MODEL: str = "nomic-embed-text"

    # --- OpenAI ---
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_EMBED_MODEL: str = "text-embedding-3-small"

    # --- Groq ---
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.3-70b-versatile"

    # --- Anthropic ---
    ANTHROPIC_API_KEY: str = ""
    ANTHROPIC_MODEL: str = "claude-3-5-haiku-20241022"

    # --- HuggingFace Embeddings ---
    HF_EMBED_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Neo4j ---
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "password"

    class Config:
        env_file = ".env"


settings = Settings()
