from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    # DeepSeek configs
    deepseek_base_url: Optional[str] = "https://api.deepseek.com"
    deepseek_api_key: Optional[str] = None
    
    # DashScope configs
    dashscope_base_url: Optional[str] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    dashscope_api_key: Optional[str] = None
    
    # OpenAI/OpenRouter configs
    openai_base_url: Optional[str] = "https://code.newcli.com/v1"
    openai_api_key: Optional[str] = None
    
    # Local Ollama configs
    ollama_base_url: Optional[str] = "http://127.0.0.1:11434"
    ollama_model: str = "qwen3:8b"
    ollama_embed_model: str = "qwen3-embedding:8b"

    # Claude configs (via relay)
    anthropic_base_url: Optional[str] = "https://code.newcli.com/claude/aws"
    anthropic_api_key: Optional[str] = None

    # HuggingFace local embedding model
    hf_embed_model: str = "sentence-transformers/all-distilroberta-v1"

    model_type: str = "claude" # "openai", "deepseek", "dashscope", "ollama", "claude"
    embed_type: str = "huggingface" # "openai", "dashscope", "ollama", "huggingface"
    default_model: str = "claude-sonnet-4-20250514"

    # MinerU configs
    mineru_base_url: Optional[str] = "https://mineru.net"
    mineru_api_key: Optional[str] = None
    
    # Database Architecture
    vector_db_type: str = "chroma" # Options: "chroma", "faiss" 
    graph_db_type: str = "neo4j"   # Options: "neo4j"
    
    # Neo4j Settings
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = "12345678"
    
    # Multimodal Flags
    use_multimodal: bool = True
    vision_model: str = "gpt-4o"
    
    # LangSmith
    langchain_tracing_v2: Optional[str] = None
    langchain_api_key: Optional[str] = None
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
