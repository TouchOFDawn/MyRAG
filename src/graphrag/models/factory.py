import platform
import subprocess
import logging
from typing import Any, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings

from graphrag.config import settings

logger = logging.getLogger(__name__)

def get_device() -> str:
    """Detects available hardware acceleration intelligently."""
    # Check Apple Silicon MPS
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "mps"
    
    # Check for CUDA availability (basic check via nvidia-smi)
    try:
        subprocess.check_output(["nvidia-smi"])
        return "cuda"
    except Exception:
        pass
        
    return "cpu"

class ModelFactory:
    """Unified factory for generating language and vision models."""

    @staticmethod
    def get_llm(model_type: Optional[str] = None, model_name: Optional[str] = None, temperature: float = 0.0, **kwargs: Any) -> BaseChatModel:
        """
        Creates an LLM instance based on the configuration or specified parameters.
        """
        model_type = model_type or settings.model_type
        
        # Determine the correct model name if not provided
        if not model_name:
            if model_type == "ollama":
                model_name = settings.ollama_model
            else:
                model_name = settings.default_model

        if model_type == "openai":
            return ChatOpenAI(
                model=model_name,
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
                temperature=temperature,
                **kwargs
            )
        elif model_type == "deepseek":
            return ChatOpenAI(
                model=model_name,
                api_key=settings.deepseek_api_key,
                base_url=settings.deepseek_base_url,
                temperature=temperature,
                **kwargs
            )
        elif model_type == "dashscope":
            return ChatOpenAI(
                model=model_name,
                api_key=settings.dashscope_api_key,
                base_url=settings.dashscope_base_url,
                temperature=temperature,
                **kwargs
            )
        elif model_type == "ollama":
            device = get_device()
            logger.info(f"Using local Ollama with detected hardware device: {device}")
            return ChatOllama(
                model=model_name,
                base_url=settings.ollama_base_url,
                temperature=temperature,
                **kwargs
            )
        elif model_type == "claude":
            from langchain_anthropic import ChatAnthropic
            model_name = model_name or "claude-opus-4-20250514"
            print
            return ChatAnthropic(
                model=model_name,
                api_key=settings.anthropic_api_key,
                base_url=settings.anthropic_base_url,
                temperature=temperature,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

    @staticmethod
    def get_embeddings(embed_type: Optional[str] = None, model_name: Optional[str] = None) -> Embeddings:
        """
        Creates an Embeddings instance based on the configuration or specified parameters.
        """
        embed_type = embed_type or settings.embed_type

        if embed_type in ["openai", "deepseek", "dashscope"]:
            from langchain_openai import OpenAIEmbeddings

            if embed_type == "dashscope":
                api_key = settings.dashscope_api_key
                base_url = settings.dashscope_base_url
                model_name = model_name or "text-embedding-v3"
            elif embed_type == "deepseek":
                api_key = settings.deepseek_api_key
                base_url = settings.deepseek_base_url
                model_name = model_name or "deepseek-embedding"
            else:
                api_key = settings.openai_api_key
                base_url = settings.openai_base_url
                model_name = model_name or "text-embedding-3-small"

            return OpenAIEmbeddings(
                model=model_name,
                api_key=api_key,
                base_url=base_url
            )
        elif embed_type == "ollama":
            model_name = model_name or settings.ollama_embed_model
            device = get_device()
            logger.info(f"Using local Ollama embeddings with detected hardware device: {device}")
            return OllamaEmbeddings(
                model=model_name,
                base_url=settings.ollama_base_url
            )
        elif embed_type == "huggingface":
            from langchain_huggingface import HuggingFaceEmbeddings
            model_name = model_name or settings.hf_embed_model
            device = get_device()
            logger.info(f"Using HuggingFace embeddings on device: {device}")
            return HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": device}
            )
        else:
            raise ValueError(f"Unsupported embed_type: {embed_type}")
