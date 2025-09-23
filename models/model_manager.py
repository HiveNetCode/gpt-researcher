from typing import Dict, Any, List
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.embeddings import Embeddings
import threading
import os
from openai import OpenAI
from .model_router import ModelRouter, LLMModel, EmbeddingModel, RerankingModel


class HiveGPTEmbeddings(Embeddings):
    """Custom embedding wrapper that uses raw OpenAI client for HiveGPT router."""
    
    def __init__(self, model_name: str, loaded_model: EmbeddingModel, api_key: str):
        self.model_name = model_name
        self.loaded_model = loaded_model
        self.client = OpenAI(
            api_key=api_key,
            base_url=loaded_model.openai_endpoint,
        )
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        response = self.client.embeddings.create(
            model=self.model_name,
            input=texts
        )
        return [item.embedding for item in response.data]
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text."""
        response = self.client.embeddings.create(
            model=self.model_name,
            input=text
        )
        return response.data[0].embedding


class ModelManager:
    """
    Manages the loading and caching of LLM and Reranking models.
    """

    def __init__(self):
        """
        Initializes the ModelManager, including its internal caches and locks.

        The ModelManager is a singleton responsible for loading and caching LLM and
        Reranking models. It keeps an internal cache of loaded models and locks to
        ensure thread-safety.
        """
        self._llm_models: Dict[str, ChatOpenAI] = {}
        self._embedding_models: Dict[str, HiveGPTEmbeddings] = {}
        self._reranker_models: Dict[str, RerankingModel] = {}
        self._llm_lock = threading.Lock()
        self._embedding_lock = threading.Lock()
        self._reranker_lock = threading.Lock()
        self.model_router = ModelRouter()

    def load_llm_model(self, model_name: str) -> ChatOpenAI:
        """
        Synchronously loads and returns a language model for the specified model name.

        This method checks if the model is already loaded and cached in the class-level
        dictionary `_llm_models`. If not, it acquires a lock to ensure thread-safe
        model loading, retrieves the model information from the Model Router, initializes
        a `ChatOpenAI` instance with the given parameters, and caches it for future use.

        Args:
            model_name (str): The name of the language model to load.

        Returns:
            ChatOpenAI: An instance of the loaded language model.
        """
        if model_name in self._llm_models:
            return self._llm_models[model_name]
        with self._llm_lock:
            if model_name not in self._llm_models:
                loaded_model: LLMModel = self.model_router.get_llm_model(model_name)
                if loaded_model is None:
                    raise ValueError(f"Model {model_name} not found in model router")
                
                llm = ChatOpenAI(
                    model_name=model_name,  # Use the alias instead of the full model name
                    api_key=os.getenv("MODEL_ROUTER_TOKEN", "dummy-token"),
                    base_url=loaded_model.openai_endpoint,
                    temperature=0.1,
                )
                self._llm_models[model_name] = llm
            return self._llm_models[model_name]

    def load_embedding_model(self, model_name: str) -> HiveGPTEmbeddings:
        """
        Synchronously loads and returns an embedding model for the specified model name.

        This method checks if the model is already loaded and cached in the class-level
        dictionary `_embedding_models`. If not, it acquires a lock to ensure thread-safe
        model loading, retrieves the model information from the Model Router, initializes
        a custom embedding wrapper with the raw OpenAI client, and caches it for future use.

        Args:
            model_name (str): The name of the embedding model to load.

        Returns:
            HiveGPTEmbeddings: An instance of the loaded embedding model wrapper.
        """
        if model_name in self._embedding_models:
            return self._embedding_models[model_name]
        with self._embedding_lock:
            if model_name not in self._embedding_models:
                loaded_model: EmbeddingModel = self.model_router.get_embedding_model(model_name)
                if loaded_model is None:
                    raise ValueError(f"Embedding model {model_name} not found in model router")
                
                api_key = os.getenv("MODEL_ROUTER_TOKEN", "dummy-token")
                embedding = HiveGPTEmbeddings(
                    model_name=model_name,
                    loaded_model=loaded_model,
                    api_key=api_key
                )
                self._embedding_models[model_name] = embedding
            return self._embedding_models[model_name]

    def load_reranker_model(self, model_name: str) -> RerankingModel:
        """
        Synchronously loads and returns a reranking model for the specified model name.

        This method checks if the model is already loaded and cached in the class-level
        dictionary `_reranker_models`. If not, it acquires a lock to ensure thread-safe
        model loading, retrieves the model information from the Model Router, and caches it
        for future use.

        Args:
            model_name (str): The name of the reranking model to load.

        Returns:
            RerankingModel: An instance of the loaded reranking model.
        """
        if model_name in self._reranker_models:
            return self._reranker_models[model_name]
        with self._reranker_lock:
            if model_name not in self._reranker_models:
                loaded_model = self.model_router.get_reranking_model(model_name)
                if loaded_model is None:
                    raise ValueError(f"Reranking model {model_name} not found in model router")
                self._reranker_models[model_name] = loaded_model
            return self._reranker_models[model_name]