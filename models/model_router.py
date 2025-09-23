# -*- coding: utf-8 -*-
import logging
import os
import requests
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from collections import OrderedDict

logger = logging.getLogger("Reranker")


@dataclass
class Model:
    """
    Abstract base class representing a model served by the HiveGPT Model Router.

    Attributes:
        name (str): The HuggingFace repository path for the model, e.g., "meta-llama/Meta-Llama-3.1-8B".
        alias (str): A shorter, more user-friendly alias or identifier for the model.
        openai_endpoint (str): The base openai endpoint through which the model can be accessed.
    """

    name: str
    alias: str
    openai_endpoint: str


@dataclass
class LLMModel(Model):
    """
    Represents an LLM served by the HiveGPT Model Router.

    Attributes:
        name (str): The HuggingFace repository path for the model, e.g., "meta-llama/Meta-Llama-3.1-8B".
        alias (str): A shorter, more user-friendly alias or identifier for the model.
        openai_endpoint (str): The base openai endpoint through which the model can be accessed.
        max_len (int): The maximum sequence length that the model can handle.
    """

    max_len: int


@dataclass
class EmbeddingModel(Model):
    """
    Represents an Embedding model served by the HiveGPT Model Router.

    Attributes:
        name (str): The HuggingFace repository path for the embedding model, e.g., "BAAI/bge-m3".
        alias (str): A shorter, more user-friendly alias or identifier for the embedding model.
        openai_endpoint (str): The base openai endpoint through which this embedding model can be accessed.
        batch_size (int): The maximum batch size for inference.
        backend (str): The backend used for the embedding model.
    """

    batch_size: int
    backend: str


@dataclass
class RerankingModel(Model):
    """
    Represents a Reranking model served by the HiveGPT Model Router.

    Attributes:
        name (str): The HuggingFace repository path for the reranking model, e.g., "BAAI/bge-reranker-large"".
        alias (str): A shorter, more user-friendly alias or identifier for the rera,king model.
        openai_endpoint (str): The base openai endpoint through which this reranking model can be accessed.
        batch_size (int): The maximum batch size for inference.
        backend (str): The backend used for the reranking model.
    """

    batch_size: int
    backend: str

    async def rerank(self, query: str, documents: List[str]) -> Union[dict, None]:
        """
        Use the reranker model to rerank the documents based on relevance to the query.
        Args:
            query (str): The original query text.
            documents (List[dict]): The initial list of documents from the vector database search.
        Returns:
            List[dict]: The reranked list of documents.
            None: in case of error.
        """
        try:
            # Import infinity client if available
            from infinity_client.api.default import rerank
            from infinity_client import Client as InfinityClient
            from infinity_client import models as InfinityTypes

            rerank_client = InfinityClient(base_url=self.openai_endpoint, raise_on_unexpected_status=True)
            rerank_result = await rerank.asyncio(
                client=rerank_client, body=InfinityTypes.RerankInput(query=query, documents=documents, model=self.name)
            )
            logger.debug(f"Rerank response: {rerank_result}")
        except Exception as e:
            logger.error(f"Failed to rerank documents: {e}")
            return None

        if rerank_result is not None and hasattr(rerank_result, 'to_dict'):
            result = rerank_result.to_dict()
            return result

        return None


class ModelRouter:
    """
    A wrapper class that fetches info from the HiveGPT Model Router
    """

    def __init__(self, host: str = None, port: int = None):
        """
        Initializes the ModelRouter.

        Args:
            host (str): The hostname of the Model Router server.
            port (int): The port number of the Model Router server.

        Note: The ModelRouter will automatically refresh the map of served models upon initialization.
        """
        # Use environment variables with defaults
        self.host = host or os.getenv("MODEL_ROUTER_HOST", "localhost")
        self.port = port or int(os.getenv("MODEL_ROUTER_PORT", "8000"))
        self.models_health_endpoint = f"http://{self.host}:{self.port}/v1/models"
        self.served_models: Dict[str, LLMModel] = {}
        self.served_embedding_models: Dict[str, EmbeddingModel] = {}
        self.served_reranking_models: Dict[str, RerankingModel] = {}
        self.logger = logging.getLogger("HiveGPT Model Router")
        self.default_llm_name = os.getenv("DEFAULT_LLM_NAME", "default")
        self.embedding_models_tag = os.getenv("EMBEDDING_MODELS_TAG", "embed")
        self.reranking_models_tag = os.getenv("RERANKING_MODELS_TAG", "rerank")
        self.refresh()

    def _generate_openai_base(self, alias: str, base_endpoint: str = "/v1") -> str:
        """
        Generates the base OpenAI endpoint URL for a given alias.

        Args:
            alias (str): The alias of the model.
            base_endpoint (str): The base endpoint for the OpenAI API.

        Returns:
            str: The base OpenAI endpoint URL for the given alias.
        """
        return f"http://{self.host}:{self.port}/{alias}{base_endpoint}"
    
    def _sort_language_models(self):
        """
        Sort returned models by alias in ascending order 
        and put the default LLM always on top.
        """
        default_model_key = self.default_llm_name

        # Get the default model
        default_model = {default_model_key: self.served_models[default_model_key]} if default_model_key in self.served_models else None

        # Sort remaining models in ascending order
        other_models = {k: v for k, v in self.served_models.items() if k != default_model_key}
        sorted_other_models = OrderedDict(sorted(other_models.items(), key=lambda item: item[0]))

        # Combine the default model and the sorted models
        sorted_llms = sorted_other_models
        if default_model is not None:
            sorted_llms = OrderedDict(**default_model, **sorted_other_models)

        # Update the served_models dictionary
        self.served_models = sorted_llms

    def refresh(self):
        """Refreshes the map of served models."""
        try:
            response = requests.get(self.models_health_endpoint)
            response.raise_for_status()
            models_json = response.json()

            models = {}
            embeddings = {}
            rerankers = {}
            for model in models_json:
                is_embedding = False
                is_reranker = False
                alias = model["model_alias"]
                name = model["model_name"]
                max_len = model["max_model_len"]
                openai_endpoint = self._generate_openai_base(alias=alias)
                if model["capabilities"] is not None:
                    is_embedding = self.embedding_models_tag in list(model["capabilities"])
                    is_reranker = self.reranking_models_tag in list(model["capabilities"])
                if is_embedding:
                    batch_size = 0
                    stats = model["stats"]
                    if stats is not None:
                        batch_size = stats["batch_size"]
                    backend = model["backend"]
                    embeddings[name] = EmbeddingModel(
                        name=name, alias=alias, openai_endpoint=openai_endpoint, batch_size=batch_size, backend=backend
                    )
                if is_reranker:
                    batch_size = 0
                    stats = model["stats"]
                    if stats is not None:
                        batch_size = stats["batch_size"]
                    backend = model["backend"]
                    rerankers[name] = RerankingModel(
                        name=name, alias=alias, openai_endpoint=openai_endpoint, batch_size=batch_size, backend=backend
                    )
                if not is_embedding and not is_reranker:
                    models[name] = LLMModel(name=name, alias=alias, openai_endpoint=openai_endpoint, max_len=max_len)

            self.served_models = models
            self._sort_language_models()
            self.served_embedding_models = embeddings
            self.served_reranking_models = rerankers
            self.logger.info("Models map successfully refreshed.")

        except requests.RequestException as e:
            self.logger.error(f"Failed to refresh models map: {e}")
            self.served_models = {}
            self.served_embedding_models = {}
            self.served_reranking_models = {}

    def get_llm_model(self, name: str) -> Optional[LLMModel]:
        """Gets the LLMModel object for the specified model name.

        Args:
            name (str): The HuggingFace repository path for the model. for example, "meta-llama/Meta-Llama-3.1-8B"

        Returns:
            Optional[Model]: The Model object.
                             Returns None if the model name is not found.
        """
        return self.served_models.get(name)

    def get_all_llm_models(self) -> Dict[str, LLMModel]:
        """Returns a map of all served LLMs.

        Returns:
            Dict[str, LLMModel]: A dictionary where keys are LLM names and values are LLMModel objects.
        """
        self._sort_language_models()
        return self.served_models

    def get_embedding_model(self, name: str) -> Optional[EmbeddingModel]:
        """Gets the EmbeddingModel object for the specified embedding model name.

        Args:
            name (str): The HuggingFace repository path for the embedding model. for example, "BAAI/bge-m3"

        Returns:
            Optional[EmbeddingModel]: The EmbeddingModel object.
                             Returns None if the embedding model name is not found.
        """
        return self.served_embedding_models.get(name)

    def get_all_embedding_models(self) -> Dict[str, EmbeddingModel]:
        """Returns a map of all served embedding models.

        Returns:
            Dict[str, EmbeddingModel]: A dictionary where keys are embedding model names and values are EmbeddingModel objects.
        """
        return self.served_embedding_models

    def get_reranking_model(self, name: str) -> Optional[RerankingModel]:
        """Gets the RerankinModel object for the specified reranking model name.

        Args:
            name (str): The HuggingFace repository path for the reranking model. for example, "BAAI/bge-reranker-large"

        Returns:
            Optional[RerankingModel]: The RerankingModel object.
                             Returns None if the reranking model name is not found.
        """
        return self.served_reranking_models.get(name)

    def get_all_reranking_models(self) -> Dict[str, RerankingModel]:
        """Returns a map of all served reranking models.

        Returns:
            Dict[str, RerankingModel]: A dictionary where keys are reranking model names and values are RerankingModel objects.
        """
        return self.served_reranking_models