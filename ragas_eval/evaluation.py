"""
Metrics evaluation module for the RAG evaluation framework.

This module provides classes and utilities for evaluating RAG system performance
using various metrics from the Ragas framework.
"""

import os
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dataclasses import dataclass, asdict
from ragas import RunConfig, evaluate, EvaluationDataset
from ragas.cache import DiskCacheBackend
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    ContextPrecision,
    ContextRecall,
)


@dataclass
class RagasEvaluationConfig:
    """Default configuration for RAG evaluation"""

    llm_model: str = (
        os.getenv("RAGAS_LLM_MODEL") or "ISTA-DASLab/gemma-3-4b-it-GPTQ-4b-128g"
    )
    llm_temperature: float = 0.1
    llm_timeout: int = 120
    llm_max_tokens: int = 4096
    llm_top_p: float = 1.0
    llm_api_base: str = "http://localhost:8000/v1"
    llm_api_key: str = os.getenv("RAGAS_LLM_API_KEY") or "ragas-api-key"

    embeddings_model: str = "BAAI/bge-m3"
    embeddings_timeout: int = 240
    embeddings_api_base: str = "http://localhost:8000/v1"
    embeddings_api_key: str = os.getenv("RAGAS_EMBEDDINGS_API_KEY") or "ragas-api-key"

    cache_dir: str = "ragas_cache"

    batch_size: int = 256
    eval_timeout: int = 240
    max_workers: Optional[int] = os.cpu_count() * 2

    metrics: List[str] = None

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ["faithfulness", "context_precision", "context_recall"]

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        return str(self.to_dict())


class RagasLLMFactory:
    """Factory for creating language models for evaluation"""

    @staticmethod
    def create_llm(
        model: str,
        api_key: str,
        api_base: str,
        temperature: float,
        top_p: float,
        timeout: int,
        max_tokens: int,
        cache: Optional[DiskCacheBackend] = None,
    ) -> LangchainLLMWrapper:
        """
        Create a language model wrapped for Ragas

        Args:
            model: Model identifier
            api_key: API key for the model
            api_base: Base URL for the API
            temperature: Model temperature setting
            cache: Optional cache backend

        Returns:
            Wrapped language model
        """
        llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=api_base,
            temperature=temperature,
            top_p=top_p,
            timeout=timeout,
            max_tokens=max_tokens,
        )
        return LangchainLLMWrapper(llm, cache=cache)


class RagasEmbeddingFactory:
    """Factory for creating embeddings models for evaluation"""

    @staticmethod
    def create_embeddings(
        model: str,
        api_key: str,
        api_base: str,
        timeout: int,
        cache: Optional[DiskCacheBackend] = None,
    ) -> LangchainEmbeddingsWrapper:
        """
        Create an embeddings model wrapped for Ragas

        Args:
            model: Model identifier
            api_key: API key for the model
            api_base: Base URL for the API
            cache: Optional cache backend

        Returns:
            Wrapped embeddings model
        """
        embeddings = OpenAIEmbeddings(
            model=model,
            api_key=api_key,
            base_url=api_base,
            timeout=timeout,
        )
        return LangchainEmbeddingsWrapper(embeddings, cache=cache)


class RagasMetricsFactory:
    """Factory for creating evaluation metrics"""

    @staticmethod
    def create_metrics(metric_names: Optional[List[str]] = None) -> List[Any]:
        """
        Create a list of metrics based on provided names

        Args:
            metric_names: Names of metrics to create (if None, creates all). Available metrics are

            "faithfulness", "context_precision", "context_recall"

        Returns:
            List of metric instances
        """
        available_metrics = {
            "faithfulness": Faithfulness,
            "context_precision": ContextPrecision,
            "context_recall": ContextRecall,
        }

        if not metric_names:
            return [metric_class() for metric_class in available_metrics.values()]

        metrics = []
        for name in metric_names:
            name = name.lower()
            if name in available_metrics:
                metrics.append(available_metrics[name]())
            else:
                raise ValueError(f"Unknown metric: {name}")

        return metrics or [
            metric_class() for metric_class in available_metrics.values()
        ]


class CacheManager:
    """Manager for Ragas cache"""

    def __init__(self, cache_dir: str = "ragas_cache"):
        """
        Initialize cache manager, used by Ragas to speed up evaluation

        Args:
            cache_dir: Directory for storing cache
        """
        self.cache_dir = cache_dir
        self._cache = None

    @property
    def cache(self) -> DiskCacheBackend:
        """
        Get or create cache instance

        Returns:
            Cache backend
        """
        if self._cache is None:
            self._cache = DiskCacheBackend(cache_dir=self.cache_dir)
        return self._cache


class RagasEvaluator:
    """Handles evaluation of RAG responses using Ragas metrics"""

    def __init__(self, config: RagasEvaluationConfig, cache_dir: str = "ragas_cache"):
        """
        Initialize the metrics evaluator

        Args:
            config: Configuration for metrics evaluation
            cache_dir: Directory for storing cache
        """
        self.config = config
        self.cache_manager = CacheManager(cache_dir)
        self.llm = self._setup_llm()
        self.embeddings = self._setup_embeddings()
        self.metrics = self._setup_metrics()

    def _setup_llm(self) -> LangchainLLMWrapper:
        """Initialize the language model for evaluation"""
        return RagasLLMFactory.create_llm(
            model=self.config.llm_model,
            api_key=self.config.llm_api_key,
            api_base=self.config.llm_api_base,
            temperature=self.config.llm_temperature,
            top_p=self.config.llm_top_p,
            timeout=self.config.llm_timeout,
            max_tokens=self.config.llm_max_tokens,
            cache=self.cache_manager.cache,
        )

    def _setup_embeddings(self) -> LangchainEmbeddingsWrapper:
        """Initialize the embeddings model"""
        return RagasEmbeddingFactory.create_embeddings(
            model=self.config.embeddings_model,
            api_key=self.config.embeddings_api_key,
            api_base=self.config.embeddings_api_base,
            timeout=self.config.embeddings_timeout,
            cache=self.cache_manager.cache,
        )

    def _setup_metrics(self) -> List[Any]:
        """Initialize the evaluation metrics"""
        return RagasMetricsFactory.create_metrics(self.config.metrics)

    def _create_run_config(self) -> RunConfig:
        """Create Ragas run configuration"""
        return RunConfig(
            timeout=self.config.eval_timeout,
            max_workers=self.config.max_workers,
        )

    def evaluate_dataset(self, dataset: EvaluationDataset) -> Dict[str, float]:
        """
        Evaluate a dataset using Ragas metrics

        Args:
            dataset: A Ragas-compatible dataset

        Returns:
            Dict containing scores for each metric {metric_name: score}
        """
        run_config = self._create_run_config()
        return evaluate(
            llm=self.llm,
            embeddings=self.embeddings,
            dataset=dataset,
            metrics=self.metrics,
            run_config=run_config,
            batch_size=self.config.batch_size,
            raise_exceptions=False,
        )
