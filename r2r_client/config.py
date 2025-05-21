"""
Configurations for the R2R Client
"""

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Optional, List


@dataclass
class HybridSearchSettings:
    """Settings for hybrid search combining full-text and semantic search."""

    full_text_weight: Optional[float] = None
    semantic_weight: Optional[float] = None
    full_text_limit: Optional[int] = None
    rrf_k: Optional[int] = None

    def to_dict(self) -> dict:
        d = {}
        for k, v in asdict(self).items():
            if v is not None:
                d[k] = v
        return d


@dataclass
class GraphSearchSettings:
    """Settings specific to knowledge graph search."""

    limits: Optional[Dict[str, int]] = None
    enabled: Optional[bool] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class SearchSettings:
    """Main search settings class that combines shared settings with specialized settings."""

    use_hybrid_search: Optional[bool] = None
    use_semantic_search: Optional[bool] = None
    use_fulltext_search: Optional[bool] = None
    filters: Optional[Dict[str, Any]] = None
    limit: Optional[int] = None
    offset: Optional[int] = None
    include_metadatas: Optional[bool] = None
    include_scores: Optional[bool] = None
    search_strategy: Optional[str] = None
    hybrid_settings: Optional[HybridSearchSettings] = None
    graph_settings: Optional[GraphSearchSettings] = None
    num_sub_queries: Optional[int] = None

    def to_dict(self) -> dict:
        result = {}
        for k, v in asdict(self).items():
            if v is not None:
                if isinstance(v, dict):
                    filtered = {dk: dv for dk, dv in v.items() if dv is not None}
                    if filtered:
                        result[k] = filtered
                elif hasattr(v, "to_dict"):
                    settings_dict = v.to_dict()
                    if settings_dict:
                        result[k] = settings_dict
                else:
                    result[k] = v
        return result


@dataclass
class GenerationConfig:
    """Default configuration for R2R generation"""

    model: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens_to_sample: Optional[int] = None
    stream: Optional[bool] = None
    functions: Optional[List[Any]] = None
    tools: Optional[List[Any]] = None
    add_generation_kwargs: Optional[List[Any]] = None
    api_base: Optional[str] = None
    response_format: Optional[str] = None
    extended_thinking: Optional[bool] = None
    thinking_budget: Optional[int] = None
    reasoning_effort: Optional[int] = None

    def to_dict(self) -> dict:
        d = {}
        for k, v in asdict(self).items():
            if v is not None:
                d[k] = v
        return d

    def __str__(self) -> str:
        return str(self.to_dict())


@dataclass
class GraphCreationSettings:
    """Settings for knowledge graph creation."""

    graph_extraction_prompt: str = "graph_extraction"
    graph_entity_description_prompt: str = "graph_entity_description"
    entity_types: List[str] = field(default_factory=list)
    relation_types: List[str] = field(default_factory=list)
    chunk_merge_count: int = 2
    max_knowledge_relationships: int = 100
    max_description_input_length: int = 65536
    generation_config: Optional[GenerationConfig] = None
    automatic_deduplication: bool = False

    def to_dict(self) -> dict:
        d = {}
        for k, v in asdict(self).items():
            if v is not None:
                if isinstance(v, (list, dict)) and not v:
                    continue
                if hasattr(v, "to_dict"):
                    settings_dict = v.to_dict()
                    if settings_dict:
                        d[k] = settings_dict
                else:
                    d[k] = v
        return d
