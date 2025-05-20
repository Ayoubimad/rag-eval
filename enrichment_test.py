from langchain_openai import ChatOpenAI
from enrichment import MetadataEnrichment, ContextualEnrichment, HybridEnrichment
from chunking import CharacterChunker

from utils import get_logger

logger = get_logger(__name__)
logger.setLevel("DEBUG")


if __name__ == "__main__":

    llm = ChatOpenAI(
        model="ISTA-DASLab/gemma-3-4b-it-GPTQ-4b-128g",
        temperature=0.0,
        api_key="random_key",
        base_url="http://172.18.21.136:8000/v1",
    )

    metadata_enrichment = MetadataEnrichment(
        llm=llm,
        include_entities=True,
        include_keywords=True,
        include_topic=True,
        max_entities=5,
        max_keywords=5,
        show_progress_bar=True,
        max_concurrency=32,
    )

    contextual_enrichment = ContextualEnrichment(
        llm=llm,
        n_chunks=2,
        max_concurrency=32,
        show_progress_bar=True,
    )

    hybrid_enrichment = HybridEnrichment(
        llm=llm,
        n_chunks=2,
        include_entities=True,
        include_keywords=True,
        include_topic=True,
        max_entities=5,
        max_keywords=5,
        metadata_first=False,
        show_progress_bar=True,
    )

    chunker = CharacterChunker(
        chunk_size=1024,
        chunk_overlap=250,
    )

    with open("./data/arxiv/arxiv_papers_md_cleaned/2504.07089v1.md", "r") as file:
        text = file.read()

    chunks = chunker.chunk(text)
    third_chunk = chunks[2]
    print("Original Chunk:")
    print(third_chunk)

    all_enriched_contextual = contextual_enrichment.enrich_chunks(chunks)
    print("\nContextual Enrichment:")
    print(all_enriched_contextual[2])

    all_enriched_chunks = metadata_enrichment.enrich_chunks(chunks)
    print("Metadata Enrichment:")
    print(all_enriched_chunks[2])

    all_enriched_hybrid = hybrid_enrichment.enrich_chunks(chunks)
    print("\nHybrid Enrichment:")
    print(all_enriched_hybrid[2])
