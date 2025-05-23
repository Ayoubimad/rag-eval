import os
import asyncio
import re
from utils import get_logger

from r2r_client import (
    R2RClient,
    GenerationConfig,
    SearchSettings,
    HybridSearchSettings,
    GraphSearchSettings,
    GraphCreationSettings,
)

from tqdm import tqdm

from chunking import (
    RecursiveChunker,
    AgenticChunker,
    SemanticChunker,
    SDPMChunker,
    CharacterChunker,
    ChonkieEmbeddings,
    ChunkingStrategy,
)

from enrichment import (
    MetadataEnrichment,
    ContextualEnrichment,
    HybridEnrichment,
    ChunkEnrichmentStrategy,
)

from langchain_openai import ChatOpenAI
from executor import run_in_executor
from concurrent.futures import ThreadPoolExecutor
from ragas_eval import RagasEvaluationConfig, RagasEvaluator
from utils import load_dataset, transform_to_ragas_dataset, clean_text
from dotenv import load_dotenv

logger = get_logger(__name__)

load_dotenv()


async def main():

    enable_graph_rag = False  # Parameter to control entity extraction and graph RAG
    enable_enrichment = False  # Parameter to control chunk enrichment

    LLM_MODEL = os.getenv("LLM_MODEL")
    LLM_API_KEY = os.getenv("LLM_API_KEY")
    LLM_API_BASE = os.getenv("LLM_API_BASE")
    EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL")
    EMBEDDINGS_API_KEY = os.getenv("EMBEDDINGS_API_KEY")
    EMBEDDINGS_API_BASE = os.getenv("EMBEDDINGS_API_BASE")
    RAG_GENERATION_MODEL = os.getenv("RAG_GENERATION_MODEL")
    RAG_GENERATION_API_BASE = os.getenv("RAG_GENERATION_API_BASE")
    RAG_GENERATION_API_KEY = os.getenv(
        "RAG_GENERATION_API_KEY"
    )  # Not needed is set in r2r backend

    logger.info(
        "Configuration loaded:\n"
        "LLM_MODEL: %s\n"
        "LLM_API_KEY: %s\n"
        "LLM_API_BASE: %s\n"
        "EMBEDDINGS_MODEL: %s\n"
        "EMBEDDINGS_API_KEY: %s\n"
        "EMBEDDINGS_API_BASE: %s\n"
        "RAG_GENERATION_MODEL: %s\n"
        "RAG_GENERATION_API_BASE: %s\n"
        "RAG_GENERATION_API_KEY: %s",
        LLM_MODEL,
        LLM_API_KEY,
        LLM_API_BASE,
        EMBEDDINGS_MODEL,
        EMBEDDINGS_API_KEY,
        EMBEDDINGS_API_BASE,
        RAG_GENERATION_MODEL,
        RAG_GENERATION_API_BASE,
        RAG_GENERATION_API_KEY,
    )

    dataset_path = "/home/e4user/rag-eval/data/datasets/ragas_testset_tesi.json"
    dir_path = "/home/e4user/rag-eval/data/Grasselli_md"

    client = R2RClient(
        base_url="http://localhost:7272", timeout=604800
    )  # 1 week timeout (7*24*60*60 seconds)

    graph_settings = GraphSearchSettings(
        enabled=enable_graph_rag,
    )

    hybrid_search_settings = HybridSearchSettings(
        full_text_weight=0.3,
        semantic_weight=0.7,
        full_text_limit=200,
        rrf_k=60,
    )

    rag_generation_config = GenerationConfig(
        model=RAG_GENERATION_MODEL,
        api_base=RAG_GENERATION_API_BASE,
        temperature=0.8,
        max_tokens=8192,
        stream=False,
    )

    graph_creation_config = GraphCreationSettings(
        generation_config=rag_generation_config,
    )

    chonkie_embeddings = ChonkieEmbeddings(
        model=EMBEDDINGS_MODEL,
        api_key=EMBEDDINGS_API_KEY,
        base_url=EMBEDDINGS_API_BASE,
        embedding_dimension=1024,
    )

    llm = ChatOpenAI(
        model=LLM_MODEL,
        api_key=LLM_API_KEY,
        base_url=LLM_API_BASE,
    )

    test_response = llm.invoke("Say hello!")
    logger.info("LLM test response: %s", test_response.content)

    ragas_evaluation_config = RagasEvaluationConfig(
        llm_model=LLM_MODEL,
        llm_api_key=LLM_API_KEY,
        llm_api_base=LLM_API_BASE,
        llm_max_tokens=4096,
        llm_temperature=0.1,
        llm_timeout=240,
        llm_top_p=1,
        embeddings_timeout=240,
        embeddings_model=EMBEDDINGS_MODEL,
        embeddings_api_key=EMBEDDINGS_API_KEY,
        embeddings_api_base=EMBEDDINGS_API_BASE,
        eval_timeout=3600,
        cache_dir="ragas_cache",
        batch_size=500,
        max_workers=os.cpu_count() * 2,
        metrics=["faithfulness", "context_precision", "context_recall"],
    )

    enrichment_configs = {
        "metadata_enrichment": MetadataEnrichment(
            llm=llm,
            include_entities=True,
            include_keywords=True,
            include_topic=True,
            max_entities=15,
            max_keywords=15,
            max_concurrency=32,
            show_progress_bar=True,
        ),
        "contextual_enrichment": ContextualEnrichment(
            llm=llm, n_chunks=2, max_concurrency=32, show_progress_bar=True
        ),
        "hybrid_enrichment": HybridEnrichment(
            llm=llm,
            n_chunks=2,
            include_entities=True,
            include_keywords=True,
            include_topic=True,
            max_entities=15,
            max_keywords=15,
            metadata_first=False,
            show_progress_bar=True,
        ),
    }

    configs = {
        "semantic_search": {
            "search_settings": SearchSettings(
                limit=5,
                use_semantic_search=True,
                use_hybrid_search=False,
                use_fulltext_search=False,
                graph_settings=graph_settings.to_dict(),
            ),
            "search_mode": "custom",
        },
        "hybrid_search": {
            "search_settings": SearchSettings(
                use_hybrid_search=True,
                use_semantic_search=True,
                use_fulltext_search=True,
                hybrid_settings=hybrid_search_settings.to_dict(),
                limit=5,
                graph_settings=graph_settings.to_dict(),
            ),
            "search_mode": "custom",
        },
        "rag_fusion": {
            "search_settings": SearchSettings(
                search_strategy="rag_fusion",
                use_hybrid_search=False,
                use_semantic_search=True,
                use_fulltext_search=False,
                hybrid_settings=hybrid_search_settings.to_dict(),
                limit=5,
                graph_settings=graph_settings.to_dict(),
            ),
            "search_mode": "custom",
        },
        "hyde": {
            "search_settings": SearchSettings(
                search_strategy="hyde",
                use_hybrid_search=False,
                use_semantic_search=True,
                use_fulltext_search=False,
                limit=5,
                hybrid_settings=hybrid_search_settings.to_dict(),
                graph_settings=graph_settings.to_dict(),
            ),
            "search_mode": "custom",
        },
    }

    chunkers = {
        # "recursive": RecursiveChunker(
        #     chunk_size=256,
        # ),
        "character": CharacterChunker(
            chunk_size=1024,
            chunk_overlap=256,
        ),
        "semantic": SemanticChunker(
            embedding_model=chonkie_embeddings,
            chunk_size=256,
            threshold=0.5,
        ),
        "sdpm": SDPMChunker(
            embedding_model=chonkie_embeddings,
            chunk_size=256,
            threshold=0.5,
        ),
        "agentic": AgenticChunker(
            max_chunk_size=1024,
            model=llm,
        ),
    }

    files = os.listdir(dir_path)
    evaluator = RagasEvaluator(ragas_evaluation_config)
    user_inputs, references, reference_contexts = load_dataset(
        dataset_path=dataset_path
    )
    logger.info(
        "Loaded dataset with %d user inputs, %d references, and %d reference contexts",
        len(user_inputs),
        len(references),
        len(reference_contexts),
    )

    ingestion_thread_pool = ThreadPoolExecutor(max_workers=16)
    search_thread_pool = ThreadPoolExecutor(max_workers=16)

    for chunker_name, chunker in chunkers.items():

        logger.info("Testing chunker: %s", chunker_name)

        logger.info("Deleting all documents from R2R...")
        # await run_in_executor(None, client.delete_all_documents)

        logger.info("Resetting graph...")
        default_collection_id = await run_in_executor(
            None, client.get_default_collection_id
        )
        await run_in_executor(None, client.graph_reset, default_collection_id)

        async def process_and_ingest_file(file):
            def read_and_chunk():
                try:
                    with open(f"{dir_path}/{file}", "r") as f:
                        text = f.read()
                    chunks = chunker.chunk(text=text, clean_function=clean_text)
                    if chunks:
                        client.ingest_chunks(
                            chunks,
                            extract_entities=enable_graph_rag,
                            graph_creation_config=(
                                graph_creation_config if enable_graph_rag else None
                            ),
                        )
                    return len(chunks) if chunks else 0
                except Exception as e:
                    logger.warning("Failed to process file %s: %s", file, e)
                    return 0

            return await run_in_executor(ingestion_thread_pool, read_and_chunk)

        tasks = [process_and_ingest_file(file) for file in files]

        if enable_graph_rag:
            logger.info(
                "Ingesting chunks with graph extraction enabled, this may take a while... \n even days for a large number of documents, I don't know how to speed it up."
            )
        else:
            logger.info("Ingesting chunks with graph extraction disabled")

        total_chunks = 0
        for task in tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc=f"Processing and ingesting documents with {chunker_name}",
        ):
            num_chunks = await task
            total_chunks += num_chunks

        logger.info(
            "\nProcessed and ingested %d chunks from %d files", total_chunks, len(files)
        )

        if enable_graph_rag:
            logger.info("Building graph for knowledge base...")
            default_collection_id = await run_in_executor(
                None, client.get_default_collection_id
            )

            logger.info("Pulling entities to graph...")
            await run_in_executor(None, client.graph_pull, default_collection_id)

            logger.info("Building communities...")
            await run_in_executor(
                None, client.graph_build_communities, default_collection_id
            )

            logger.info("Graph building completed")

        for strategy_name, strategy_config in configs.items():
            logger.info(
                "\nTesting %s strategy with %s chunker and search settings: %s",
                strategy_name,
                chunker_name,
                strategy_config["search_settings"].to_dict(),
            )

            async def process_query(index, user_input):
                r2r_response = await run_in_executor(
                    search_thread_pool,
                    client.process_rag_query,
                    user_input,
                    rag_generation_config,
                    strategy_config["search_settings"],
                    strategy_config["search_mode"],
                )
                return index, r2r_response

            tasks = [
                process_query(i, user_input) for i, user_input in enumerate(user_inputs)
            ]
            results = []
            for task in tqdm(
                asyncio.as_completed(tasks),
                total=len(tasks),
                desc=f"Processing queries with {strategy_name}",
            ):
                index, response = await task
                results.append((index, response))

            results.sort(
                key=lambda x: x[0]
            )  # Fixed: mismatched order after 2 days, I want to cry
            r2r_responses = [response for _, response in results]

            ragas_eval_dataset = transform_to_ragas_dataset(
                user_inputs=user_inputs,
                r2r_responses=r2r_responses,
                references=references,
                reference_contexts=reference_contexts,
            )

            logger.info("Running evaluation...")
            results = evaluator.evaluate_dataset(ragas_eval_dataset)
            import ast

            literal = ast.literal_eval(str(results))
            logger.info(
                "Faithfulness: %s, Context Precision: %s, Context Recall: %s",
                literal["faithfulness"],
                literal["context_precision"],
                literal["context_recall"],
            )

    ingestion_thread_pool.shutdown(wait=True)
    search_thread_pool.shutdown(wait=True)


if __name__ == "__main__":
    asyncio.run(main())
