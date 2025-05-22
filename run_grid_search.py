import os
import asyncio
import csv
import copy
import time
from itertools import product
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from tqdm import tqdm

from utils import get_logger, load_dataset, transform_to_ragas_dataset, clean_text
from executor import run_in_executor

from r2r_client import (
    R2RClient,
    GenerationConfig,
    SearchSettings,
    HybridSearchSettings,
    GraphSearchSettings,
    GraphCreationSettings,
)

from chunking import (
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
from ragas_eval import RagasEvaluationConfig, RagasEvaluator
from evaluation_logger import EvaluationLogger

logger = get_logger(__name__)

load_dotenv()


def grid_search(
    search_configs: dict[str, dict],
    chunkers: dict[str, ChunkingStrategy],
    enrichment_configs: dict[str, ChunkEnrichmentStrategy],
    ragas_evaluation_config: RagasEvaluationConfig,
    rag_generation_config: GenerationConfig,
    dataset_path: str,
    file_dir_path: str,
    output_csv_path: str = "grid_search_results.csv",
):
    """
    Execute a comprehensive grid search across all combinations of parameters:
    - Graph RAG enabled/disabled
    - Chunk enrichment strategies (including no enrichment)
    - Chunking strategies
    - Search/retrieval configurations

    Results are saved to a CSV file after each evaluation.

    Args:
        search_configs: Dictionary of search configuration name to search settings
        chunkers: Dictionary of chunker name to chunker instance
        enrichment_configs: Dictionary of enrichment strategy name to enrichment instance
        ragas_evaluation_config: Configuration for RAGAS evaluation
        rag_generation_config: Configuration for RAG generation
        dataset_path: Path to the dataset for evaluation
        file_dir_path: Path to the directory containing files to ingest
        output_csv_path: Path to save the CSV results
    """

    client = R2RClient(
        base_url="http://localhost:7272", timeout=604800
    )  # 1 week timeout (7*24*60*60 seconds)

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

    # Create or open CSV file for writing results
    csv_exists = os.path.exists(output_csv_path)
    csv_file = open(output_csv_path, "a" if csv_exists else "w", newline="")
    csv_writer = csv.writer(csv_file)

    # Write header if the file doesn't exist
    if not csv_exists:
        csv_writer.writerow(
            [
                "timestamp",
                "chunker",
                "graph_enabled",
                "enrichment_strategy",
                "search_strategy",
                "faithfulness",
                "context_precision",
                "context_recall",
            ]
        )

    # Create evaluation logger
    eval_logger = EvaluationLogger(
        log_file_path=output_csv_path.replace(".csv", "_progress.log")
    )

    # Generate all combinations
    graph_options = [False, True]
    # Add "none" as enrichment strategy to evaluate without enrichment
    enrichment_options = ["none"] + list(enrichment_configs.keys())

    # Total combinations for progress tracking
    total_combinations = (
        len(chunkers)
        * len(graph_options)
        * len(enrichment_options)
        * len(search_configs)
    )
    current_combination = 0

    # Thread pools for parallel processing
    ingestion_thread_pool = ThreadPoolExecutor(max_workers=16)
    search_thread_pool = ThreadPoolExecutor(max_workers=16)

    files = os.listdir(file_dir_path)

    try:
        # Reordered the parameters to match the requested order:
        # 1. graph_enabled, 2. enrichment, 3. chunker, 4. strategies
        for enable_graph_rag, enrichment_name, chunker_name in product(
            graph_options, enrichment_options, chunkers.keys()
        ):
            chunker = chunkers[chunker_name]
            enable_chunk_enrichment = enrichment_name != "none"

            current_combination += 1
            logger.info(
                "\n[%d/%d] Testing combination: Graph=%s, Enrichment=%s, Chunker=%s",
                current_combination,
                total_combinations,
                enable_graph_rag,
                enrichment_name,
                chunker_name,
            )

            # Create graph creation config based on graph_enabled flag
            graph_creation_config = None
            if enable_graph_rag:
                graph_creation_config = GraphCreationSettings(
                    generation_config=rag_generation_config,
                )

            graph_settings = GraphSearchSettings(
                enabled=enable_graph_rag,
            )

            logger.info("Deleting all documents from R2R...")
            client.delete_all_documents()

            logger.info("Resetting graph...")
            default_collection_id = client.get_default_collection_id()
            client.graph_reset(default_collection_id)

            if enable_graph_rag:
                logger.info(
                    "Ingesting chunks with graph extraction enabled, this may take a while..."
                )
            else:
                logger.info("Ingesting chunks with graph extraction disabled")

            # Create async function to handle file processing and ingestion
            async def process_all_files():
                async def process_and_ingest_file(file):
                    def read_and_chunk():
                        try:
                            with open(f"{file_dir_path}/{file}", "r") as f:
                                text = f.read()
                            chunks = chunker.chunk(text=text, clean_function=clean_text)

                            if chunks:
                                # Apply enrichment if enabled and enrichment strategy is available
                                if (
                                    enable_chunk_enrichment
                                    and enrichment_name in enrichment_configs
                                ):
                                    enricher = enrichment_configs[enrichment_name]
                                    chunks = enricher.enrich_chunks(chunks)

                                client.ingest_chunks(
                                    chunks,
                                    extract_entities=enable_graph_rag,
                                    graph_creation_config=graph_creation_config,
                                )
                            return len(chunks) if chunks else 0
                        except Exception as e:
                            logger.warning("Failed to process file %s: %s", file, e)
                            return 0

                    return await run_in_executor(ingestion_thread_pool, read_and_chunk)

                tasks = [process_and_ingest_file(file) for file in files]

                total_chunks = 0
                for task in tqdm(
                    asyncio.as_completed(tasks),
                    total=len(tasks),
                    desc=f"Processing and ingesting documents with {chunker_name}",
                ):
                    num_chunks = await task
                    total_chunks += num_chunks

                return total_chunks

            # Run the async function
            total_chunks = asyncio.run(process_all_files())

            logger.info(
                "\nProcessed and ingested %d chunks from %d files",
                total_chunks,
                len(files),
            )

            if enable_graph_rag:
                logger.info("Building graph for knowledge base...")

                logger.info("Pulling entities to graph...")
                client.graph_pull(default_collection_id)

                logger.info("Building communities...")
                client.graph_build_communities(default_collection_id)

                logger.info("Graph building completed")

            # Update all search configurations with the current graph settings
            for strategy_name, strategy_config in search_configs.items():
                # Deep copy search settings to avoid modifying the original

                search_settings = copy.deepcopy(strategy_config["search_settings"])

                # Update graph settings based on current combination
                search_settings.graph_settings = graph_settings.to_dict()

                logger.info(
                    "\nTesting %s strategy with graph=%s, enrichment=%s, chunker=%s",
                    strategy_name,
                    enable_graph_rag,
                    enrichment_name,
                    chunker_name,
                )

                # Log the current evaluation type
                eval_logger.log_evaluation(
                    graph_enabled=enable_graph_rag,
                    enrichment_strategy=enrichment_name,
                    chunker_name=chunker_name,
                    search_strategy=strategy_name,
                )

                ## Query Processing Workflow
                ## ------------------------
                ## This section orchestrates the parallel processing of all user queries against the RAG system:
                ## 1. For each user query, we create an async task to process it through R2R
                ## 2. We track the original index to maintain order of results
                ## 3. Queries are processed concurrently for efficiency, with progress displayed
                ## 4. Finally, we reorder results to match the original dataset order

                # Create async function to handle query processing
                async def process_all_queries():
                    ## Process a single query with the current search strategy
                    ## The index is tracked to preserve the original order when reassembling results
                    async def process_query(index, user_input):
                        r2r_response = await run_in_executor(
                            search_thread_pool,
                            client.process_rag_query,
                            user_input,
                            rag_generation_config,
                            search_settings,
                            strategy_config["search_mode"],
                        )
                        return index, r2r_response

                    ## Create a task for each user query to enable parallel processing
                    ## This significantly improves throughput compared to sequential processing
                    tasks = [
                        process_query(i, user_input)
                        for i, user_input in enumerate(user_inputs)
                    ]

                    ## Collect results as they complete (non-deterministic order)
                    ## Using asyncio.as_completed allows us to process results as soon as they're available
                    query_results = []
                    for task in tqdm(
                        asyncio.as_completed(tasks),
                        total=len(tasks),
                        desc=f"Processing queries with {strategy_name}",
                    ):
                        index, response = await task
                        query_results.append((index, response))

                    ## Restore the original dataset order by sorting on the tracked indices
                    ## This ensures evaluation metrics align with the expected order of the dataset
                    query_results.sort(key=lambda x: x[0])  # Sort by index
                    return [response for _, response in query_results]

                # Run the async function using asyncio.run
                r2r_responses = asyncio.run(process_all_queries())

                ragas_eval_dataset = transform_to_ragas_dataset(
                    user_inputs=user_inputs,
                    r2r_responses=r2r_responses,
                    references=references,
                    reference_contexts=reference_contexts,
                )

                logger.info("Running evaluation...")
                eval_results = evaluator.evaluate_dataset(ragas_eval_dataset)
                import ast

                literal = ast.literal_eval(str(eval_results))

                # Log results
                logger.info(
                    "Results for %s + %s + graph=%s + enrichment=%s:",
                    chunker_name,
                    strategy_name,
                    enable_graph_rag,
                    enrichment_name,
                )
                logger.info(
                    "Faithfulness: %s, Context Precision: %s, Context Recall: %s",
                    literal["faithfulness"],
                    literal["context_precision"],
                    literal["context_recall"],
                )

                # Write results to CSV immediately
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                csv_writer.writerow(
                    [
                        timestamp,
                        chunker_name,
                        str(enable_graph_rag),
                        enrichment_name,
                        strategy_name,
                        literal["faithfulness"],
                        literal["context_precision"],
                        literal["context_recall"],
                    ]
                )
                csv_file.flush()

    except Exception as e:
        logger.error("Error during grid search: %s", e)
        import traceback

        logger.error(traceback.format_exc())
    finally:
        # Clean up resources
        ingestion_thread_pool.shutdown(wait=True)
        search_thread_pool.shutdown(wait=True)
        csv_file.close()
        eval_logger.finalize()  # Log completion of evaluation

    logger.info("\nGrid search completed. Results saved to %s", output_csv_path)
    return output_csv_path


async def run_search():
    LLM_MODEL = os.getenv("LLM_MODEL")
    LLM_API_KEY = os.getenv("LLM_API_KEY")
    LLM_API_BASE = os.getenv("LLM_API_BASE")
    EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL")
    EMBEDDINGS_API_KEY = os.getenv("EMBEDDINGS_API_KEY")
    EMBEDDINGS_API_BASE = os.getenv("EMBEDDINGS_API_BASE")
    RAG_GENERATION_MODEL = os.getenv("RAG_GENERATION_MODEL")
    RAG_GENERATION_API_BASE = os.getenv("RAG_GENERATION_API_BASE")
    RAG_GENERATION_API_KEY = os.getenv("RAG_GENERATION_API_KEY")

    logger.info("Starting grid search with environment configuration")

    dataset_path = "/home/e4user/rag-eval/data/datasets/ragas_testset_tesi.json"
    dir_path = "/home/e4user/rag-eval/data/Grasselli_md"
    output_csv = "/home/e4user/rag-eval/grid_search_results.csv"

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

    rag_generation_config = GenerationConfig(
        model=RAG_GENERATION_MODEL,
        api_base=RAG_GENERATION_API_BASE,
        temperature=0.8,
        max_tokens=8192,
        stream=False,
    )

    hybrid_search_settings = HybridSearchSettings(
        full_text_weight=0.3,
        semantic_weight=0.7,
        full_text_limit=200,
        rrf_k=60,
    )

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

    search_configs = {
        "semantic_search": {
            "search_settings": SearchSettings(
                limit=5,
                use_semantic_search=True,
                use_hybrid_search=False,
                use_fulltext_search=False,
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
            ),
            "search_mode": "custom",
        },
        "rag_fusion": {
            "search_settings": SearchSettings(
                search_strategy="rag_fusion",
                use_hybrid_search=False,
                use_semantic_search=True,
                use_fulltext_search=False,
                limit=5,
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
            ),
            "search_mode": "custom",
        },
        "hyde_hybrid": {
            "search_settings": SearchSettings(
                search_strategy="hyde",
                use_hybrid_search=True,
                use_semantic_search=True,
                use_fulltext_search=True,
                limit=5,
            ),
            "search_mode": "custom",
        },
        "rag_fusion_hybrid": {
            "search_settings": SearchSettings(
                search_strategy="rag_fusion",
                use_hybrid_search=True,
                use_semantic_search=True,
                use_fulltext_search=True,
                limit=5,
            ),
            "search_mode": "custom",
        },
    }

    chunkers = {
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

    logger.info("Starting grid search across all parameter combinations")
    result_csv = await grid_search(
        search_configs=search_configs,
        chunkers=chunkers,
        enrichment_configs=enrichment_configs,
        ragas_evaluation_config=ragas_evaluation_config,
        rag_generation_config=rag_generation_config,
        dataset_path=dataset_path,
        file_dir_path=dir_path,
        output_csv_path=output_csv,
    )

    logger.info(f"Grid search completed. Results saved to: {result_csv}")


if __name__ == "__main__":
    # To run this script with nohup without creating a nohup.out file, use:
    # nohup python run_grid_search.py > /dev/null 2>&1 &
    asyncio.run(run_search())
