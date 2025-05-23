import os
import asyncio
import csv
import copy
import time
import yaml
import sys
import re
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


def load_config(config_path):
    """
    Load configuration from a YAML file.
    Replace environment variable placeholders with actual values.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        dict: Loaded and processed configuration
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Process environment variable substitutions
    def process_env_vars(item):
        if isinstance(item, str):
            # Substitute environment variables ${ENV_VAR}
            pattern = r"\$\{([^}]+)\}"
            matches = re.findall(pattern, item)
            for match in matches:
                if match == "CPU_COUNT":
                    env_value = os.cpu_count() * 2
                else:
                    env_value = os.getenv(match)
                if env_value is not None:
                    item = item.replace(f"${{{match}}}", str(env_value))

            # Try to convert numeric strings to their appropriate types
            if item.isdigit():
                return int(item)
            try:
                float_val = float(item)
                # Check if it's an integer in float form
                if float_val.is_integer():
                    return int(float_val)
                return float_val
            except ValueError:
                return item
        elif isinstance(item, dict):
            return {k: process_env_vars(v) for k, v in item.items()}
        elif isinstance(item, list):
            return [process_env_vars(i) for i in item]
        else:
            return item

    return process_env_vars(config)


def create_search_configs(config_data):
    """
    Create search configurations from config data

    Args:
        config_data: Configuration data from YAML

    Returns:
        dict: Search configurations
    """
    search_configs = {}

    for name, config in config_data["search_configs"].items():
        search_settings_dict = config["search_settings"]

        # Convert nested dictionaries if present
        if "hybrid_settings" in search_settings_dict and isinstance(
            search_settings_dict["hybrid_settings"], dict
        ):
            search_settings_dict["hybrid_settings"] = HybridSearchSettings(
                **search_settings_dict["hybrid_settings"]
            ).to_dict()

        search_configs[name] = {
            "search_settings": SearchSettings(**search_settings_dict),
            "search_mode": config["search_mode"],
        }

    return search_configs


def create_chunkers(config_data, embedding_model):
    """
    Create chunker instances from config data

    Args:
        config_data: Configuration data from YAML
        embedding_model: Embedding model to use for semantic chunkers

    Returns:
        dict: Chunker instances
    """
    chunkers = {}

    for name, config in config_data["chunkers"].items():
        chunker_type = config["type"]
        params = config["params"]

        if chunker_type == "CharacterChunker":
            chunkers[name] = CharacterChunker(**params)
        elif chunker_type == "SemanticChunker":
            chunkers[name] = SemanticChunker(embedding_model=embedding_model, **params)
        elif chunker_type == "SDPMChunker":
            chunkers[name] = SDPMChunker(embedding_model=embedding_model, **params)
        elif chunker_type == "AgenticChunker":
            # Agentic chunker needs an LLM
            chunkers[name] = AgenticChunker(model=_llm, **params)

    return chunkers


def create_enrichment_configs(config_data, llm):
    """
    Create enrichment configurations from config data

    Args:
        config_data: Configuration data from YAML
        llm: LLM model to use for enrichment

    Returns:
        dict: Enrichment configurations
    """
    enrichment_configs = {}

    for name, config in config_data["enrichments"].items():
        enrichment_type = config["type"]
        params = config["params"]

        if enrichment_type == "MetadataEnrichment":
            enrichment_configs[name] = MetadataEnrichment(llm=llm, **params)
        elif enrichment_type == "ContextualEnrichment":
            enrichment_configs[name] = ContextualEnrichment(llm=llm, **params)
        elif enrichment_type == "HybridEnrichment":
            enrichment_configs[name] = HybridEnrichment(llm=llm, **params)

    return enrichment_configs


def grid_search(
    search_configs: dict[str, dict],
    chunkers: dict[str, ChunkingStrategy],
    enrichment_configs: dict[str, ChunkEnrichmentStrategy],
    ragas_evaluation_config: RagasEvaluationConfig,
    rag_generation_config: GenerationConfig,
    dataset_path: str,
    file_dir_path: str,
    output_csv_path: str = "grid_search_results.csv",
    r2r_base_url: str = "http://localhost:7272",
    r2r_timeout: int = 604800,
    ingestion_thread_pool_workers: int = 16,
    search_thread_pool_workers: int = 16,
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
        r2r_base_url: Base URL for R2R client
        r2r_timeout: Timeout for R2R client
        ingestion_thread_pool_workers: Number of workers for ingestion thread pool
        search_thread_pool_workers: Number of workers for search thread pool
    """

    client = R2RClient(base_url=r2r_base_url, timeout=r2r_timeout)

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
    ingestion_thread_pool = ThreadPoolExecutor(
        max_workers=ingestion_thread_pool_workers
    )
    search_thread_pool = ThreadPoolExecutor(max_workers=search_thread_pool_workers)

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


def run_search_from_config(config_path):
    """Run search based on a YAML configuration file"""
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        return

    logger.info(f"Loading configuration from {config_path}")
    config = load_config(config_path)

    # Setup global LLM for use in chunkers and enrichment
    global _llm
    _llm = ChatOpenAI(
        model=config["llm"]["model"],
        api_key=config["llm"]["api_key"],
        base_url=config["llm"]["base_url"],
    )

    # Test LLM connection
    test_response = _llm.invoke("Say hello!")
    logger.info("LLM test response: %s", test_response.content)

    # Create embeddings model
    chonkie_embeddings = ChonkieEmbeddings(
        model=config["embeddings"]["model"],
        api_key=config["embeddings"]["api_key"],
        base_url=config["embeddings"]["base_url"],
        embedding_dimension=config["embeddings"]["embedding_dimension"],
    )

    # Create RAG generation config
    rag_generation_config = GenerationConfig(
        model=config["rag_generation"]["model"],
        api_base=config["rag_generation"]["api_base"],
        temperature=config["rag_generation"]["temperature"],
        max_tokens=config["rag_generation"]["max_tokens"],
        stream=config["rag_generation"]["stream"],
    )

    # Create RAGAS evaluation config
    ragas_evaluation_config = RagasEvaluationConfig(**config["ragas_evaluation"])

    # Create search configurations
    search_configs = create_search_configs(config)

    # Create chunkers
    chunkers = create_chunkers(config, chonkie_embeddings)

    # Create enrichment configurations
    enrichment_configs = create_enrichment_configs(config, _llm)

    logger.info(
        "Starting grid search across all parameter combinations from configuration"
    )
    result_csv = grid_search(
        search_configs=search_configs,
        chunkers=chunkers,
        enrichment_configs=enrichment_configs,
        ragas_evaluation_config=ragas_evaluation_config,
        rag_generation_config=rag_generation_config,
        dataset_path=config["base"]["dataset_path"],
        file_dir_path=config["base"]["dir_path"],
        output_csv_path=config["base"]["output_csv"],
        r2r_base_url=config["base"]["r2r_base_url"],
        r2r_timeout=config["base"]["r2r_timeout"],
        ingestion_thread_pool_workers=config["base"]["ingestion_thread_pool_workers"],
        search_thread_pool_workers=config["base"]["search_thread_pool_workers"],
    )

    logger.info(f"Grid search completed. Results saved to: {result_csv}")


if __name__ == "__main__":
    # Allow specifying a config file as a command line argument
    # Default to grid_search_config.yaml if not specified
    config_path = sys.argv[1] if len(sys.argv) > 1 else "grid_search_config.yaml"
    asyncio.run(run_search_from_config(config_path))
