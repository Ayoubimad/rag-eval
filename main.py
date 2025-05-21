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
)

from tqdm import tqdm

from chunking import (
    RecursiveChunker,
    AgenticChunker,
    SemanticChunker,
    SDPMChunker,
    CharacterChunker,
    ChonkieEmbeddings,
)
from langchain_openai import ChatOpenAI
from executor import run_in_executor
from concurrent.futures import ThreadPoolExecutor
from ragas_eval import RagasEvaluationConfig, RagasEvaluator
from utils import load_dataset, transform_to_ragas_dataset
from dotenv import load_dotenv

logger = get_logger(__name__)

load_dotenv()


def clean_text(text: str) -> str:
    """Normalize whitespace in text.

    Args:
        text: The text to clean

    Returns:
        Text cleaned of non-ascii characters, base64 images, and normalized whitespace
    """
    # Remove base64 images
    cleaned_text = re.sub(r"!\[.*?\]\(data:image/[^;]*;base64,[^)]*\)", "", text)
    # Remove emojis while preserving mathematical symbols and other useful unicode
    cleaned_text = re.sub(r"[\U0001F300-\U0001F9FF]", "", cleaned_text)
    # Remove formula-not-decoded comments
    cleaned_text = re.sub(r"<!-- formula-not-decoded -->", "", cleaned_text)
    return cleaned_text


async def main():

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

    client = R2RClient(base_url="http://localhost:7272", timeout=3600)

    hybrid_search_settings = HybridSearchSettings(
        full_text_weight=1,
        semantic_weight=5,
        full_text_limit=200,
        rrf_k=60,
    )

    graph_settings = GraphSearchSettings(
        enabled=False,
    )

    rag_generation_config = GenerationConfig(
        model=RAG_GENERATION_MODEL,
        api_base=RAG_GENERATION_API_BASE,
        temperature=0.8,
        max_tokens_to_sample=8192,
        stream=False,
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
        llm_temperature=0.1,
        llm_max_tokens=4096,
        llm_top_p=1,
        llm_timeout=3600,
        embeddings_model=EMBEDDINGS_MODEL,
        embeddings_api_key=EMBEDDINGS_API_KEY,
        embeddings_api_base=EMBEDDINGS_API_BASE,
        embeddings_timeout=3600,
        batch_size=500,
        max_workers=32,
        eval_timeout=3600,
    )

    configs = {
        "semantic_search": {
            "search_settings": SearchSettings(
                limit=5,
                use_semantic_search=True,
                graph_settings=graph_settings,
            ),
            "search_mode": "custom",
        },
        "hybrid_search": {
            "search_settings": SearchSettings(
                use_hybrid_search=True,
                use_semantic_search=True,
                use_fulltext_search=True,
                hybrid_settings=hybrid_search_settings,
                limit=5,
                graph_settings=graph_settings,
            ),
            "search_mode": "custom",
        },
        "rag_fusion": {
            "search_settings": SearchSettings(
                search_strategy="rag_fusion",
                use_hybrid_search=True,
                use_semantic_search=True,
                use_fulltext_search=True,
                hybrid_settings=hybrid_search_settings,
                limit=5,
                graph_settings=graph_settings,
            ),
            "search_mode": "custom",
        },
        "hyde": {
            "search_settings": SearchSettings(
                search_strategy="hyde",
                use_hybrid_search=True,
                use_semantic_search=True,
                use_fulltext_search=True,
                limit=5,
                hybrid_settings=hybrid_search_settings,
                graph_settings=graph_settings,
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

    for chunker_name, chunker in chunkers.items():

        ingestion_thread_pool = ThreadPoolExecutor(max_workers=16)
        search_thread_pool = ThreadPoolExecutor(max_workers=16)

        logger.info("Testing chunker: %s", chunker_name)

        logger.info("Deleting all documents from R2R...")
        await run_in_executor(None, client.delete_all_documents)

        async def process_and_ingest_file(file):
            def read_and_chunk():
                try:
                    with open(f"{dir_path}/{file}", "r") as f:
                        text = f.read()
                    chunks = chunker.chunk(text=text, clean_function=clean_text)
                    if chunks:
                        client.ingest_chunks(chunks)
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

        ingestion_thread_pool.shutdown(wait=True)

        logger.info(
            "\nProcessed and ingested %d chunks from %d files", total_chunks, len(files)
        )

        for strategy_name, strategy_config in configs.items():
            logger.info(
                "\nTesting %s strategy with %s chunker and search settings: %s",
                strategy_name,
                chunker_name,
                strategy_config["search_settings"].to_dict(),
            )

            async def process_query(user_input):
                r2r_response = await run_in_executor(
                    search_thread_pool,
                    client.process_rag_query,
                    user_input,
                    rag_generation_config,
                    strategy_config["search_settings"],
                    strategy_config["search_mode"],
                )
                return r2r_response

            tasks = [process_query(user_input) for user_input in user_inputs]
            r2r_responses = []
            for task in tqdm(
                asyncio.as_completed(tasks),
                total=len(tasks),
                desc=f"Processing queries with {strategy_name}",
            ):
                r2r_responses.append(await task)

            search_thread_pool.shutdown(wait=True)

            ragas_eval_dataset = transform_to_ragas_dataset(
                user_inputs=user_inputs,
                r2r_responses=r2r_responses,
                references=references,
                reference_contexts=reference_contexts,
            )

            logger.info(
                "\nEvaluation results for %s + %s:", chunker_name, strategy_name
            )
            results = evaluator.evaluate_dataset(ragas_eval_dataset)
            logger.info("%s", results)


if __name__ == "__main__":
    asyncio.run(main())
