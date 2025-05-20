"""
Simple script to generate synthetic test datasets using RAGAS.
"""

import json
import os
import dotenv

from utils import get_logger
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from ragas import RunConfig
from ragas.testset.persona import Persona
from ragas.cache import DiskCacheBackend
from ragas.integrations.langchain import LangchainLLMWrapper, LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator
from ragas.testset.synthesizers import (
    SingleHopSpecificQuerySynthesizer,
    MultiHopAbstractQuerySynthesizer,
    MultiHopSpecificQuerySynthesizer,
)

logger = get_logger(__name__)

dotenv.load_dotenv()

DATA_DIR = "/home/e4user/rag-eval/data/arxiv/arxiv_papers_md_cleaned"
OUTPUT_PATH = "datasets/ragas_arxiv_papers_dataset.json"
TESTSET_SIZE = 1000
TIMEOUT = 60000
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL")
EMBEDDINGS_API_BASE = os.getenv("EMBEDDINGS_API_BASE")
EMBEDDINGS_API_KEY = os.getenv("EMBEDDINGS_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL")
LLM_API_BASE = os.getenv("LLM_API_BASE")
LLM_API_KEY = os.getenv("LLM_API_KEY")
CACHE_DIR = "ragas_cache"
MAX_TOKENS = 90000
TOP_P = 0.95
TEMPERATURE = 0.8

PERSONAS = [
    Persona(
        name="Researcher",
        role_description="Frequently publishes scientific papers and is looking for the latest developments in their fields.",
    ),
    Persona(
        name="Student",
        role_description="Is studying a specific subject and needs easy-to-understand explanations, summaries of complex topics, and relevant study materials.",
    ),
    Persona(
        name="Teacher",
        role_description="Seeks structured content to prepare lessons, track educational trends, and find supplementary material for students.",
    ),
    Persona(
        name="General User",
        role_description="Has no specialized background and is looking for accessible, concise information on a wide range of topics.",
    ),
]

logger.info(
    "Configuration loaded:\n"
    "DATA_DIR: %s\n"
    "OUTPUT_PATH: %s\n"
    "TESTSET_SIZE: %d\n"
    "TIMEOUT: %d\n"
    "EMBEDDINGS_MODEL: %s\n"
    "EMBEDDINGS_API_BASE: %s\n"
    "EMBEDDINGS_API_KEY: %s\n"
    "LLM_MODEL: %s\n"
    "LLM_API_BASE: %s\n"
    "LLM_API_KEY: %s\n"
    "CACHE_DIR: %s\n"
    "MAX_TOKENS: %d\n"
    "TOP_P: %.2f\n"
    "TEMPERATURE: %.2f",
    DATA_DIR,
    OUTPUT_PATH,
    TESTSET_SIZE,
    TIMEOUT,
    EMBEDDINGS_MODEL,
    EMBEDDINGS_API_BASE,
    EMBEDDINGS_API_KEY,
    LLM_MODEL,
    LLM_API_BASE,
    LLM_API_KEY,
    CACHE_DIR,
    MAX_TOKENS,
    TOP_P,
    TEMPERATURE,
)


def main():
    logger.info("Starting dataset generation...")

    logger.info("Loading documents from %s", DATA_DIR)
    loader = DirectoryLoader(DATA_DIR, glob="*.md", loader_cls=TextLoader)
    documents = loader.load()
    logger.info("Loaded %d documents", len(documents))

    logger.info("Setting up models...")
    embeddings = OpenAIEmbeddings(
        model=EMBEDDINGS_MODEL,
        api_key=EMBEDDINGS_API_KEY,
        base_url=EMBEDDINGS_API_BASE,
        timeout=TIMEOUT,
    )

    llm = ChatOpenAI(
        model=LLM_MODEL,
        api_key=LLM_API_KEY,
        base_url=LLM_API_BASE,
        timeout=TIMEOUT,
        max_tokens=MAX_TOKENS,
        top_p=TOP_P,
        temperature=TEMPERATURE,
    )

    cache = DiskCacheBackend(cache_dir=CACHE_DIR)
    llm_wrapper = LangchainLLMWrapper(llm, cache=cache)
    embeddings_wrapper = LangchainEmbeddingsWrapper(embeddings, cache=cache)

    logger.info("Generating test dataset with %d documents...", len(documents))
    generator = TestsetGenerator(
        llm=llm_wrapper, embedding_model=embeddings_wrapper, persona_list=PERSONAS
    )

    run_config = RunConfig(timeout=TIMEOUT, max_workers=os.cpu_count() * 4)

    query_distribution = [
        (SingleHopSpecificQuerySynthesizer(llm=llm_wrapper), 0.5),
        (MultiHopAbstractQuerySynthesizer(llm=llm_wrapper), 0.25),
        (MultiHopSpecificQuerySynthesizer(llm=llm_wrapper), 0.25),
    ]

    dataset = generator.generate_with_langchain_docs(
        documents=documents,
        query_distribution=query_distribution,
        run_config=run_config,
        testset_size=TESTSET_SIZE,
        raise_exceptions=False,
        with_debugging_logs=False,
    )

    logger.info("Formatting and saving dataset to %s...", OUTPUT_PATH)
    df = dataset.to_pandas()
    ragtester_dataset = {
        "user_input": df["user_input"].tolist(),
        "reference": df["reference"].tolist(),
        "reference_contexts": df["reference_contexts"].tolist(),
    }

    output_dir = os.path.dirname(OUTPUT_PATH)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(ragtester_dataset, f, indent=2)

    logger.info("Dataset successfully generated at: %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
