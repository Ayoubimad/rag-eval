"""
runs some examples of the chunking strategies, and a simple comparison between sequential and parallel processing
"""

from chunking import (
    CharacterChunker,
    SemanticChunker,
    SDPMChunker,
    AgenticChunker,
    RecursiveChunker,
    ChonkieEmbeddings,
)

import time
import re
import os
import dotenv
import asyncio
import logging
from logging import getLogger
from concurrent.futures import ThreadPoolExecutor
from executor import run_in_executor

from langchain_openai import ChatOpenAI

dotenv.load_dotenv()


class ColoredFormatter(logging.Formatter):
    """Custom formatter with different colors for different log levels"""

    grey = "\033[90m"
    blue = "\033[94m"
    green = "\033[92m"
    yellow = "\033[93m"
    red = "\033[91m"
    reset = "\033[0m"

    FORMATS = {
        logging.DEBUG: grey,
        logging.INFO: green,
        logging.WARNING: yellow,
        logging.ERROR: red,
        logging.CRITICAL: red,
    }

    def format(self, record):
        log_color = self.FORMATS.get(record.levelno, self.grey)
        formatter = logging.Formatter(
            f"{self.grey}%(asctime)s{self.reset} - {self.blue}%(name)s{self.reset} - {log_color}%(levelname)s{self.reset} - %(message)s"
        )
        return formatter.format(record)


logger = getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter())
    logger.addHandler(handler)


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


async def process_file_sequential(file, chunker, clean_text_func):
    """Process a single file sequentially"""
    with open(os.path.join(directory, file), "r") as f:
        text = f.read()
    return chunker.chunk(text, clean_text_func)


async def process_files_parallel(files, chunker, clean_text_func):
    """Process multiple files in parallel using executor"""
    thread_pool = ThreadPoolExecutor(max_workers=min(32, len(files)))

    async def process_file(file):
        def read_and_chunk():
            try:
                with open(os.path.join(directory, file), "r") as f:
                    text = f.read()
                return chunker.chunk(text, clean_text_func)
            except Exception as e:
                logger.warning("Failed to process file %s: %s", file, str(e))
                return None

        return await run_in_executor(thread_pool, read_and_chunk)

    tasks = [process_file(file) for file in files]
    results = []
    for task in asyncio.as_completed(tasks):
        result = await task
        if result:
            results.extend(result)

    thread_pool.shutdown(wait=True)
    return results


# Load configuration
LLM_MODEL = os.getenv("LLM_MODEL")
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("LLM_API_BASE")
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL")
EMBEDDINGS_API_KEY = os.getenv("EMBEDDINGS_API_KEY")
EMBEDDINGS_BASE_URL = os.getenv("EMBEDDINGS_API_BASE")

logger.info(
    "Configuration loaded:\n"
    "LLM_MODEL: %s\n"
    "LLM_API_KEY: %s\n"
    "LLM_BASE_URL: %s\n"
    "EMBEDDINGS_MODEL: %s\n"
    "EMBEDDINGS_API_KEY: %s\n"
    "EMBEDDINGS_BASE_URL: %s",
    LLM_MODEL,
    LLM_API_KEY,
    LLM_BASE_URL,
    EMBEDDINGS_MODEL,
    EMBEDDINGS_API_KEY,
    EMBEDDINGS_BASE_URL,
)

# Initialize models and chunkers
llm_model = ChatOpenAI(
    model=LLM_MODEL,
    base_url=LLM_BASE_URL,
    api_key=LLM_API_KEY,
)

character_chunker = CharacterChunker(
    chunk_size=1024,
    chunk_overlap=256,
)

semantic_chunker = SemanticChunker(
    embedding_model=ChonkieEmbeddings(
        model=EMBEDDINGS_MODEL,
        api_key=EMBEDDINGS_API_KEY,
        base_url=EMBEDDINGS_BASE_URL,
        embedding_dimension=1024,
    ),
    chunk_size=256,
    threshold=0.5,
)

sdpm_chunker = SDPMChunker(
    embedding_model=ChonkieEmbeddings(
        model=EMBEDDINGS_MODEL,
        api_key=EMBEDDINGS_API_KEY,
        base_url=EMBEDDINGS_BASE_URL,
        embedding_dimension=1024,
    ),
    chunk_size=256,
    threshold=0.5,
)

agentic_chunker = AgenticChunker(
    model=llm_model,
    max_chunk_size=1024,
)

recursive_chunker = RecursiveChunker(
    chunk_size=1024,
)

# Load test files
directory = "/home/e4user/rag-eval/data/Grasselli_md"
files = os.listdir(directory)

# ===== PART 1: Sequential vs Parallel Processing Comparison =====
logger.info("\n=== PART 1: Sequential vs Parallel Processing Comparison ===")
test_files = files

chunkers = {
    "Character": character_chunker,
    "Semantic": semantic_chunker,
    "SDPM": sdpm_chunker,
    "Agentic": agentic_chunker,
    "Recursive": recursive_chunker,
}

for chunker_name, chunker in chunkers.items():
    logger.info("\n=== Testing %s Chunker ===", chunker_name)
    logger.info("Number of files to process: %d", len(test_files))

    # Sequential processing
    start_time = time.time()
    sequential_chunks = []
    for file in test_files:
        with open(os.path.join(directory, file), "r") as f:
            text = f.read()
        try:
            chunks = chunker.chunk(text, clean_text)
            sequential_chunks.extend(chunks)
        except Exception as e:
            logger.error("Failed to process file %s sequentially: %s", file, str(e))
    end_time = time.time()
    logger.info("Sequential processing time: %.2fs", end_time - start_time)
    logger.info("Total chunks produced: %d", len(sequential_chunks))

    # Parallel processing
    start_time = time.time()
    try:
        parallel_chunks = asyncio.run(
            process_files_parallel(test_files, chunker, clean_text)
        )
        end_time = time.time()
        logger.info("Parallel processing time: %.2fs", end_time - start_time)
        logger.info("Total chunks produced: %d", len(parallel_chunks))
    except Exception as e:
        logger.error("Parallel processing failed: %s", str(e))

# ===== PART 2: Chunker Comparison Tests =====
logger.info("\n=== PART 2: Chunker Comparison Tests ===")
logger.info("Testing each chunker on individual files:")

for file in files:
    with open(os.path.join(directory, file), "r") as f:
        text = f.read()
    logger.info("\nAnalyzing document: %s", file)

    try:
        start_time = time.time()
        semantic_chunks = semantic_chunker.chunk(text, clean_text)
        end_time = time.time()
        logger.info(
            "Semantic Chunks: %d, %.2fs", len(semantic_chunks), end_time - start_time
        )
    except Exception as e:
        logger.error("Semantic Chunker failed: %s", str(e))

    try:
        start_time = time.time()
        sdpm_chunks = sdpm_chunker.chunk(text, clean_text)
        end_time = time.time()
        logger.info("SDPM Chunks: %d, %.2fs", len(sdpm_chunks), end_time - start_time)
    except Exception as e:
        logger.error("SDPM Chunker failed: %s", str(e))

    try:
        start_time = time.time()
        agentic_chunks = agentic_chunker.chunk(text, clean_text)
        end_time = time.time()
        logger.info(
            "Agentic Chunks: %d, %.2fs", len(agentic_chunks), end_time - start_time
        )
    except Exception as e:
        logger.error("Agentic Chunker failed: %s", str(e))

    try:
        start_time = time.time()
        character_chunks = character_chunker.chunk(text, clean_text)
        end_time = time.time()
        logger.info(
            "Character Chunks: %d, %.2fs", len(character_chunks), end_time - start_time
        )
    except Exception as e:
        logger.error("Character Chunker failed: %s", str(e))
