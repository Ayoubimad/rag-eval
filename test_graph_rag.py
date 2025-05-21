import re
import os
from r2r_client import R2RClient
from chunking import CharacterChunker
from dotenv import load_dotenv
from r2r_client import GraphCreationSettings, GenerationConfig

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


RAG_GENERATION_MODEL = os.getenv("RAG_GENERATION_MODEL")
RAG_GENERATION_API_BASE = os.getenv("RAG_GENERATION_API_BASE")

with open(
    "/home/e4user/rag-eval/data/arxiv/arxiv_papers_md_cleaned/2504.07092v1.md"
) as f:
    content = f.read()

chunker = CharacterChunker()
chunks = chunker.chunk(content, clean_text)

client = R2RClient(base_url="http://localhost:7272", timeout=3600)
result = client.client.collections.list(
    offset=0,
    limit=10,
)

generation_config = GenerationConfig(
    model=RAG_GENERATION_MODEL,
    temperature=0.8,
    max_tokens_to_sample=1000,
    api_base=RAG_GENERATION_API_BASE,
)  # API KEY NOT NEEDED, SET IN R2R BACKEND


graph_creation_config = GraphCreationSettings(
    generation_config=generation_config,
)

client.delete_all_documents()
response = client.client.graphs.reset(
    collection_id="122fdf6a-e116-546b-a8f6-e4cb2e2c0a09",
)

exit()

client.ingest_chunks(
    chunks, extract_entities=True, graph_creation_config=graph_creation_config
)
client.client.graphs.pull(collection_id="122fdf6a-e116-546b-a8f6-e4cb2e2c0a09")
