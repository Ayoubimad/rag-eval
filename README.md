# Comparative Analysis of RAG Components: Chunking Strategies, Retrieval Methods, and Enrichment Strategies 

Thesis Project on Retrieval-Augmented Generation Systems

## Introduction

This research project presents a comprehensive empirical study of Retrieval-Augmented Generation (RAG) systems, focusing on three critical components:
1. Document chunking strategies and their impact on retrieval effectiveness
2. Comparative analysis of retrieval methodologies
3. Chunking enrichment strategies and their impact on retrieval effectiveness

## Research Components

### Chunking Strategies

- **Agentic**: Uses an agent-based approach to determine chunk boundaries based on semantic or structural cues in the text.
- **Character**: Splits documents into chunks of a fixed number of characters, regardless of sentence or paragraph boundaries.
- **Recursive**: Recursively splits text using a hierarchy of separators (e.g., paragraphs, sentences, words) to create coherent chunks.
- **SDPM (Semantic Double Pass Merging)**: Uses a two-pass approach to merge text segments based on semantic similarity, optimizing chunk boundaries for coherence and retrieval effectiveness.
- **Semantic**: Divides text into chunks based on semantic similarity or topic coherence, using embeddings or clustering.

### Enrichment Strategies

- **Contextual Enrichment**: Enhances each chunk by incorporating relevant information from surrounding chunks, making the chunk more self-contained and improving its retrievability.
- **Metadata Enrichment**: Adds metadata such as entities, keywords, and document topics to each chunk, supporting more effective retrieval and filtering.
- **Hybrid Enrichment**: Combines multiple enrichment approaches (e.g., contextual and metadata) to maximize the informativeness and utility of each chunk for downstream RAG tasks.

### Retrieval Approaches

- **Semantic Search**: Retrieves relevant chunks using dense vector representations (embeddings) to capture semantic similarity between queries and documents.
- **Hybrid Search**: Combines semantic search with traditional keyword-based (sparse) retrieval to leverage both lexical and semantic signals for improved accuracy.
- **RAG Fusion**: Generates multiple subqueries for each user query, retrieves results for each subquery, and aggregates the results from multiple retrieval runs to enhance diversity and robustness in retrieved contexts.
- **HyDE (Hypothetical Document Embeddings)**: Generates hypothetical answers or documents for a query, embeds them, and uses these embeddings to retrieve supporting evidence from the corpus.
- **Combinations**: Integrates two or more of the above approaches (e.g., Hybrid + RAG Fusion, Semantic + HyDE) to further boost retrieval effectiveness and coverage.

### Evaluation Framework

This project uses the **RAGAS** framework for evaluation. RAGAS provides automated metrics for assessing RAG pipelines, including:

- **Faithfulness**: Measures how accurately the generated answer reflects the information present in the retrieved context.
- **Context Precision**: Indicates the proportion of retrieved context that is actually relevant to answering the query. High precision means most of the retrieved information is useful and on-topic.
- **Context Recall**: Indicates how much of the relevant information needed to answer the query was actually retrieved. High recall means the system successfully found most or all of the information required for a complete answer.

### Technology Stack

This project utilizes **R2R** as the underlying RAG system, which provides the backend for document ingestion, retrieval, and generation.

## ⚠️ Warnings & Known Issues

- **Graph Extraction Performance**: Ingesting chunks with graph extraction enabled can take an extremely long time (potentially days for large document collections). This is a limitation of the R2R backend and current optimization techniques.

- **Memory Requirements**: Processing large datasets may require significant RAM, especially when using semantic chunkers and embeddings.

- **API Rate Limits**: Be mindful of API rate limits when using external LLM providers like OpenAI. The evaluation process makes numerous API calls.

## Example Results

After running the evaluation, you'll see output similar to:

```
Evaluation results for semantic + hybrid_search:
Faithfulness: 0.89, Context Precision: 0.76, Context Recall: 0.82

Evaluation results for sdpm + rag_fusion:
Faithfulness: 0.91, Context Precision: 0.79, Context Recall: 0.85
```

These metrics help you understand the effectiveness of different chunking and retrieval combinations.

## Installation & Setup

### Prerequisites

- Python 3.9+ 
- R2R instance running locally or accessible via API
- OpenAI API key or compatible LLM API access

### Installation Steps

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/rag-eval.git
   cd rag-eval
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your environment variables by creating a `.env` file in the project root.

## Configuration & Usage

### Environment Variables

Create a `.env` file in the project root with the following variables:

```
# LLM Configuration
LLM_MODEL=gpt-4o                   # or any compatible model
LLM_API_KEY=your_openai_api_key    # Your OpenAI API key
LLM_API_BASE=https://api.openai.com/v1  # or your custom API endpoint

# Embeddings Configuration
EMBEDDINGS_MODEL=text-embedding-3-large   # or any compatible embedding model
EMBEDDINGS_API_KEY=your_openai_api_key
EMBEDDINGS_API_BASE=https://api.openai.com/v1

# RAG Generation Configuration
RAG_GENERATION_MODEL=<provider>/gpt-4o      # this must follow R2R naming convention, for instance `openai/gpt-4o`
RAG_GENERATION_API_BASE=https://api.openai.com/v1
RAG_GENERATION_API_KEY=your_openai_api_key # this must be configured in the R2R backend instance
```

### R2R Instance Setup

This project requires an R2R instance to be running. R2R is a RAG system that provides the backend for document ingestion, retrieval, and generation.

1. Set up R2R according to its documentation.
2. Make sure R2R is accessible at `http://localhost:7272` (default) or update the URL in `main.py`:

```python
client = R2RClient(
    base_url="http://localhost:7272", timeout=60
)
```

### Running the Main Evaluation Script

The main script `main.py` provides a comprehensive framework for evaluating different chunking strategies and retrieval approaches:

```bash
python main.py
```

The script will:
1. Process and ingest documents using the specified chunking strategies
2. Apply different retrieval methods
3. Evaluate results using the RAGAS framework

### Customizing Evaluation Parameters

You can customize the evaluation by modifying the parameters in `main.py`:

```python
# Control graph extraction (warning: this can be very slow)
enable_graph_rag = False  # Set to True to enable entity extraction and graph RAG

# Configure chunking strategies
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
    # ... other chunkers
}

# Configure retrieval strategies
configs = {
    "semantic_search": {
        "search_settings": SearchSettings(
            limit=5,
            use_semantic_search=True,
            use_hybrid_search=False,
            # ... other settings
        ),
        "search_mode": "custom",
    },
    # ... other retrieval methods
}
```

### Adding New Documents

Place your markdown documents in the `./data/<your_document_md>/` directory (or create your own directory and modify the `dir_path` variable in `main.py`). For example:

```
data/
└── your_documents_md/
    ├── document1.md
    ├── document2.md
    └── document3.md
```

### Test Dataset

The evaluation requires a test dataset in JSON format, You can create your own test dataset following this schema:

```json
{
    "user_inputs": [
        "What is the definition of RAG?",
        "Explain the benefits of semantic chunking"
    ],
    "reference": [
        "RAG (Retrieval-Augmented Generation) is a technique that combines retrieval of relevant documents with language model generation to produce more accurate and factual responses.",
        "Semantic chunking improves retrieval by creating coherent chunks based on meaning rather than arbitrary length, leading to better context preservation."
    ],
    "reference_contexts": [
        [
            "RAG, or Retrieval-Augmented Generation, is an AI framework that enhances language model outputs by retrieving relevant information from a knowledge base before generating responses.",
            "This approach improves accuracy and grounds the model's outputs in specific source documents."
        ],
        [
            "Semantic chunking is a document splitting strategy that analyzes the content's meaning to create logically coherent segments.",
            "Unlike fixed-size chunking, it preserves semantic relationships and contextual integrity, resulting in more effective information retrieval."
        ]
    ]
}
```
