# Comparative Analysis of RAG Components: Chunking Strategies, Retrieval Methods, and Evaluation Metrics

Thesis Project on Retrieval-Augmented Generation Systems

## Abstract

This research project presents a comprehensive empirical study of Retrieval-Augmented Generation (RAG) systems, focusing on three critical components:
1. Document chunking strategies and their impact on retrieval effectiveness
2. Comparative analysis of retrieval methodologies
3. Chunking enrichment strategies and their impact on retrieval effectiveness

### Chunking strategies:

- **Agentic**: Uses an agent-based approach to determine chunk boundaries based on semantic or structural cues in the text.
- **Character**: Splits documents into chunks of a fixed number of characters, regardless of sentence or paragraph boundaries.
- **Recursive**: Recursively splits text using a hierarchy of separators (e.g., paragraphs, sentences, words) to create coherent chunks.
- **SDPM (Semantic Double Pass Merging)**: Uses a two-pass approach to merge text segments based on semantic similarity, optimizing chunk boundaries for coherence and retrieval effectiveness.
- **Semantic**: Divides text into chunks based on semantic similarity or topic coherence, often using embeddings or clustering.

### Enrichment strategies:

- **Contextual Enrichment**: Enhances each chunk by incorporating relevant information from surrounding chunks, making the chunk more self-contained and improving its retrievability.
- **Metadata Enrichment**: Adds metadata such as entities, keywords, and document topics to each chunk, supporting more effective retrieval and filtering.
- **Hybrid Enrichment**: Combines multiple enrichment approaches (e.g., contextual and metadata) to maximize the informativeness and utility of each chunk for downstream RAG tasks.

### Retrieval approaches:

- **Semantic Search**: Retrieves relevant chunks using dense vector representations (embeddings) to capture semantic similarity between queries and documents.
- **Hybrid Search**: Combines semantic search with traditional keyword-based (sparse) retrieval to leverage both lexical and semantic signals for improved accuracy.
- **RAG Fusion**: Generates multiple subqueries for each user query, retrieves results for each subquery, and aggregates the results from multiple retrieval runs to enhance diversity and robustness in retrieved contexts.
- **HyDE (Hypothetical Document Embeddings)**: Generates hypothetical answers or documents for a query, embeds them, and uses these embeddings to retrieve supporting evidence from the corpus.
- **Combinations**: Integrates two or more of the above approaches (e.g., Hybrid + RAG Fusion, Semantic + HyDE) to further boost retrieval effectiveness and coverage.

### Evaluation framework:

This project uses the **RAGAS** framework for evaluation. RAGAS provides automated metrics for assessing RAG pipelines, including:

- **Context Precision & Recall**: Measures how accurately and completely the retrieved context supports the generated answers.
- **Testset Generation**: Supports the creation of evaluation datasets tailored for RAG systems, enabling robust and reproducible benchmarking.

