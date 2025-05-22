# Grid Search Configuration System

This system allows you to run grid searches with configurations loaded from YAML files, making it easier to manage and customize your RAG evaluation experiments.

## Configuration Files

- `grid_search_config.yaml`: Complete configuration with all available options
- `grid_search_test_config.yaml`: Simplified configuration for testing

## How to Run

### Method 1: Using the bash script with nohup (recommended for long-running experiments)
```bash
# Run with default config (grid_search_config.yaml)
./run_grid_search_yaml.sh

# Run with specific config file
./run_grid_search_yaml.sh grid_search_test_config.yaml
```

### Method 2: Run directly with Python
```bash
# Run with default config (grid_search_config.yaml)
python run_grid_search_yaml.py

# Run with specific config file
python run_grid_search_yaml.py grid_search_test_config.yaml
```

## Configuration Structure

The configuration file is organized into sections:

### Base Settings
```yaml
base:
  dataset_path: "/path/to/dataset.json"
  dir_path: "/path/to/content/directory"
  output_csv: "/path/to/output.csv"
  r2r_base_url: "http://localhost:7272"
  r2r_timeout: 604800  # 1 week timeout (7*24*60*60 seconds)
  ingestion_thread_pool_workers: 16
  search_thread_pool_workers: 16
```

### Model Configurations
```yaml
llm:
  model: "${LLM_MODEL}"  # Values from environment variables
  api_key: "${LLM_API_KEY}"
  base_url: "${LLM_API_BASE}"

embeddings:
  model: "${EMBEDDINGS_MODEL}"
  api_key: "${EMBEDDINGS_API_KEY}"
  base_url: "${EMBEDDINGS_API_BASE}"
  embedding_dimension: 1024
```

### Search Configurations
Each search strategy is defined with its settings:
```yaml
search_configs:
  semantic_search:
    search_settings:
      limit: 5
      use_semantic_search: true
      use_hybrid_search: false
      use_fulltext_search: false
    search_mode: "custom"
```

### Chunkers
Each chunker is defined with its type and parameters:
```yaml
chunkers:
  character:
    type: "CharacterChunker"
    params:
      chunk_size: 1024
      chunk_overlap: 256
```

### Enrichment Strategies
Each enrichment strategy is defined with its type and parameters:
```yaml
enrichments:
  metadata_enrichment:
    type: "MetadataEnrichment"
    params:
      include_entities: true
      include_keywords: true
      # ...more parameters
```

## Environment Variables

The configuration system supports substituting environment variables using the `${VAR_NAME}` syntax. The following special variables are available:

- `${CPU_COUNT}`: Automatically set to `os.cpu_count() * 2`
- Any environment variables defined in your `.env` file

## Customizing Your Experiments

To run a custom experiment:

1. Create a copy of one of the existing YAML files
2. Modify the configurations as needed
3. Run the grid search with your custom config file
```
