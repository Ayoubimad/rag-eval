import json
import re
from typing import Dict, Any, Tuple, List
from ragas.integrations.r2r import (
    transform_to_ragas_dataset as transform_to_ragas_dataset_r2r,
)

import logging
from typing import Optional


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a configured logger with consistent formatting.

    Args:
        name: Optional name for the logger (typically __name__ from the calling module)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()

        grey = "\033[90m"
        blue = "\033[94m"
        green = "\033[92m"
        yellow = "\033[93m"
        red = "\033[91m"
        reset = "\033[0m"

        formatter = logging.Formatter(
            f"{grey}%(asctime)s{reset} - {blue}%(name)s{reset} - {green}%(levelname)s{reset} - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        logger.setLevel(logging.INFO)

    return logger


def load_dataset(dataset_path: str) -> Tuple[List[str], List[str], List[List[str]]]:
    """
    Load the evaluation dataset from a file.

    Args:
        dataset_path: Path to the dataset file

    Returns:
        Loaded dataset
    """
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
    return dataset["user_input"], dataset["reference"], dataset["reference_contexts"]


def transform_to_ragas_dataset(
    user_inputs: List[str],
    r2r_responses: Any,
    references: List[str],
    reference_contexts: List[List[str]],
) -> Dict[str, Any]:
    """
    Transform the dataset to a Ragas dataset.
    """
    return transform_to_ragas_dataset_r2r(
        user_inputs=user_inputs,
        r2r_responses=r2r_responses,
        references=references,
        reference_contexts=reference_contexts,
    )


def clean_text(text: str) -> str:
    """Normalize whitespace in text and remove ambiguous Unicode characters.

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
    # Remove zero-width and other ambiguous Unicode characters
    cleaned_text = re.sub(
        r"[\u200B-\u200F\u2028-\u202F\u2060-\u206F]", "", cleaned_text
    )
    return cleaned_text
