import json
from typing import Dict, Any, Tuple, List
from ragas.integrations.r2r import (
    transform_to_ragas_dataset as transform_to_ragas_dataset_r2r,
)


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
    r2r_responses: List[str],
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
