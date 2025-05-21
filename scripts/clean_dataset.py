import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from executor.executor import run_in_executor


class DatasetCleaner:
    def __init__(
        self,
        model: str = "ISTA-DASLab/gemma-3-4b-it-GPTQ-4b-128g",
        api_base: str = "http://172.18.21.136:8000/v1",
        temperature: float = 0.2,
        max_workers: int = 4,
    ):
        self.llm = ChatOpenAI(
            model=model,
            base_url=api_base,
            temperature=temperature,
            api_key="random_api_key",
        )
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    async def clean_question(self, question: str, context: List[str]) -> str:
        """Clean and improve a single question using the LLM."""
        prompt = f"""You are given a user question and a reference context. Perform **one** of the following actions:

            1. If the question is relevant to the context but unclear, vague, or poorly worded, **rewrite it** to make it:
            - clearer and more specific
            - grammatically correct and properly formatted
            - concise and faithful to the original intent
            - suitable for evaluation in a Retrieval-Augmented Generation (RAG) setting
            - realistic, as if asked by a user **who has no access to the reference context
                Avoid formulations like "according to the text", "based on the passage", etc.**
                'Example of not valid question: How is the CLIP-Score metric calculated, according to the provided text?'
                'Example of valid question: How is the CLIP-Score metric calculated?'
                'Example of not valid question: How is BLIP-Diffusion used, according to the document?'
                'Example of valid question: How is BLIP-Diffusion used?'
                
            2. If the question is:
            - completely irrelevant to the context,
            - not answerable based on the context,
            - or clearly not suitable for RAG (e.g., meta-questions, ambiguous references),
            then return **"REMOVE"**.

            ---

            **Question:**  
            {question}

            **Context (for relevance check):**  
            {' '.join(context)}

            ---

            ## Response Format  
            Output **only** the improved question or **"REMOVE"** — no explanations, no extra formatting, no prefixes.
"""
        response = await self.llm.ainvoke(prompt)

        return response.content.strip()

    def clean_question_sync(self, question: str, context: List[str]) -> str:
        """Synchronous version of clean_question."""
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(self.clean_question(question, context))
            return result
        finally:
            loop.close()

    async def process_batch(
        self, batch_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Process a batch of dataset entries using thread executor."""
        cleaned_batch = []
        tasks = []

        for entry in batch_data:
            task = run_in_executor(
                self.executor,
                self.clean_question,
                entry["user_input"],
                entry["reference_contexts"],
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks)

        for entry, cleaned_question in zip(batch_data, results):
            if cleaned_question != "REMOVE":
                cleaned_entry = entry.copy()
                cleaned_entry["user_input"] = cleaned_question
                cleaned_batch.append(cleaned_entry)

        return cleaned_batch

    async def clean_dataset(self, dataset_path: str, output_path: str):
        """Clean the entire dataset and save the results."""
        with open(dataset_path, "r") as f:
            dataset = json.load(f)

        dataset_entries = [
            {"user_input": q, "reference": r, "reference_contexts": c}
            for q, r, c in zip(
                dataset["user_input"],
                dataset["reference"],
                dataset["reference_contexts"],
            )
        ]

        cleaned_entries = []
        with tqdm(total=len(dataset_entries)) as pbar:
            for entry in dataset_entries:
                cleaned_question = await run_in_executor(
                    self.executor,
                    self.clean_question_sync,  # Use the sync version
                    entry["user_input"],
                    entry["reference_contexts"],
                )
                if cleaned_question != "REMOVE":
                    cleaned_entry = entry.copy()
                    cleaned_entry["user_input"] = cleaned_question
                    cleaned_entries.append(cleaned_entry)
                pbar.update(1)

        cleaned_dataset = {
            "user_input": [entry["user_input"] for entry in cleaned_entries],
            "reference": [entry["reference"] for entry in cleaned_entries],
            "reference_contexts": [
                entry["reference_contexts"] for entry in cleaned_entries
            ],
        }

        with open(output_path, "w") as f:
            json.dump(cleaned_dataset, f, indent=2, ensure_ascii=False)

        print(f"Original dataset size: {len(dataset_entries)}")
        print(f"Cleaned dataset size: {len(cleaned_entries)}")
        print(
            f"Removed {len(dataset_entries) - len(cleaned_entries)} irrelevant questions"
        )

    def __del__(self):
        """Cleanup executor on deletion."""
        self.executor.shutdown(wait=True)


def main():
    dataset_path = "/home/e4user/rag-eval/datasets/ragas_testset_arxi_papers.json"
    output_path = (
        "/home/e4user/rag-eval/datasets/ragas_testset_arxi_papers_cleaned.json"
    )

    cleaner = DatasetCleaner(max_workers=16)
    asyncio.run(cleaner.clean_dataset(dataset_path, output_path))


if __name__ == "__main__":
    main()
