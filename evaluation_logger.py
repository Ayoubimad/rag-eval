import os
import time
import logging
from datetime import datetime


class EvaluationLogger:
    def __init__(self, log_file_path="evaluation_progress.log"):
        """
        Initialize the evaluation logger.

        Args:
            log_file_path: Path to save the evaluation progress log
        """
        self.log_file_path = log_file_path

        # Set up logger
        self.logger = logging.getLogger("evaluation_logger")
        self.logger.setLevel(logging.INFO)

        # Create file handler if it doesn't exist
        if not os.path.exists(os.path.dirname(log_file_path)) and os.path.dirname(
            log_file_path
        ):
            os.makedirs(os.path.dirname(log_file_path))

        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter("%(asctime)s - %(message)s")
        file_handler.setFormatter(formatter)

        # Add handler to logger
        self.logger.addHandler(file_handler)

        # Also log to console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # Track start time
        self.start_time = time.time()

        # Log header
        self.logger.info("=== Evaluation Run Started ===")
        self.logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    def log_evaluation(
        self, graph_enabled, enrichment_strategy, chunker_name, search_strategy
    ):
        """
        Log the current evaluation type that is running.

        Args:
            graph_enabled: Whether graph RAG is enabled
            enrichment_strategy: Name of the enrichment strategy being used
            chunker_name: Name of the chunker being used
            search_strategy: Name of the search strategy being used
        """
        elapsed = time.time() - self.start_time
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)

        self.logger.info(
            f"[{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}] "
            f"Evaluation: Graph={graph_enabled}, "
            f"Enrichment={enrichment_strategy}, "
            f"Chunker={chunker_name}, "
            f"Search={search_strategy}"
        )

    def finalize(self):
        """
        Log completion of evaluation run.
        """
        elapsed = time.time() - self.start_time
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)

        self.logger.info("=== Evaluation Run Completed ===")
        self.logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(
            f"Total runtime: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
        )
