import logging
import os
import json
from datetime import datetime

class ExperimentLogger:
    def __init__(self, log_dir: str = "logs", topic: str = "experiment", mode: str = None, openness: str = None, retrieval_strategy: str = None, batch_dir: str = None, run_name_prefix: str = None):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Setup standard python logging for console output
        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s'
        )
        
        # Suppress noisy HTTP request logs
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        
        self.console_logger = logging.getLogger("Experiment")

        # Create base log directory
        self.base_log_dir = log_dir
        os.makedirs(self.base_log_dir, exist_ok=True)
        
        if batch_dir:
            # If part of a batch, use the provided batch directory
            self.topic_log_dir = batch_dir
            os.makedirs(self.topic_log_dir, exist_ok=True)
            
            # For batch runs, we might want a simpler folder name or just the standard one
            # Let's keep the standard naming but inside the batch folder
        else:
            # Create topic-specific subdirectory
            self.topic_log_dir = os.path.join(self.base_log_dir, topic)
            os.makedirs(self.topic_log_dir, exist_ok=True)
        
        # Find next available ID based on existing FOLDERS in the TOPIC directory
        # Look for folders like "mode_openness_001_date" or "mode_001_date"
        existing_items = os.listdir(self.topic_log_dir)
        next_id = 1
        ids = []
        
        # Determine prefix for matching
        # Since we are inside the topic folder, we don't need topic in the prefix
        prefix_parts = []
        if run_name_prefix:
            prefix_parts.append(run_name_prefix)
        if mode:
            prefix_parts.append(mode)
        if openness:
            prefix_parts.append(str(openness))
            
        prefix = "_".join(prefix_parts)
        
        for item in existing_items:
            # If prefix is empty (no mode/openness), we match everything that looks like an ID_Date
            if (not prefix or item.startswith(prefix)) and os.path.isdir(os.path.join(self.topic_log_dir, item)):
                parts = item.split("_")
                # Try to find the ID part. It's usually the one before the date parts.
                # Format: mode_openness_ID_YYYYMMDD_HHMMSS
                for part in parts:
                    if part.isdigit() and len(part) == 3:
                        ids.append(int(part))
        
        if ids:
            next_id = max(ids) + 1
        
        self.run_id = f"{next_id:03d}"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create dedicated run directory inside topic directory
        folder_parts = []
        if run_name_prefix:
            folder_parts.append(run_name_prefix)
        if mode:
            folder_parts.append(mode)
        if openness:
            folder_parts.append(str(openness))
        if retrieval_strategy:
            folder_parts.append(str(retrieval_strategy))
        folder_parts.append(self.run_id)
        folder_parts.append(timestamp)
        
        folder_name = "_".join(folder_parts)
        self.run_dir = os.path.join(self.topic_log_dir, folder_name)
        os.makedirs(self.run_dir, exist_ok=True)
        
        # Filenames (simplified since they are in a dedicated folder)
        self.log_file = os.path.join(self.run_dir, "log.jsonl")
        self.memory_log_file = os.path.join(self.run_dir, "memory.jsonl")
        self.summary_file = os.path.join(self.run_dir, "summary.md")
        
        self.console_logger.info(f"Logging experiment run to: {self.run_dir}")

    def log_event(self, event_type: str, data: dict):
        """
        Logs a structured event to the main JSONL file.
        """
        entry = {
            "event_type": event_type,
            "data": data
        }
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def log_memory(self, round_num: int, arguments: list):
        """
        Logs the memory snapshot to the separate memory JSONL file.
        """
        entry = {
            "round": round_num,
            "arguments": arguments
        }
        with open(self.memory_log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def log_summary(self, summary_text: str):
        """
        Writes the human-readable summary to a markdown file.
        """
        with open(self.summary_file, "w") as f:
            f.write(summary_text)
        self.console_logger.info(f"Summary saved to: {self.summary_file}")

    def info(self, message: str):
        self.console_logger.info(message)

    def error(self, message: str):
        self.console_logger.error(message)
