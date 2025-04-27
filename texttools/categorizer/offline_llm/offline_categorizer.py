import json
from pathlib import Path
from enum import Enum
from typing import List, Any, Tuple, Dict
from base import OfflineBatchProcessor

from openai import OpenAI

class OfflineCategorizer(OfflineBatchProcessor):
    """
    Offline categorization using batch API.
    """

    def __init__(
        self,
        *,
        client: OpenAI,
        categories: Enum,
        handlers: List[Any] = None,
        state_dir: Path = Path(".batch_jobs"),
        prompt_template: str = None,
        model: str = "gpt-4o",
        temperature: float = 0.0,
        **client_kwargs
    ):
        super().__init__(client=client, handlers=handlers, state_dir=state_dir)
        self.categories = categories
        self.prompt_template = prompt_template or (
            "You are a text classifier. Choose exactly one category from the list."
        )
        self.model = model
        self.temperature = temperature
        self.client_kwargs = client_kwargs

    def _prepare_upload_file(self, payload: List[str]) -> Path:
        """
        Create a JSONL file with prompts for the LLM to classify.
        """
        upload_path = self.state_dir / "batch_upload.jsonl"

        with open(upload_path, "w", encoding="utf-8") as f:
            for text in payload:
                messages = [
                    {"role": "system", "content": self.prompt_template},
                    {"role": "user", "content": text},
                ]
                record = {
                    "custom_id": text,  
                    "messages": messages,
                }
                f.write(json.dumps(record) + "\n")

        return upload_path

    def _submit_batch(self, file_path: Path) -> str:
        """
        Submit the batch file to the OpenAI batch endpoint.
        """
        batch = self.client.batches.create(
            input_file=open(file_path, "rb"),
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"type": "offline_categorizer"},
            **self.client_kwargs,
        )
        return batch.id

    def _parse_entry(self, entry: Dict[str, Any]) -> Tuple[str, Enum]:
        """
        Parse each line of the output file.
        """
        original_text = entry["custom_id"]
        output_content = entry["response"]["choices"][0]["message"]["content"]

        for cat in self.categories:
            if cat.name.lower() == output_content.strip().lower():
                return original_text, cat

        raise ValueError(f"Unknown category returned: {output_content}")
