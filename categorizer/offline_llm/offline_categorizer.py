import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from enum import Enum

from openai import OpenAI
from pydantic import create_model
from ..base import BaseCategorizer
from ..handlers import NoOpResultHandler

class OfflineCategorizer(BaseCategorizer):
    """
    Offline (batch) categorizer that persists its own job state locally
    in a JSON file, so it survives restarts without extra setup.
    """

    DEFAULT_STATE = Path(".offline_categorizer_state.json")

    def __init__(
        self,
        client: OpenAI,
        categories: Enum,
        *,
        model: str,
        temperature: float = 0.0,
        prompt_template: str = None,
        batch_size: int = 500,
        polling_interval: float = 30.0,
        handlers: Optional[List[NoOpResultHandler]] = None,
        state_file: Path = None,
        **client_kwargs: Any
    ):
        super().__init__(categories, handlers=handlers)
        self.client = client
        self.model = model
        self.temperature = temperature
        self.prompt_template = prompt_template or "You are a classifier. Pick one category."
        self.batch_size = batch_size
        self.polling_interval = polling_interval
        self.client_kwargs = client_kwargs

        self._OutputModel = create_model(
            "CategorizationOutput",
            category=(self.categories, ...)
        )

        self.state_file = state_file or self.DEFAULT_STATE
        self._jobs = self._load_state()

    def _load_state(self) -> List[Dict[str, str]]:
        if self.state_file.exists():
            try:
                return json.loads(self.state_file.read_text())
            except Exception:
                return []
        return []

    def _save_state(self) -> None:
        self.state_file.write_text(json.dumps(self._jobs, indent=2))

    def start_process(self, texts: List[str]) -> None:
        """
        Chunk texts, upload JSONL, create batch jobs, and persist job metadata.
        """
        chunks = [
            texts[i : i + self.batch_size]
            for i in range(0, len(texts), self.batch_size)
        ]

        for idx, chunk in enumerate(chunks):
            with tempfile.NamedTemporaryFile("w+", suffix=".jsonl", delete=False) as tf:
                for i, txt in enumerate(chunk):
                    cid = f"chunk{idx}_item{i}"
                    body = {
                        "model": self.model,
                        "input": [
                            {"role": "system", "content": self.prompt_template},
                            {"role": "user",   "content": self.preprocess(txt)},
                        ],
                        "text_format": self._OutputModel.schema(),
                        "temperature": self.temperature,
                        **self.client_kwargs,
                    }
                    tf.write(json.dumps({
                        "custom_id": cid,
                        "method": "POST",
                        "url": "/v1/responses",
                        "body": body,
                    }) + "\n")
                tmp_path = Path(tf.name)

            up = self.client.files.create(file=open(tmp_path, "rb"), purpose="batch")
            batch = self.client.batches.create(
                input_file_id=up.id,
                endpoint="/v1/responses",
                completion_window="24h",
            )

            job = {"file_id": up.id, "batch_id": batch.id}
            self._jobs.append(job)
            self._save_state()

    def check_status(self) -> Optional[Dict[str, Enum]]:
        """
        Poll in-flight batches. Once all are completed, download, parse,
        dispatch results, clean up state file, and return the dict.
        Otherwise return None.
        """
        if not self._jobs:
            return None

        for job in list(self._jobs):
            status = self.client.batches.retrieve(job["batch_id"]).status
            if status in ("failed", "cancelled"):
                raise RuntimeError(f"Batch {job['batch_id']} ended with {status}")
            if status != "completed":
                return None

        results: Dict[str, Enum] = {}
        for job in self._jobs:
            out_id = self.client.batches.retrieve(job["batch_id"]).output_file_id
            raw = self.client.files.content(out_id).text
            for line in raw.splitlines():
                entry = json.loads(line)
                parsed = self._OutputModel.parse_obj(entry["response"]["body"])
                results[entry["custom_id"]] = parsed.category

        self._dispatch(results)

        self._jobs = []
        try:
            self.state_file.unlink()
        except Exception:
            pass

        return results

    def categorize(self, text: str) -> Enum:
        """
        Provide the single-item interface by wrapping the batch pipeline.
        """
        single = self.categorize_batch([text])
        return next(iter(single.values()))
