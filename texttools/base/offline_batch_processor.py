import json
from pathlib import Path
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from openai import OpenAI

class OfflineBatchProcessor(ABC):
    def __init__(
        self,
        *,
        client: OpenAI,
        handlers: Optional[List[Any]] = None,
        state_dir: Path = Path(".batch_jobs"),
    ):
        """
        Base class for managing offline batch processes.
        """
        self.client = client
        self.handlers = handlers or []
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._jobs: List[Dict[str, Any]] = []
        self._state_file: Optional[Path] = None

    def _save_state(self):
        if self._state_file:
            with open(self._state_file, "w", encoding="utf-8") as f:
                json.dump(self._jobs, f)

    def _load_state(self, name: str):
        self._state_file = self.state_dir / f"{name}.json"
        if self._state_file.exists():
            with open(self._state_file, "r", encoding="utf-8") as f:
                self._jobs = json.load(f)
        else:
            self._jobs = []

    def _clear_state(self):
        if self._state_file and self._state_file.exists():
            try:
                self._state_file.unlink()
            except Exception:
                pass
        self._jobs = []

    @abstractmethod
    def _prepare_upload_file(self, payload: Any) -> Path:
        """
        Prepares a JSONL file to upload.
        """
        pass

    @abstractmethod
    def _submit_batch(self, file_path: Path) -> str:
        """
        Submits the batch to remote service. 
        Returns the batch_id.
        """
        pass

    @abstractmethod
    def _parse_entry(self, entry: Dict[str, Any]) -> Any:
        """
        Parses a single line from the batch result file.
        """
        pass

    def _dispatch(self, results: Dict[str, Any]):
        for handler in self.handlers:
            handler.handle(results)

    def start(self, payload: Any, job_name: str):
        """
        Starts a new batch job.
        """
        file_path = self._prepare_upload_file(payload)
        batch_id = self._submit_batch(file_path)

        self._state_file = self.state_dir / f"{job_name}.json"
        self._jobs = [{"batch_id": batch_id}]
        self._save_state()

    def check_status(self) -> str:
        """
        Returns "completed", "in_progress", "failed", or "cancelled".
        """
        if not self._jobs:
            return "completed"

        for job in self._jobs:
            status = self.client.batches.retrieve(job["batch_id"]).status
            if status in ("failed", "cancelled"):
                return status
            if status != "completed":
                return "in_progress"
        return "completed"

    def is_ready(self) -> bool:
        """
        Shortcut: returns True if batch processing is completed.
        """
        return self.check_status() == "completed"

    def fetch_results(self) -> Dict[str, Any]:
        """
        Fetches the results after batch is ready.
        Dispatches to handlers and clears state.
        """
        if not self._jobs and self._state_file and self._state_file.exists():
                print(f"Reloading jobs from {self._state_file}")
                with open(self._state_file, "r", encoding="utf-8") as f:
                    self._jobs = json.load(f)
        if not self._jobs:
            print("No batch jobs found.")
            return {}

        results = {}

        for job in self._jobs:
            batch_info = self.client.batches.retrieve(job["batch_id"])
            file_id = batch_info.output_file_id
            content = self.client.files.content(file_id).text

            for line in content.splitlines():
                entry = json.loads(line)
                key, parsed = self._parse_entry(entry)
                results[key] = parsed

        self._dispatch(results)
        self._clear_state()

        return results
