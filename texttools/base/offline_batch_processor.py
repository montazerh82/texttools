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
        All state is stored in local JSON files, not instance variables.
        """
        self.client = client
        self.handlers = handlers or []
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def _save_state(self, job_name: str, jobs: List[Dict[str, Any]]):
        """Saves the jobs list to a JSON file for the given job_name."""
        state_file = self.state_dir / f"{job_name}.json"
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(jobs, f)

    def _load_state(self, job_name: str) -> List[Dict[str, Any]]:
        """Loads the jobs list from the JSON file for the given job_name."""
        state_file = self.state_dir / f"{job_name}.json"
        if state_file.exists():
            with open(state_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def _clear_state(self, job_name: str):
        """Clears the state file for the given job_name."""
        state_file = self.state_dir / f"{job_name}.json"
        if state_file.exists():
            try:
                state_file.unlink()
            except Exception:
                pass

    @abstractmethod
    def _prepare_upload_file(self, payload: Any) -> Path:
        """Prepares a JSONL file to upload."""
        pass

    @abstractmethod
    def _submit_batch(self, file_path: Path) -> str:
        """Submits the batch to remote service. Returns the batch_id."""
        pass

    @abstractmethod
    def _parse_entry(self, entry: Dict[str, Any]) -> Any:
        """Parses a single line from the batch result file."""
        pass

    def _dispatch(self, results: Dict[str, Any]):
        """Dispatches results to handlers."""
        for handler in self.handlers:
            handler.handle(results)

    def start(self, payload: Any, job_name: str):
        """
        Starts a new batch job and saves its state to a local JSON file.
        """
        file_path = self._prepare_upload_file(payload)
        batch_id = self._submit_batch(file_path)
        jobs = [{"batch_id": batch_id, "output_file_id": None}]
        self._save_state(job_name, jobs)

    def check_status(self, job_name: str) -> str:
        """
        Checks the status of the batch job by reading from the JSON file.
        Returns 'completed', 'in_progress', 'failed', or 'cancelled'.
        """
        jobs = self._load_state(job_name)
        if not jobs:
            return "completed"

        for job in jobs:
            status = self.client.batches.retrieve(job["batch_id"]).status
            if status in ("failed", "cancelled"):
                return status
            if status != "completed":
                return "in_progress"
        return "completed"

    def is_ready(self, job_name: str) -> bool:
        """
        Shortcut: returns True if batch processing is completed for the job_name.
        """
        return self.check_status(job_name) == "completed"

    def fetch_results(self, job_name: str) -> Dict[str, Any]:
        """
        Fetches results, dispatches to handlers, and clears state.
        Will load existing output_file_id from state if present.
        """
        jobs = self._load_state(job_name)
        if not jobs:
            print(f"No batch jobs found for '{job_name}'.")
            return {}

        results = {}
        updated = False

        for job in jobs:
            batch_id = job["batch_id"]
            file_id = job.get("output_file_id")

            # if we haven’t yet stored an output_file_id, retrieve it
            if not file_id:
                info = self.client.batches.retrieve(batch_id)
                if info.status != "completed":
                    print(f"Batch {batch_id} status: {info.status}")
                    continue

                # support both old and new SDK fields
                file_id = getattr(info, "output_file_id", None)
                if not file_id:
                    plural = getattr(info, "output_file_ids", None) or getattr(info, "output_files", None)
                    if isinstance(plural, list) and plural:
                        first = plural[0]
                        file_id = getattr(first, "id", None) or first

                if not file_id:
                    print(f"No output file for batch {batch_id}")
                    continue

                # persist it
                job["output_file_id"] = file_id
                updated = True

            # now we have a file_id, fetch content
            content = self.client.files.content(file_id).text
            for line in content.splitlines():
                entry = json.loads(line)
                key, parsed = self._parse_entry(entry)
                results[key] = parsed

        # if we wrote back new file_ids, save state
        if updated:
            self._save_state(job_name, jobs)

        if results:
            self._dispatch(results)
            self._clear_state(job_name)

        return results