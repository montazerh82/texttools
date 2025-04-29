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
        Starts a new batch job **only if** there isn't one in-flight already,
        and saves the full batch metadata to JSON.
        """
        # if state already exists, assume it has the full batch dict:
        existing = self._load_state(job_name)
        if existing:
            print(f"Resuming existing batch for '{job_name}' → batch_id={existing[0]['id']!r}")
            return

        file_path = self._prepare_upload_file(payload)
        batch_meta : dict = self._submit_batch(file_path)

        # batch_meta.update({
        #     "output_file_id": None,
        #     "error_file_id": None,
        # })
        self._save_state(job_name, [batch_meta])
        print(f"Started new batch: {batch_meta['id']!r}")

    def check_status(self, job_name: str) -> str:
        """
        Checks the status of the batch job by reading from the JSON file.
        Returns 'completed', 'in_progress', 'failed', or 'cancelled'.
        """
        jobs = self._load_state(job_name)
        if not jobs:
            return "completed"

        for job in jobs:
            status = self.client.batches.retrieve(job["id"]).status
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
        Loads the saved batch metadata, refreshes status,
        stores any new file_ids back to disk, and finally
        downloads results/errors.
        """
        jobs = self._load_state(job_name)
        if not jobs:
            print(f"No batch jobs found for '{job_name}'.")
            return {}

        all_results = {}
        all_errors = {}
        updated = False

        for job in jobs:
            batch_id    = job["id"]
            out_file_id = job.get("output_file_id")
            err_file_id = job.get("error_file_id")

            # first time through: pull fresh metadata to grab file IDs
            if not out_file_id and not err_file_id:
                info = self.client.batches.retrieve(batch_id)
                if info.status != "completed":
                    print(f"Batch {batch_id} not yet done (status={info.status}).")
                    continue

                # pick up either single or plural field:
                out_file_id = getattr(info, "output_file_id", None)
                if not out_file_id:
                    plural = getattr(info, "output_file_ids", None) or getattr(info, "output_files", None)
                    if isinstance(plural, list) and plural:
                        first = plural[0]
                        out_file_id = getattr(first, "id", None) or first

                err_file_id = getattr(info, "error_file_id", None)

                # write them back into our persisted JSON
                job["output_file_id"] = out_file_id
                job["error_file_id"]  = err_file_id
                updated = True

            # if there’s an output file, pull and parse successes
            if out_file_id:
                content = self.client.files.content(out_file_id).text
                for line in content.splitlines():
                    entry = json.loads(line)
                    key, parsed = self._parse_entry(entry)
                    all_results[key] = parsed

            # if there’s an error file, pull and parse failures
            if err_file_id:
                err_txt = self.client.files.content(err_file_id).text

        # if we added file IDs, persist them
        if updated:
            self._save_state(job_name, jobs)

        if all_results:
            self._dispatch(all_results)
            self._clear_state(job_name)

        return {"results": all_results, "errors": err_txt}