import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from openai import OpenAI
from pydantic import BaseModel


class SimpleBatchManager:
    def __init__(
        self,
        client: OpenAI,
        model: str,
        output_model: Type[BaseModel],
        prompt_template: str,
        handlers: Optional[List[Any]] = None,
        state_dir: Path = Path(".batch_jobs"),
        **client_kwargs: Any,
    ):
        self.client = client
        self.model = model
        self.output_model = output_model
        self.prompt_template = prompt_template
        self.handlers = handlers or []
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.client_kwargs = client_kwargs

    def _state_file(self, job_name: str) -> Path:
        return self.state_dir / f"{job_name}.json"

    def _load_state(self, job_name: str) -> List[Dict[str, Any]]:
        path = self._state_file(job_name)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def _save_state(self, job_name: str, jobs: List[Dict[str, Any]]):
        with open(self._state_file(job_name), "w", encoding="utf-8") as f:
            json.dump(jobs, f)

    def _clear_state(self, job_name: str):
        path = self._state_file(job_name)
        if path.exists():
            path.unlink()

    def _build_task(self, text: str, idx: int) -> Dict[str, Any]:
        schema = self.output_model.model_json_schema()
        return {
            "custom_id": str(idx),
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": self.prompt_template},
                    {"role": "user", "content": text},
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": schema,
                    "strict": True,
                },
                **self.client_kwargs,
            },
        }

    def _prepare_file(self, payload: List[str]) -> Path:
        tasks = [self._build_task(text, i) for i, text in enumerate(payload)]
        file_path = self.state_dir / f"batch_{uuid.uuid4().hex}.jsonl"
        with open(file_path, "w", encoding="utf-8") as f:
            for task in tasks:
                f.write(json.dumps(task) + "\n")
        return file_path

    def start(self, payload: List[str], job_name: str):
        if self._load_state(job_name):
            return  # batch already started
        path = self._prepare_file(payload)
        upload = self.client.files.create(file=open(path, "rb"), purpose="batch")
        job = self.client.batches.create(
            input_file_id=upload.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        ).to_dict()
        self._save_state(job_name, [job])

    def check_status(self, job_name: str) -> str:
        jobs = self._load_state(job_name)
        if not jobs:
            return "completed"
        updated = False
        for job in jobs:
            batch = self.client.batches.retrieve(job["id"])
            job["status"] = batch.status
            updated = True
            if batch.status in ("failed", "cancelled"):
                break
            if batch.status != "completed":
                break
        if updated:
            self._save_state(job_name, jobs)
        return jobs[0]["status"]

    def is_ready(self, job_name: str) -> bool:
        return self.check_status(job_name) == "completed"

    def fetch_results(self, job_name: str) -> Dict[str, Any]:
        jobs = self._load_state(job_name)
        if not jobs:
            return {}
        job = jobs[0]
        batch_id = job["id"]

        info = self.client.batches.retrieve(batch_id)
        out_file_id = getattr(info, "output_file_id", None)
        if not out_file_id:
            plural = getattr(info, "output_file_ids", None) or getattr(info, "output_files", None)
            if isinstance(plural, list) and plural:
                out_file_id = getattr(plural[0], "id", None) or plural[0]

        job["output_file_id"] = out_file_id
        self._save_state(job_name, jobs)

        results = {}
        if out_file_id:
            txt = self.client.files.content(out_file_id).text
            for line in txt.splitlines():
                entry = json.loads(line)
                cid = entry["custom_id"]
                parsed = self.output_model.model_validate(entry["response"]).model_dump()
                results[cid] = parsed

        if results:
            for handler in self.handlers:
                handler.handle(results)
            self._clear_state(job_name)

        return {"results": results}
