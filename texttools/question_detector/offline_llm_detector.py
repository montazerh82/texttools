import json
import uuid
import time
from pathlib import Path
from openai import OpenAI
from pydantic import create_model
from texttools.base import OfflineBatchProcessor
from typing import Any, Dict, List, Optional, Tuple
from texttools.base.base_question_detector import BaseQuestionDetector


class OfflineQuestionDetector(OfflineBatchProcessor, BaseQuestionDetector):
    """
    Offline batch-based binary question detector using the OpenAI Batch API.

    Usage:
        detector = OfflineQuestionDetector(
            client=client,
            model="gpt-4o-mini",
            temperature=0.0,
            prompt_template="You are a binary classifier. Answer only with `true` or `false`.",
            handlers=[my_handler],
            max_tokens=10
        )
        # Single synchronous detection
        is_q = detector.detect("Is this a question?")
        # Batch detection
        detector.start([...], job_name="qd_job")
        # ... poll for readiness ...
        results = detector.fetch_results()
    """
    def __init__(
        self,
        client: OpenAI,
        *,
        model: str,
        temperature: float = 0.0,
        prompt_template: Optional[str] = None,
        handlers: Optional[List[Any]] = None,
        **client_kwargs: Any
    ):
        super().__init__(client=client, handlers=handlers)
        BaseQuestionDetector.__init__(self, handlers=handlers)

        self.model = model
        self.temperature = temperature
        self.prompt_template = prompt_template or (
            "You are a binary classifier. Answer only with `true` or `false`."
        )
        
        # Pydantic model for structured batch output
        self._OutputModel = create_model(
            "DetectionOutput",
            result=(bool, ...),  
        )
        
        self.client_kwargs = client_kwargs

    def _build_task(self, text: str, idx: int) -> Dict[str, Any]:
        clean = self.preprocess(text)
        return {
            "custom_id": str(idx),
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": self.model,
                "temperature": self.temperature,
                "response_format": self._OutputModel.model_json_schema(),
                "messages": [
                    {"role": "system", "content": self.prompt_template},
                    {"role": "user", "content": clean}
                ],
                **self.client_kwargs
            }
        }

    def _prepare_upload_file(self, payload: List[str]) -> Path:
        tasks = [self._build_task(text, i) for i, text in enumerate(payload)]
        file_id = uuid.uuid4().hex
        file_path = self.state_dir / f"batch_questions_{file_id}.jsonl"
        with open(file_path, "w", encoding="utf-8") as f:
            for task in tasks:
                f.write(json.dumps(task) + "\n")
        return file_path

    def _submit_batch(self, file_path: Path) -> str:
        batch_file = self.client.files.create(
            file=open(file_path, "rb"),
            purpose="batch"
        )
        batch_job = self.client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        return batch_job.to_dict()

    def _parse_entry(self, entry: Dict[str, Any]) -> Tuple[str, bool]:
        custom_id = entry.get("custom_id")
        body = entry.get("response", {}).get("body", {})
        # Handle structured OutputModel result if present
        if "result" in body:
            result = bool(body["result"])
        else:
            # Fallback to legacy content parsing
            choices = body.get("choices", [])
            if not choices:
                raise ValueError(f"No choices in batch response for id {custom_id}")
            content = choices[0]["message"]["content"]
            obj = json.loads(content)
            result = bool(obj.get("result"))
        return custom_id, result

    def detect(
        self,
        text: str,
        timeout: int = 300000,
        poll_interval: int = 5
    ) -> bool:
        """
        Synchronously detects if a single string is a question.
        Blocks until the batch job completes or timeout is reached.
        """
        job_name = f"qd_{uuid.uuid4().hex}"
        self.start([text], job_name=job_name)
        elapsed = 0
        while not self.is_ready():
            if elapsed >= timeout:
                raise TimeoutError(f"Question detection timed out after {timeout}s")
            time.sleep(poll_interval)
            elapsed += poll_interval

        results = self.fetch_results()
        result_bool = results.get("0")
        if result_bool is None:
            raise ValueError("No result returned for question detection")
        self._dispatch({"result": result_bool})
        return result_bool
