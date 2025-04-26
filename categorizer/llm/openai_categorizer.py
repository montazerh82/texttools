from enum import Enum
from typing import Any, Dict, List
from pydantic import BaseModel, create_model
from openai import OpenAI
from ..base import BaseCategorizer

class LLMCategorizer(BaseCategorizer):
    """
    LLM-based categorizer using OpenAI's client.responses.parse
    for Structured Outputs (Pydantic models).
    """

    def __init__(
        self,
        client: OpenAI,
        categories: Enum,
        *,
        model: str,
        temperature: float = 0.0,
        prompt_template: str = None,
        **client_kwargs: Any
    ):
        """
        :param client: an instantiated OpenAI client
        :param categories: an Enum class of allowed categories
        :param model: the model name (e.g. "gpt-4o-2024-08-06")
        :param temperature: sampling temperature
        :param prompt_template: override default prompt instructions
        :param client_kwargs: any other OpenAI kwargs (e.g. `max_tokens`, `top_p`, etc.)
        """
        super().__init__(categories)
        self.client = client
        self.model = model
        self.temperature = temperature
        self.client_kwargs = client_kwargs

        self.prompt_template = prompt_template or (
            "You are a text classifier. "
            "Choose exactly one category from the list."
        )

        self._OutputModel = create_model(
            "CategorizationOutput",
            category=(self.categories, ...),  
        )

    def _build_messages(self, text: str) -> List[Dict[str, str]]:
        clean = self.preprocess(text)
        return [
            {"role": "system", "content": self.prompt_template},
            {"role": "user",   "content": clean},
        ]

    def categorize(self, text: str) -> Enum:
        msgs = self._build_messages(text)
        resp = self.client.responses.parse(
            model=self.model,
            input=msgs,
            text_format=self._OutputModel,
            temperature=self.temperature,
            **self.client_kwargs,
        )
        output: BaseModel = resp.output_parsed
        return output.category