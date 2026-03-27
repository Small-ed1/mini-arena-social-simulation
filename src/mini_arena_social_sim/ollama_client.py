from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Generic, TypeVar
from urllib import error, request

from pydantic import BaseModel

from .schemas import BackendConfig

ModelT = TypeVar("ModelT", bound=BaseModel)


@dataclass(slots=True)
class StructuredChatResult(Generic[ModelT]):
    parsed: ModelT
    raw_content: str
    response_json: dict[str, Any]


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, config: BackendConfig):
        self.config = config

    def structured_chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        response_model: type[ModelT],
        temperature: float | None = None,
    ) -> StructuredChatResult[ModelT]:
        data = self._post_chat(
            model=model,
            messages=messages,
            response_model=response_model,
            temperature=temperature,
        )
        content = data.get("message", {}).get("content", "")
        if not content:
            raise OllamaError(f"Empty response from Ollama for model {model!r}.")

        try:
            parsed_payload = self._extract_json(content)
        except OllamaError:
            repaired = self._repair_json(
                model=model,
                raw_content=content,
                response_model=response_model,
            )
            if repaired is None:
                raise
            parsed_payload = repaired

        return StructuredChatResult(
            parsed=response_model.model_validate(parsed_payload),
            raw_content=content,
            response_json=data,
        )

    def _post_chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        response_model: type[ModelT],
        temperature: float | None,
    ) -> dict[str, Any]:
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "format": response_model.model_json_schema(),
            "options": {
                "temperature": self.config.temperature
                if temperature is None
                else temperature,
                "num_ctx": self.config.num_ctx,
            },
        }
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            f"{self.config.base_url.rstrip('/')}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(req) as response:
                data = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise OllamaError(f"Ollama HTTP error {exc.code}: {details}") from exc
        except error.URLError as exc:
            raise OllamaError(
                "Could not reach local Ollama. Start `ollama serve` or use `--backend heuristic`."
            ) from exc
        return data

    @staticmethod
    def _extract_json(content: str) -> dict:
        stripped = content.strip()
        candidates: list[str] = [stripped]
        balanced = OllamaClient._balanced_json_object(stripped)
        if balanced and balanced not in candidates:
            candidates.append(balanced)

        for candidate in candidates:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

        raise OllamaError(f"Ollama returned non-JSON content: {content[:300]}")

    @staticmethod
    def _balanced_json_object(content: str) -> str | None:
        start = content.find("{")
        if start < 0:
            return None

        depth = 0
        in_string = False
        escape_next = False
        for index in range(start, len(content)):
            char = content[index]
            if in_string:
                if escape_next:
                    escape_next = False
                elif char == "\\":
                    escape_next = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return content[start : index + 1]
        return None

    def _repair_json(
        self,
        *,
        model: str,
        raw_content: str,
        response_model: type[ModelT],
    ) -> dict[str, Any] | None:
        truncated = raw_content[:24000]
        repair_messages = [
            {
                "role": "system",
                "content": (
                    "You repair malformed model output into one valid JSON object. "
                    "Preserve meaning, omit commentary, and return JSON only."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "malformed_output": truncated,
                        "instruction": (
                            "Convert the malformed output into valid JSON that matches the requested schema. "
                            "Keep fields concise."
                        ),
                    },
                    ensure_ascii=True,
                ),
            },
        ]
        try:
            repair_data = self._post_chat(
                model=model,
                messages=repair_messages,
                response_model=response_model,
                temperature=0.0,
            )
        except OllamaError:
            return None

        repair_content = repair_data.get("message", {}).get("content", "")
        if not repair_content:
            return None
        try:
            return self._extract_json(repair_content)
        except OllamaError:
            return None
