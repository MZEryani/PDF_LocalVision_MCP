from __future__ import annotations

import os
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import httpx

DEFAULT_OMLX_BASE_URL = "http://localhost:8000/v1"
DEFAULT_OMLX_MODEL = "Q-Vision"
DEFAULT_OMLX_API_KEY = "omlx"
DEFAULT_TIMEOUT_SECONDS = 300.0


class OmlxConnectionError(RuntimeError):
    """Raised when the local oMLX server is unreachable or missing the target model."""


def _build_remediation_message(base_url: str, model: str, detail: str) -> str:
    return (
        f"Failed to reach vision model endpoint '{model}' at '{base_url}': {detail}\n"
        "Remediation steps:\n"
        f"  1. Verify the endpoint is running: `curl {base_url}/models`\n"
        "  2. For local oMLX: ensure server is active (`omlx start` or `omlx serve`) with your model alias configured\n"
        "  3. For Ollama / vLLM / other servers: verify the model is loaded and accessible at the configured base URL\n"
        "  4. Check environment variables: VISION_BASE_URL (or OPENAI_BASE_URL / OMLX_BASE_URL) and VISION_MODEL (or OPENAI_MODEL / OMLX_MODEL)"
    )


def _resolve_env_cascade(
    explicit_value: str | None,
    var_suffixes: list[str],
    default_value: str | None = None,
) -> str | None:
    if explicit_value is not None:
        return explicit_value
    for var in var_suffixes:
        val = os.environ.get(var)
        if val is not None and val.strip():
            return val.strip()
    return default_value


class OmlxClient:
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        resolved_base_url = _resolve_env_cascade(
            base_url,
            ["VISION_BASE_URL", "OPENAI_BASE_URL", "OMLX_BASE_URL"],
            DEFAULT_OMLX_BASE_URL,
        )
        self.base_url = (resolved_base_url or DEFAULT_OMLX_BASE_URL).rstrip("/")

        explicit_model = _resolve_env_cascade(
            model,
            ["VISION_MODEL", "OPENAI_MODEL", "OMLX_MODEL"],
            None,
        )
        self.model = explicit_model or DEFAULT_OMLX_MODEL
        self._user_specified_model = explicit_model is not None

        resolved_api_key = _resolve_env_cascade(
            api_key,
            ["OPENAI_API_KEY", "VISION_API_KEY", "OMLX_API_KEY"],
            DEFAULT_OMLX_API_KEY,
        )
        self.api_key = resolved_api_key or DEFAULT_OMLX_API_KEY
        self._external_client = http_client
        self._health_checked = False

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    @asynccontextmanager
    async def _client_session(self) -> AsyncIterator[httpx.AsyncClient]:
        if self._external_client is not None:
            yield self._external_client
        else:
            async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT_SECONDS) as client:
                yield client

    async def _request(self, method: str, endpoint: str, json_body: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        try:
            async with self._client_session() as client:
                response = await client.request(
                    method,
                    url,
                    headers=self._headers(),
                    json=json_body,
                )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as exc:
            raise OmlxConnectionError(
                _build_remediation_message(self.base_url, self.model, str(exc))
            ) from exc

    async def ensure_ready(self) -> None:
        if self._health_checked:
            return
        payload = await self._request("GET", "/models")
        models_data = payload.get("data", [])
        available_ids = [
            str(item.get("id"))
            for item in models_data
            if isinstance(item, dict) and "id" in item and item.get("id") is not None
        ]

        if not available_ids:
            raise OmlxConnectionError(
                _build_remediation_message(
                    self.base_url,
                    self.model,
                    "Endpoint returned an empty model list from /models",
                )
            )

        # 1. Exact match
        if self.model in available_ids:
            self._health_checked = True
            return

        # 2. Case-insensitive, basename, or prefix/suffix match
        matched_id: str | None = None
        target = self.model.lower()
        target_base = target.split("/")[-1]
        for available_id in available_ids:
            candidate_id = available_id.lower()
            candidate_basename = candidate_id.split("/")[-1]
            if (
                target == candidate_id
                or target_base == candidate_basename
                or candidate_id.startswith(target)
                or candidate_id.endswith(target)
                or candidate_basename.startswith(target_base)
                or candidate_basename.endswith(target_base)
            ):
                matched_id = available_id
                break

        if matched_id is not None:
            self.model = matched_id
            self._health_checked = True
            return

        # 3. Auto-select if caller did not explicitly specify a model
        if not self._user_specified_model:
            if len(available_ids) == 1:
                self.model = available_ids[0]
                self._health_checked = True
                return
            vision_candidates = [
                aid
                for aid in available_ids
                if any(v in aid.lower() for v in ("vision", "vl", "llava", "omni", "multimodal"))
            ]
            if len(vision_candidates) == 1:
                self.model = vision_candidates[0]
                self._health_checked = True
                return

        raise OmlxConnectionError(
            _build_remediation_message(
                self.base_url,
                self.model,
                f"Model '{self.model}' not found in available models: {available_ids}",
            )
        )

    async def complete_messages(self, messages: list[dict[str, Any]]) -> str:
        await self.ensure_ready()
        wire_messages = [dict(m) for m in messages]
        if wire_messages and isinstance(wire_messages[0].get("content"), str):
            wire_messages[0]["content"] = f"[Request-ID: {uuid.uuid4()}]\n{wire_messages[0]['content']}"
        payload = await self._request(
            "POST",
            "/chat/completions",
            json_body={
                "model": self.model,
                "messages": wire_messages,
                "temperature": 0.0,
            },
        )
        choices = payload.get("choices", [])
        if not choices:
            return ""
        message = choices[0].get("message", {})
        return str(message.get("content") or "")
