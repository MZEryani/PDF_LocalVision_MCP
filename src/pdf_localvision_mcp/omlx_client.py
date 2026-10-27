from __future__ import annotations

import os
from typing import Any

import httpx

DEFAULT_OMLX_BASE_URL = "http://localhost:8000/v1"
DEFAULT_OMLX_MODEL = "Q-Vision"
DEFAULT_OMLX_API_KEY = "omlx"
DEFAULT_TIMEOUT_SECONDS = 300.0


class OmlxConnectionError(RuntimeError):
    """Raised when the local oMLX server is unreachable or missing the target model."""


def _build_remediation_message(base_url: str, model: str, detail: str) -> str:
    return (
        f"Failed to reach local oMLX vision model '{model}' at '{base_url}': {detail}\n"
        "Remediation steps:\n"
        "  1. Start the managed oMLX background server: `omlx start`\n"
        f"  2. Or launch the OpenAI-compatible server directly: `omlx serve` (with alias '{model}' configured)\n"
        "  3. Verify the endpoint with: `curl http://localhost:8000/v1/models`"
    )


class OmlxClient:
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = (base_url or os.environ.get("OMLX_BASE_URL") or DEFAULT_OMLX_BASE_URL).rstrip("/")
        self.model = model or os.environ.get("OMLX_MODEL") or DEFAULT_OMLX_MODEL
        self.api_key = api_key or os.environ.get("OMLX_API_KEY") or DEFAULT_OMLX_API_KEY
        self._external_client = http_client
        self._health_checked = False

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def _request(self, method: str, endpoint: str, json_body: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        try:
            if self._external_client is not None:
                response = await self._external_client.request(
                    method,
                    url,
                    headers=self._headers(),
                    json=json_body,
                )
            else:
                async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT_SECONDS) as client:
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
        available_ids = [item.get("id") for item in models_data if isinstance(item, dict) and "id" in item]
        if available_ids and self.model not in available_ids:
            raise OmlxConnectionError(
                _build_remediation_message(
                    self.base_url,
                    self.model,
                    f"Model '{self.model}' not found in available oMLX models: {available_ids}",
                )
            )
        self._health_checked = True

    async def complete_messages(self, messages: list[dict[str, Any]]) -> str:
        await self.ensure_ready()
        payload = await self._request(
            "POST",
            "/chat/completions",
            json_body={
                "model": self.model,
                "messages": messages,
                "temperature": 0.1,
            },
        )
        choices = payload.get("choices", [])
        if not choices:
            return ""
        message = choices[0].get("message", {})
        return str(message.get("content") or "")
