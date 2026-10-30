from __future__ import annotations

import httpx
import pytest

from pdf_localvision_mcp.omlx_client import (
    DEFAULT_OMLX_BASE_URL,
    DEFAULT_OMLX_MODEL,
    OmlxClient,
    OmlxConnectionError,
)


async def test_endpoint_resolution_prefers_generic_vision_and_openai_env_vars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("OPENAI_MODEL", "llama3.2-vision")
    monkeypatch.setenv("OPENAI_API_KEY", "ollama-key")
    monkeypatch.setenv("OMLX_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("OMLX_MODEL", "Q-Vision")
    monkeypatch.setenv("OMLX_API_KEY", "omlx-key")

    client = OmlxClient()
    assert client.base_url == "http://localhost:11434/v1"
    assert client.model == "llama3.2-vision"
    assert client.api_key == "ollama-key"


async def test_endpoint_resolution_vision_prefix_overrides_openai_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VISION_BASE_URL", "http://localhost:9000/v1")
    monkeypatch.setenv("VISION_MODEL", "qwen-vl")
    monkeypatch.setenv("VISION_API_KEY", "vision-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("OPENAI_MODEL", "llama-vision")

    client = OmlxClient()
    assert client.base_url == "http://localhost:9000/v1"
    assert client.model == "qwen-vl"
    assert client.api_key == "vision-key"


async def test_endpoint_resolution_falls_back_to_omlx_env_and_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for var in [
        "VISION_BASE_URL",
        "VISION_MODEL",
        "VISION_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_MODEL",
        "OPENAI_API_KEY",
        "OMLX_BASE_URL",
        "OMLX_MODEL",
        "OMLX_API_KEY",
    ]:
        monkeypatch.delenv(var, raising=False)

    client = OmlxClient()
    assert client.base_url == DEFAULT_OMLX_BASE_URL
    assert client.model == DEFAULT_OMLX_MODEL
    assert client.api_key == "omlx"


async def test_ensure_ready_fuzzy_matches_model_id() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={"data": [{"id": "mlx-community/Qwen2.5-VL-7B-Instruct-8bit"}]},
            )
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = OmlxClient(
            base_url="http://mock/v1",
            model="Qwen2.5-VL-7B-Instruct",
            http_client=http_client,
        )
        await client.ensure_ready()
        assert client.model == "mlx-community/Qwen2.5-VL-7B-Instruct-8bit"


async def test_ensure_ready_auto_selects_sole_model_when_default_q_vision_not_found() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={"data": [{"id": "meta-llama/Llama-3.2-11B-Vision-Instruct"}]},
            )
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = OmlxClient(
            base_url="http://mock/v1",
            http_client=http_client,
        )
        await client.ensure_ready()
        assert client.model == "meta-llama/Llama-3.2-11B-Vision-Instruct"


async def test_ensure_ready_raises_generic_remediation_on_mismatch() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={"data": [{"id": "some-text-model-1"}, {"id": "some-text-model-2"}]},
            )
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = OmlxClient(
            base_url="http://mock/v1",
            model="custom-vision-model",
            http_client=http_client,
        )
        with pytest.raises(OmlxConnectionError) as exc_info:
            await client.ensure_ready()

        msg = str(exc_info.value)
        assert "custom-vision-model" in msg
        assert "Verify the endpoint is running" in msg
        assert "VISION_BASE_URL" in msg


def test_project_metadata_and_license_are_complete() -> None:
    import tomllib
    from pathlib import Path

    root_dir = Path(__file__).resolve().parent.parent
    pyproject_path = root_dir / "pyproject.toml"
    license_path = root_dir / "LICENSE"

    assert pyproject_path.is_file()
    assert license_path.is_file()

    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)

    project = data.get("project", {})
    assert project.get("name") == "pdf-localvision-mcp"
    assert project.get("license") == "MIT"
    assert len(project.get("authors", [])) > 0
    assert "urls" in project
    assert "Homepage" in project["urls"]
    assert "Repository" in project["urls"]

