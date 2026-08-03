"""Build integration checks for the compiled React Studio."""

from __future__ import annotations

import json
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent


def test_studio_build_is_local_vite_react_with_no_cdn():
    package = json.loads((REPO / "studio" / "package.json").read_text())
    assert package["scripts"]["build"] == "tsc -b && vite build"
    assert "react" in package["dependencies"]
    assert package["dependencies"]["@modelcontextprotocol/sdk"] == "1.30.0"
    assert "vite" in package["devDependencies"]
    assert package["devDependencies"]["@playwright/test"] == "1.62.1"
    assert package["scripts"]["test:browser"] == "playwright test"
    for source in (REPO / "studio" / "src").rglob("*"):
        if source.is_file():
            assert "https://" not in source.read_text(), source


def test_studio_uses_same_origin_mcp_without_inspector_proxy():
    source = (REPO / "studio" / "src" / "mcp.ts").read_text()
    assert 'endpoint = "/mcp"' in source
    assert "window.location.origin" in source
    assert "StreamableHTTPClientTransport" in source
    assert "inspector" not in source.lower()
    assert "proxy" not in source.lower()


def test_docker_builds_studio_before_installing_python_package():
    dockerfile = (REPO / "Dockerfile").read_text()
    assert "FROM node:24-bookworm-slim AS studio-builder" in dockerfile
    assert "RUN npm ci" in dockerfile
    assert "RUN npm run build" in dockerfile
    assert "COPY --from=studio-builder /snoocle_server/studio ./snoocle_server/studio" in dockerfile


def test_python_version_build_arg_is_global_for_every_python_stage():
    dockerfile = (REPO / "Dockerfile").read_text()
    instructions = [
        (line_number, line.strip())
        for line_number, line in enumerate(dockerfile.splitlines(), start=1)
        if line.strip() and not line.lstrip().startswith("#")
    ]
    first_from = next(line_number for line_number, line in instructions if line.startswith("FROM "))
    python_arg = next(
        line_number for line_number, line in instructions if line == "ARG PYTHON_VERSION=3.11"
    )

    assert python_arg < first_from, (
        "PYTHON_VERSION must be declared before the first FROM so Docker can "
        "expand it in every later Python stage"
    )
    assert dockerfile.count("FROM python:${PYTHON_VERSION}-slim-bookworm") == 2
