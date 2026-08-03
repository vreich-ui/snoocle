"""Build integration checks for the compiled React Studio."""

from __future__ import annotations

import json
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent


def test_studio_build_is_local_vite_react_with_no_cdn():
    package = json.loads((REPO / "studio" / "package.json").read_text())
    assert package["scripts"]["build"] == "tsc -b && vite build"
    assert "react" in package["dependencies"]
    assert "vite" in package["devDependencies"]
    for source in (REPO / "studio" / "src").rglob("*"):
        if source.is_file():
            assert "https://" not in source.read_text(), source


def test_docker_builds_studio_before_installing_python_package():
    dockerfile = (REPO / "Dockerfile").read_text()
    assert "FROM node:24-bookworm-slim AS studio-builder" in dockerfile
    assert "RUN npm ci" in dockerfile
    assert "RUN npm run build" in dockerfile
    assert "COPY --from=studio-builder /snoocle_server/studio ./snoocle_server/studio" in dockerfile
