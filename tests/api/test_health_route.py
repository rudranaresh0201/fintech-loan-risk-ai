"""API tests for health endpoint."""

from __future__ import annotations

from backend.routes import health


def test_get_health_ok(client, monkeypatch) -> None:
    """GET /health should report healthy when artifacts load."""

    monkeypatch.setattr(health, "load_artifacts", lambda: object())

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["models_loaded"] is True


def test_get_health_degraded(client, monkeypatch) -> None:
    """GET /health should report degraded when artifacts fail loading."""

    def fail_loader() -> object:
        raise RuntimeError("missing model file")

    monkeypatch.setattr(health, "load_artifacts", fail_loader)

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "degraded"
    assert payload["models_loaded"] is False
    assert "missing model file" in payload["message"]
