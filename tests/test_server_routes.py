from __future__ import annotations

from fastapi.testclient import TestClient

from src.server.app import app

client = TestClient(app)


class TestRoot:
    def test_root_returns_info(self) -> None:
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "my_custom_llm"
        assert "endpoints" in data
        assert "capabilities" in data

    def test_capabilities_listed(self) -> None:
        response = client.get("/")
        data = response.json()
        caps = data["capabilities"]
        assert caps["rag"] is True
        assert caps["tool_calling"] is True
        assert caps["voice"] is True
        assert caps["vision"] is True


class TestModels:
    def test_health_endpoint(self) -> None:
        response = client.get("/api/models/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data

    def test_list_models(self) -> None:
        response = client.get("/api/models")
        assert response.status_code == 200


class TestChat:
    def test_chat_requires_post(self) -> None:
        response = client.get("/api/chat")
        assert response.status_code == 405


class TestGenerate:
    def test_generate_text_endpoint(self) -> None:
        response = client.post("/api/generate/text", json={"prompt": "Hello"})
        assert response.status_code in (200, 422)


class TestMemory:
    def test_list_memory(self) -> None:
        response = client.get("/api/memory")
        assert response.status_code == 200

    def test_remember_and_recall(self) -> None:
        client.post("/api/memory/remember", json={"key": "test_key", "value": "test_value"})
        response = client.get("/api/memory/recall/test_key")
        assert response.status_code == 200


class TestRAG:
    def test_rag_status(self) -> None:
        response = client.get("/api/rag/status")
        assert response.status_code == 200


class TestStats:
    def test_stats_endpoint(self) -> None:
        response = client.get("/api/stats")
        assert response.status_code == 200


class TestBranching:
    def test_list_templates(self) -> None:
        response = client.get("/api/branch/templates")
        assert response.status_code == 200

    def test_create_branch(self) -> None:
        response = client.post("/api/branch/create", json={"session_id": "test_sess", "label": "test"})
        assert response.status_code == 200
        data = response.json()
        assert "branch_id" in data


class TestPlugins:
    def test_list_plugins(self) -> None:
        response = client.get("/api/plugins")
        assert response.status_code == 200
