"""API 路由单元测试

使用 FastAPI TestClient 测试 HTTP 接口；LLM 调用由 conftest 中的 FakeZhipuAI mock，
因此测试离线、确定、无需真实 API Key。
"""

from fastapi.testclient import TestClient

from agentforge.api.routes import app

client = TestClient(app)


def test_root():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "AgentForge" in resp.json()["message"]


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


def test_create_and_list_users():
    resp = client.post("/users/create", json={"username": "测试用户"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["user_id"]
    assert data["username"] == "测试用户"

    resp = client.get("/users/list")
    assert resp.status_code == 200
    assert any(u["username"] == "测试用户" for u in resp.json())


def test_get_conversations_404():
    resp = client.get("/users/nonexistent_user/conversations")
    assert resp.status_code == 404


def test_chat():
    resp = client.post("/chat", json={"message": "你好"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["reply"] == "这是测试回答"


def test_chat_stream():
    resp = client.post("/chat/stream", json={"message": "你好"})
    assert resp.status_code == 200
    assert "data:" in resp.text
    assert "[DONE]" in resp.text


def test_workflow():
    resp = client.post("/workflow", json={"task": "写一句问候语"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["approved"] is True
    assert data["subtasks_count"] >= 1
