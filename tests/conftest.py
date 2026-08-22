"""pytest 共享 fixtures

通过 mock 外部 LLM 客户端（zhipuai.ZhipuAI），让测试离线、确定、可重复运行。
"""

import pytest


class _FakeMessage:
    """模拟 LLM 返回的消息对象"""

    def __init__(self, content: str):
        self.content = content


class _FakeChoice:
    """模拟 completion.choices 中的单个选项"""

    def __init__(self, content: str):
        self.message = _FakeMessage(content)


class _FakeResponse:
    """模拟非流式 completion 响应"""

    def __init__(self, content: str):
        self.choices = [_FakeChoice(content)]


class _FakeStreamChunk:
    """模拟流式响应中的单个 chunk"""

    def __init__(self, content: str):
        self.choices = [type("_C", (), {"delta": _FakeMessage(content)})()]


def _content_for(messages) -> str:
    """根据系统提示词返回合适的假响应内容"""
    system = messages[0]["content"] if messages else ""
    if "任务规划专家" in system:
        return (
            '{"subtasks": [{"id": 1, "description": "测试子任务", '
            '"depends_on": [], "assigned_to": "worker"}], "total_count": 1}'
        )
    if "质量审查专家" in system:
        return (
            '{"score": 8, "strengths": ["内容完整"], '
            '"improvements": ["可进一步细化"], "approved": true, '
            '"feedback": "审查通过，质量良好。"}'
        )
    return "Thought: 分析问题\nFinal Answer: 这是测试回答"


class _FakeCompletions:
    """模拟 client.chat.completions"""

    def create(self, **kwargs):
        content = _content_for(kwargs.get("messages", []))
        if kwargs.get("stream"):
            return [_FakeStreamChunk(ch) for ch in content]
        return _FakeResponse(content)


class _FakeChat:
    """模拟 client.chat"""

    def __init__(self):
        self.completions = _FakeCompletions()


class FakeZhipuAI:
    """替代 zhipuai.ZhipuAI 的假客户端，不发起任何网络请求"""

    def __init__(self, api_key=None, **kwargs):
        self.api_key = api_key
        self.chat = _FakeChat()


@pytest.fixture(autouse=True)
def mock_zhipuai(monkeypatch):
    """自动将各模块中的 ZhipuAI 替换为 FakeZhipuAI"""
    modules = [
        "agentforge.agents.react_agent",
        "agentforge.agents.planner_agent",
        "agentforge.agents.worker_agent",
        "agentforge.agents.reviewer_agent",
        "agentforge.tools.image_gen",
    ]
    for name in modules:
        monkeypatch.setattr(f"{name}.ZhipuAI", FakeZhipuAI)
