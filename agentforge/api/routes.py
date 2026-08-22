"""FastAPI 路由模块

提供 AgentForge 的 HTTP RESTful API，包括：
- 对话接口（/chat、/chat/stream）
- 多智能体工作流接口（/workflow）
- 用户管理接口（/users/*）
- 健康检查（/health）
"""

import json
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agentforge.agents.react_agent import ReactAgent
from agentforge.graph.workflow import run_workflow
from agentforge.user.conversation_store import ConversationStore
from agentforge.user.user_manager import UserManager
from config.logging_config import get_logger

logger = get_logger("api")

# 全局用户管理器（进程内单例）
user_manager = UserManager()

# 按 user_id 缓存智能体实例，保持每个用户的会话上下文
agents_cache: Dict[str, ReactAgent] = {}

# FastAPI 应用实例
app = FastAPI(
    title="AgentForge API",
    description="基于 LangChain + LangGraph 的多智能体协作框架 HTTP 接口",
    version="1.0.0",
)


class ChatRequest(BaseModel):
    """对话请求"""

    message: str
    user_id: Optional[str] = None
    use_memory: bool = True


class WorkflowRequest(BaseModel):
    """工作流请求"""

    task: str
    user_id: Optional[str] = None
    max_iterations: int = 3


class ChatResponse(BaseModel):
    """对话响应"""

    reply: str
    thinking: Optional[str] = None
    action_used: Optional[str] = None
    user_id: Optional[str] = None


class WorkflowResponse(BaseModel):
    """工作流响应"""

    final_answer: str
    subtasks_count: int
    iteration: int
    approved: bool
    review_feedback: Optional[str] = None


class UserResponse(BaseModel):
    """用户信息响应"""

    user_id: str
    username: str
    created_at: str
    message_count: int


class CreateUserRequest(BaseModel):
    """创建用户请求"""

    username: str


@app.get("/")
async def root() -> Dict[str, Any]:
    """根路径：返回服务信息"""
    return {
        "message": "AgentForge API",
        "version": "1.0.0",
        "features": ["react_agent", "multi_agent_workflow", "user_system", "memory"],
    }


@app.post("/users/create", response_model=UserResponse)
async def create_user(request: CreateUserRequest) -> UserResponse:
    """创建用户（已存在则复用）"""
    user = user_manager.get_or_create_user(request.username)
    store = ConversationStore(user.user_id)
    return UserResponse(
        user_id=user.user_id,
        username=user.username,
        created_at=user.created_at,
        message_count=store.count(),
    )


@app.get("/users/list")
async def list_users() -> list:
    """列出所有用户"""
    users = user_manager.list_users()
    return [{"user_id": u.user_id, "username": u.username} for u in users]


@app.get("/users/{user_id}/conversations")
async def get_conversations(user_id: str) -> list:
    """获取指定用户的对话历史"""
    user = user_manager.get_user(user_id)
    if not user:
        raise HTTPException(404, "用户不存在")
    store = ConversationStore(user_id)
    return store.get_history()


def _get_agent(user_id: Optional[str], message: str) -> tuple[ReactAgent, Optional[ConversationStore]]:
    """按用户获取（或创建）智能体实例，并记录用户消息，返回 (agent, store)"""
    store: Optional[ConversationStore] = None
    if user_id:
        if user_id not in agents_cache:
            agents_cache[user_id] = ReactAgent()
        agent = agents_cache[user_id]
        store = ConversationStore(user_id)
        store.add_message("user", message)
    else:
        agent = ReactAgent()
    return agent, store


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """单轮对话"""
    logger.info(f"[api] /chat user={request.user_id} msg={request.message[:30]}...")
    agent, store = _get_agent(request.user_id, request.message)
    try:
        response = agent.think(request.message)
        if store:
            store.add_message("assistant", response.content)
        return ChatResponse(
            reply=response.content,
            thinking=response.thinking,
            action_used=response.action,
            user_id=request.user_id,
        )
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.post("/workflow", response_model=WorkflowResponse)
async def workflow(request: WorkflowRequest) -> WorkflowResponse:
    """运行多智能体工作流"""
    try:
        result = run_workflow(request.task, request.max_iterations)
        if request.user_id:
            store = ConversationStore(request.user_id)
            store.add_message("user", f"[工作流] {request.task}")
            store.add_message("assistant", result.get("final_answer", ""))
        review = result.get("review_result") or {}
        return WorkflowResponse(
            final_answer=result.get("final_answer", ""),
            subtasks_count=len(result.get("subtasks", [])),
            iteration=result.get("iteration", 0),
            approved=review.get("approved", False),
            review_feedback=review.get("feedback"),
        )
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.get("/health")
async def health() -> Dict[str, str]:
    """健康检查"""
    return {"status": "healthy"}


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """流式对话（SSE）"""
    agent, store = _get_agent(request.user_id, request.message)

    async def generate():
        full_response = ""
        for chunk in agent.think_stream(request.message):
            full_response += chunk
            data = {"content": chunk}
            yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
        if store:
            store.add_message("assistant", full_response)
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


def start_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    """启动 Uvicorn 服务"""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start_server()
