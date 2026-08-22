"""通用 HTTP API 调用工具

允许智能体通过 HTTP 协议调用外部 RESTful API，支持 GET/POST/PUT/DELETE。
"""

import json
from typing import Any, Dict, Optional

import httpx
from langchain_core.tools import Tool
from pydantic import BaseModel, Field


class ApiCallInput(BaseModel):
    """API 调用工具的输入参数"""

    method: str = Field(description="HTTP 方法：GET, POST, PUT, DELETE")
    url: str = Field(description="目标 API URL")
    headers: Optional[str] = Field(default="{}", description="JSON 格式的请求头")
    body: Optional[str] = Field(default="{}", description="JSON 格式的请求体")


def call_api(method: str, url: str, headers: str = "{}", body: str = "{}") -> str:
    """
    执行一次 HTTP API 调用并返回结果摘要。

    Args:
        method: HTTP 方法（GET/POST/PUT/DELETE）
        url: 请求地址
        headers: JSON 字符串形式的请求头
        body: JSON 字符串形式的请求体（GET 时作为查询参数）

    Returns:
        包含状态码、响应头和响应体的 JSON 字符串
    """
    try:
        headers_dict: Dict[str, Any] = json.loads(headers) if headers else {}
        body_dict: Dict[str, Any] = json.loads(body) if body else {}

        with httpx.Client(timeout=15.0) as client:
            method_upper = method.upper()
            if method_upper == "GET":
                response = client.get(url, headers=headers_dict, params=body_dict)
            elif method_upper == "POST":
                response = client.post(url, headers=headers_dict, json=body_dict)
            elif method_upper == "PUT":
                response = client.put(url, headers=headers_dict, json=body_dict)
            elif method_upper == "DELETE":
                response = client.delete(url, headers=headers_dict)
            else:
                return f"错误：不支持的 HTTP 方法 {method}"

            result = {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "body": response.text[:2000],
            }
            return json.dumps(result, ensure_ascii=False, indent=2)
    except json.JSONDecodeError as exc:
        return f"错误：JSON 解析失败 - {exc}"
    except Exception as exc:
        return f"错误：{exc}"


api_call_tool = Tool(
    name="call_api",
    description="通用 RESTful API 调用工具。输入 HTTP 方法、URL、headers 和 body（JSON 字符串），返回响应结果摘要。",
    func=call_api,
    args_schema=ApiCallInput,
)

__all__ = ["api_call_tool", "call_api"]
