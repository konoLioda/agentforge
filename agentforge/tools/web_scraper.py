"""网页抓取工具

抓取网页 HTML 并提取纯文本，供智能体阅读网页内容。
"""

import re

import httpx
from langchain_core.tools import Tool
from pydantic import BaseModel, Field


class WebScraperInput(BaseModel):
    """网页抓取工具的输入参数"""

    url: str = Field(description="目标网页 URL")


def scrape_webpage(url: str) -> str:
    """
    抓取网页并提取纯文本。

    Args:
        url: 目标网页 URL

    Returns:
        提取后的纯文本（截断到前 5000 字符）
    """
    try:
        with httpx.Client(timeout=10.0, follow_redirects=True) as client:
            response = client.get(url)
            response.raise_for_status()
            html = response.text
            # 先移除 script/style 标签及其内容，再剥离其余 HTML 标签
            text = re.sub(r"<script.*?</script>", "", html, flags=re.DOTALL)
            text = re.sub(r"<style.*?</style>", "", text, flags=re.DOTALL)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            return text[:5000]
    except Exception as exc:
        return f"错误：{exc}"


web_scraper_tool = Tool(
    name="web_scraper",
    description="网页抓取工具。输入网页 URL，返回去除 HTML 标签后的纯文本内容。",
    func=scrape_webpage,
    args_schema=WebScraperInput,
)

__all__ = ["web_scraper_tool", "scrape_webpage"]
