"""图片生成工具

使用智谱 CogView 模型根据文本描述生成图片并保存到本地。
"""

import httpx
from pathlib import Path

from langchain_core.tools import Tool
from pydantic import BaseModel, Field
from zhipuai import ZhipuAI

from config.logging_config import get_logger
from config.settings import settings

logger = get_logger("image_gen")


class ImageGenInput(BaseModel):
    """图片生成工具的输入参数"""

    prompt: str = Field(description="图片描述文本")


def generate_image(prompt: str) -> str:
    """
    根据文本描述生成图片。

    Args:
        prompt: 图片描述文本

    Returns:
        生成结果的文本描述（包含保存路径或错误信息）
    """
    try:
        client = ZhipuAI(api_key=settings.ZHIPU_API_KEY)
        response = client.images.generations(
            model="cogview-3-plus",
            prompt=prompt,
        )

        if not response.data or len(response.data) == 0:
            return "错误：图片生成返回空结果"

        image_url = response.data[0].url

        output_dir = Path(__file__).resolve().parent.parent.parent / "data" / "images"
        output_dir.mkdir(parents=True, exist_ok=True)

        with httpx.Client(timeout=30.0) as http_client:
            img_response = http_client.get(image_url)
            img_response.raise_for_status()

            filename = f"gen_{hash(prompt) % 100000}.png"
            filepath = output_dir / filename
            with open(filepath, "wb") as f:
                f.write(img_response.content)

        logger.info(f"图片生成成功：{filepath}")
        return f"图片生成成功：{filepath}\n描述：{prompt}"
    except Exception as exc:
        logger.error(f"图片生成失败：{exc}")
        return f"错误：图片生成失败 - {exc}"


image_gen_tool = Tool(
    name="image_gen",
    description="图片生成工具。根据文本描述生成图片（使用智谱 CogView 模型），并保存到本地。",
    func=generate_image,
    args_schema=ImageGenInput,
)

__all__ = ["image_gen_tool", "generate_image"]
