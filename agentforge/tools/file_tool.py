"""文件读写工具

提供读取文件、写入文件、列出目录、解压 ZIP 的能力，供智能体操作本地文件。
"""

import zipfile
from pathlib import Path
from typing import Optional

from langchain_core.tools import Tool
from pydantic import BaseModel, Field


class ReadFileInput(BaseModel):
    """读取文件的输入参数"""

    file_path: str = Field(description="文件路径")


class WriteFileInput(BaseModel):
    """写入文件的输入参数"""

    file_path: str = Field(description="文件路径")
    content: str = Field(description="文件内容")


class ListFilesInput(BaseModel):
    """列出目录的输入参数"""

    directory: Optional[str] = Field(default=".", description="目录路径")


def read_file(file_path: str) -> str:
    """读取文本文件内容"""
    path = Path(file_path)
    if not path.exists():
        return f"错误：文件不存在 {file_path}"
    if not path.is_file():
        return f"错误：不是文件 {file_path}"
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as exc:
        return f"错误：{exc}"


def write_file(file_path: str, content: str) -> str:
    """写入文本文件（相对路径基于项目根目录）"""
    if not Path(file_path).is_absolute():
        project_root = Path(__file__).resolve().parent.parent.parent
        file_path = str(project_root / file_path)
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"写入成功：{file_path}"
    except Exception as exc:
        return f"错误：{exc}"


def list_directory(directory: str = ".") -> str:
    """列出目录内容"""
    dir_path = Path(directory).resolve()
    if not dir_path.exists():
        return f"错误：目录不存在 {directory}"
    if not dir_path.is_dir():
        return f"错误：不是目录 {directory}"
    items = []
    for item in sorted(dir_path.iterdir()):
        if item.is_dir():
            items.append(f"[DIR] {item.name}")
        else:
            items.append(f"[FILE] {item.name} ({item.stat().st_size} bytes)")
    return "\n".join(items) if items else "（空目录）"


def extract_zip(zip_path: str, extract_to: Optional[str] = None) -> str:
    """解压 ZIP 文件"""
    zip_obj = Path(zip_path)
    if not zip_obj.exists() or zip_obj.suffix != ".zip":
        return f"错误：无效的 ZIP 文件 {zip_path}"
    extract_dir = Path(extract_to) if extract_to else zip_obj.parent
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_dir)
        return f"解压成功：{extract_dir}"
    except Exception as exc:
        return f"错误：{exc}"


read_file_tool = Tool(
    name="read_file",
    description="读取文件内容。参数 file_path 是文件路径。示例输入: {\"file_path\": \"test.txt\"}",
    func=read_file,
    args_schema=ReadFileInput,
)

write_file_tool = Tool(
    name="write_file",
    description="写入文件。参数 file_path 是文件路径，content 是文件内容。示例输入: {\"file_path\": \"test.txt\", \"content\": \"hello\"}",
    func=write_file,
    args_schema=WriteFileInput,
)

list_files_tool = Tool(
    name="list_files",
    description="列出目录内容。参数 directory 是目录路径，默认为当前目录。示例输入: {\"directory\": \".\"}",
    func=list_directory,
    args_schema=ListFilesInput,
)

extract_zip_tool = Tool(
    name="extract_zip",
    description="解压 ZIP 压缩包到指定目录。",
    func=extract_zip,
)

__all__ = ["read_file_tool", "write_file_tool", "list_files_tool", "extract_zip_tool"]
