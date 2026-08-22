"""对话存储模块

负责单个用户的对话历史持久化，使用 JSON 文件存储。
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class ConversationStore:
    """对话存储：管理单个用户的对话历史并持久化到 JSON 文件"""

    def __init__(self, user_id: str, data_dir: Optional[str] = None):
        """
        Args:
            user_id: 用户 ID
            data_dir: 对话存储目录，默认使用 data/users/{user_id}
        """
        if data_dir is None:
            data_dir = Path(__file__).resolve().parent.parent.parent / "data" / "users" / user_id

        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.conversations_file = self.data_dir / "conversations.json"
        self.conversations: List[Dict] = []
        self._load()

    def _load(self) -> None:
        """从 JSON 文件加载对话历史"""
        if self.conversations_file.exists():
            try:
                with open(self.conversations_file, "r", encoding="utf-8") as f:
                    self.conversations = json.load(f)
            except Exception:
                self.conversations = []

    def _save(self) -> None:
        """将对话历史持久化到 JSON 文件"""
        with open(self.conversations_file, "w", encoding="utf-8") as f:
            json.dump(self.conversations, f, ensure_ascii=False, indent=2)

    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None) -> None:
        """
        追加一条消息。

        Args:
            role: 角色（user/assistant）
            content: 消息内容
            metadata: 附加元数据
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        }
        if metadata:
            message["metadata"] = metadata

        self.conversations.append(message)
        self._save()

    def get_history(self, limit: Optional[int] = None) -> List[Dict]:
        """
        获取对话历史。

        Args:
            limit: 最多返回条数，None 表示返回全部

        Returns:
            消息列表
        """
        if limit and limit < len(self.conversations):
            return self.conversations[-limit:]
        return self.conversations

    def clear(self) -> None:
        """清空对话历史"""
        self.conversations = []
        self._save()

    def count(self) -> int:
        """返回消息条数"""
        return len(self.conversations)

    def export_text(self) -> str:
        """导出对话为纯文本"""
        lines = []
        for msg in self.conversations:
            role = "用户" if msg["role"] == "user" else "助手"
            lines.append(f"[{role}] {msg['content']}")
        return "\n".join(lines)
