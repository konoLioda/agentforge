"""用户管理模块

负责多用户的创建、查询与删除，用户数据持久化到 JSON 文件。
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel


class User(BaseModel):
    """用户实体"""

    user_id: str
    username: str
    created_at: str
    last_active: str


class UserManager:
    """用户管理器：维护用户注册信息并持久化到 data/users.json"""

    def __init__(self, data_dir: Optional[str] = None):
        """
        Args:
            data_dir: 数据存储目录，默认使用项目根目录下的 data 目录
        """
        if data_dir is None:
            data_dir = Path(__file__).resolve().parent.parent.parent / "data"

        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.users_file = self.data_dir / "users.json"
        self.users: Dict[str, User] = {}

        # 从磁盘加载已有用户
        self._load_users()

    def _load_users(self) -> None:
        """从 JSON 文件加载用户列表"""
        if self.users_file.exists():
            try:
                with open(self.users_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for uid, udata in data.items():
                        self.users[uid] = User(**udata)
            except Exception:
                self.users = {}

    def _save_users(self) -> None:
        """将用户列表持久化到 JSON 文件"""
        with open(self.users_file, "w", encoding="utf-8") as f:
            data = {uid: u.model_dump() for uid, u in self.users.items()}
            json.dump(data, f, ensure_ascii=False, indent=2)

    def get_or_create_user(self, username: str) -> User:
        """
        按用户名获取用户，不存在则创建。

        Args:
            username: 用户名

        Returns:
            已有或新建的 User 对象
        """
        # 已存在则更新活跃时间
        for user in self.users.values():
            if user.username == username:
                user.last_active = datetime.now().isoformat()
                self._save_users()
                return user

        # 创建新用户
        user_id = f"user_{len(self.users) + 1:04d}"
        now = datetime.now().isoformat()
        user = User(
            user_id=user_id,
            username=username,
            created_at=now,
            last_active=now,
        )
        self.users[user_id] = user
        self._save_users()

        # 为用户创建对话存储目录
        user_dir = self.data_dir / "users" / user_id
        user_dir.mkdir(parents=True, exist_ok=True)

        return user

    def list_users(self) -> List[User]:
        """返回所有用户"""
        return list(self.users.values())

    def get_user(self, user_id: str) -> Optional[User]:
        """按 ID 获取用户，不存在返回 None"""
        return self.users.get(user_id)

    def delete_user(self, user_id: str) -> bool:
        """删除指定用户，返回是否删除成功"""
        if user_id not in self.users:
            return False

        # 删除该用户的对话目录
        user_dir = self.data_dir / "users" / user_id
        if user_dir.exists():
            shutil.rmtree(user_dir)

        del self.users[user_id]
        self._save_users()
        return True
