"""
角色状态管理模块

内核层通过此模块动态更新角色状态，状态会附加到角色提示词中，
形成"记忆+成长"效果。每个角色对应一个 JSON 状态文件。
"""

import json
from pathlib import Path
from typing import Any


def _sanitize(obj: Any) -> Any:
    """递归清洗数据结构中的所有字符串，移除非法代理字符"""
    if isinstance(obj, str):
        return obj.encode("utf-8", errors="replace").decode("utf-8")
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


class StateManager:
    def __init__(self, state_dir: str | Path):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(exist_ok=True)

    def _state_path(self, character_name: str) -> Path:
        return self.state_dir / f"{character_name}.json"

    def load_state(self, character_name: str) -> dict:
        path = self._state_path(character_name)
        if path.exists():
            return _sanitize(json.loads(path.read_text(encoding="utf-8")))
        return {
            "name": character_name,
            "current_mood": "平静",
            "physical_state": "正常",
            "recent_experiences": [],
            "relationship_changes": {},
            "scene_count": 0,
        }

    def save_state(self, character_name: str, state: dict):
        path = self._state_path(character_name)
        cleaned = _sanitize(state)
        path.write_text(
            json.dumps(cleaned, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def get_effective_prompt(self, base_prompt: str, character_name: str) -> str:
        """合并基础提示词和当前状态，生成角色实际使用的提示词"""
        state = self.load_state(character_name)
        lines = ["\n## 【当前状态 - 内核层动态维护】"]
        lines.append(f"- 情绪：{state.get('current_mood', '平静')}")
        lines.append(f"- 身体状况：{state.get('physical_state', '正常')}")

        if state.get("recent_experiences"):
            lines.append("- 近期经历：")
            for exp in state["recent_experiences"][-3:]:
                lines.append(f"  · {exp}")

        if state.get("relationship_changes"):
            lines.append("- 关系变化：")
            for person, change in state["relationship_changes"].items():
                lines.append(f"  · {person}：{change}")

        return base_prompt + "\n".join(lines)

    def apply_updates(self, updates: dict):
        """批量更新角色状态（由内核层调用）"""
        updates = _sanitize(updates)
        for name, state_updates in updates.items():
            current = self.load_state(name)
            current.update(state_updates)
            current["scene_count"] = current.get("scene_count", 0) + 1

            # 保持 recent_experiences 不超过 10 条
            if "recent_experiences" in current:
                current["recent_experiences"] = current["recent_experiences"][-10:]

            self.save_state(name, current)

    def get_all_states(self) -> dict:
        """获取所有角色当前状态（用于内核层规划）"""
        states = {}
        for path in self.state_dir.glob("*.json"):
            name = path.stem
            states[name] = _sanitize(json.loads(path.read_text(encoding="utf-8")))
        return states

    def reset_all(self):
        """重置所有角色状态"""
        for path in self.state_dir.glob("*.json"):
            path.unlink()
