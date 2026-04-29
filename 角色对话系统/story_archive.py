"""
故事存档模块

将每次运行的内核层产出、角色对话、角色状态全部结构化写入文件，
支持继承续写、回看复盘。
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


def _sanitize(obj: Any) -> Any:
    """递归清洗所有字符串中的非法代理字符"""
    if isinstance(obj, str):
        return obj.encode("utf-8", errors="replace").decode("utf-8")
    if isinstance(obj, dict):
        return {_sanitize(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


class StoryArchive:
    """故事存档管理器"""

    def __init__(self, archive_dir: str | Path):
        self.archive_dir = Path(archive_dir)
        self.archive_dir.mkdir(parents=True, exist_ok=True)

    # ========== 写入 ==========

    def save_meta(self, story_premise: str, character_names: list[str], model: str):
        """保存/更新故事元信息"""
        meta_path = self.archive_dir / "meta.json"
        meta = {
            "story_premise": story_premise,
            "character_names": character_names,
            "model": model,
            "created_at": datetime.now().isoformat(),
            "total_scenes": self._count_scenes(),
            "last_updated": datetime.now().isoformat(),
        }
        if meta_path.exists():
            existing = json.loads(meta_path.read_text(encoding="utf-8"))
            existing.update(meta)
            existing["created_at"] = existing.get("created_at", meta["created_at"])
            meta = existing
        meta_path.write_text(
            json.dumps(_sanitize(meta), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def save_scene(
        self,
        scene_number: int,
        kernel_raw: dict,
        final_directive: dict,
        dialogue_text: str,
        summary: str,
        states_before: dict,
        states_after: dict,
    ):
        """保存单场场景的全部产出"""
        scene_dir = self.archive_dir / f"scene_{scene_number:03d}"
        scene_dir.mkdir(parents=True, exist_ok=True)

        # 1. 内核层原始输出
        (scene_dir / "kernel_raw.json").write_text(
            json.dumps(_sanitize(kernel_raw), ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # 2. 最终场景指令
        (scene_dir / "directive.json").write_text(
            json.dumps(_sanitize(final_directive), ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # 3. 角色对话原文
        (scene_dir / "dialogue.txt").write_text(_sanitize(dialogue_text), encoding="utf-8")

        # 4. 场景摘要
        (scene_dir / "summary.txt").write_text(_sanitize(summary), encoding="utf-8")

        # 5. 场景开始前角色状态
        (scene_dir / "states_before.json").write_text(
            json.dumps(_sanitize(states_before), ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # 6. 场景结束后角色状态
        (scene_dir / "states_after.json").write_text(
            json.dumps(_sanitize(states_after), ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # 更新 meta 中的场景计数
        self.save_meta(
            "",  # 占位，不会被覆盖
            [],
            "",
        )

    # ========== 读取 ==========

    def load_meta(self) -> Optional[dict]:
        """读取故事元信息"""
        meta_path = self.archive_dir / "meta.json"
        if not meta_path.exists():
            return None
        return json.loads(meta_path.read_text(encoding="utf-8"))

    def load_scene(self, scene_number: int) -> Optional[dict]:
        """读取某场场景的全部数据"""
        scene_dir = self.archive_dir / f"scene_{scene_number:03d}"
        if not scene_dir.exists():
            return None

        result = {"scene_number": scene_number}

        for name in ["kernel_raw", "directive", "states_before", "states_after"]:
            path = scene_dir / f"{name}.json"
            if path.exists():
                result[name] = json.loads(path.read_text(encoding="utf-8"))

        for name in ["dialogue", "summary"]:
            path = scene_dir / f"{name}.txt"
            if path.exists():
                result[name] = path.read_text(encoding="utf-8")

        return result

    def get_last_scene_states(self) -> Optional[dict]:
        """获取最后一场结束后的角色状态（用于续写）"""
        count = self._count_scenes()
        if count == 0:
            return None
        last = self.load_scene(count)
        if last and "states_after" in last:
            return last["states_after"]
        return None

    def get_all_summaries(self) -> str:
        """获取所有场景摘要（用于续写时快速了解前情）"""
        count = self._count_scenes()
        parts = []
        for i in range(1, count + 1):
            scene = self.load_scene(i)
            if scene and "summary" in scene:
                parts.append(f"【第 {i} 场】{scene['summary']}")
        return "\n".join(parts)

    # ========== 工具 ==========

    def _count_scenes(self) -> int:
        count = 0
        for d in sorted(self.archive_dir.glob("scene_*")):
            if d.is_dir():
                count += 1
        return count

    def print_tree(self):
        """打印存档目录树"""
        meta = self.load_meta()
        if not meta:
            print("（存档为空）")
            return

        print(f"故事：{meta.get('story_premise', '')[:40]}...")
        print(f"共 {meta.get('total_scenes', 0)} 场")
        print()

        for d in sorted(self.archive_dir.glob("scene_*")):
            num = d.name.replace("scene_", "")
            directive_file = d / "directive.json"
            title = ""
            if directive_file.exists():
                data = json.loads(directive_file.read_text(encoding="utf-8"))
                title = data.get("scene_title", "")
            print(f"  {d.name}/  {title}")
            for f in sorted(d.iterdir()):
                size = f.stat().st_size
                print(f"    {f.name}  ({size} 字节)")
