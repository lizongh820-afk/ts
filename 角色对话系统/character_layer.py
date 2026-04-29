"""
角色层 - 故事演员（可见）

加载角色提示词，结合状态管理器的动态状态，
创建 AutoGen 对话组执行场景表演。
"""

import asyncio
import re
from pathlib import Path

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.base import TaskResult
from autogen_ext.models.openai import OpenAIChatCompletionClient

from state_manager import StateManager


def sanitize_text(text: str) -> str:
    """移除非法 Unicode 代理字符"""
    return text.encode("utf-8", errors="replace").decode("utf-8")


def parse_role_files(role_dir: Path) -> list[dict]:
    """读取角色提示词目录，返回 [{id, display_name, system_prompt}]"""
    roles = []
    for fpath in sorted(role_dir.glob("*.md")):
        content = sanitize_text(fpath.read_text(encoding="utf-8").strip())
        # 提取 # 角色名
        name_match = re.search(r"#\s*角色[：:]\s*(\S+)", content)
        display_name = name_match.group(1) if name_match else fpath.stem.split("-")[0]
        roles.append({
            "id": f"char_{len(roles)}",
            "display_name": display_name,
            "system_prompt": content,
        })
    return roles


def create_model_client(api_key: str, base_url: str, model: str, model_family: str = "deepseek") -> OpenAIChatCompletionClient:
    return OpenAIChatCompletionClient(
        model=model,
        api_key=api_key,
        base_url=base_url,
        max_retries=8,
        model_info={
            "vision": True,
            "function_calling": True,
            "json_output": True,
            "family": model_family,
            "structured_output": False,
            "multiple_system_messages": True,
        },
    )


class CharacterLayer:
    """角色层 - 管理角色Agent并执行场景对话"""

    def __init__(
        self,
        role_dir: str | Path,
        api_key: str,
        base_url: str,
        model: str,
        state_manager: StateManager,
        model_family: str = "deepseek",
    ):
        self.role_dir = Path(role_dir)
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.model_family = model_family
        self.state_manager = state_manager
        self.roles = parse_role_files(self.role_dir)
        self.name_to_id = {r["display_name"]: r["id"] for r in self.roles}

    def get_all_display_names(self) -> list[str]:
        return [r["display_name"] for r in self.roles]

    async def perform_scene(self, scene_directive: dict, max_turns: int = 12) -> str:
        """
        执行一场场景对话

        参数：
            scene_directive: 内核层下发的场景指令 dict
            max_turns: 最大对话轮次

        返回：
            dialogue_text: 完整对话文本
        """
        # 提取指令信息
        directive_text = scene_directive.get("stage_direction", "")
        start_with = scene_directive.get("start_with", "")
        decide_entries = scene_directive.get("decide_entries", {})

        model_client = create_model_client(self.api_key, self.base_url, self.model, self.model_family)

        # 选择登场角色
        active_names = self._decide_active_roles(decide_entries)
        if not active_names:
            active_names = self.get_all_display_names()[:4]

        # 创建角色Agent（指定首发角色排第一个）
        agent_list = []
        for role in self.roles:
            if role["display_name"] not in active_names:
                continue

            effective_prompt = self.state_manager.get_effective_prompt(
                role["system_prompt"], role["display_name"]
            )
            agent = AssistantAgent(
                name=role["id"],
                model_client=model_client,
                system_message=effective_prompt,
                description=role["display_name"],
                model_client_stream=True,
            )
            agent_list.append(agent)

        # 把 start_with 指定的角色排到最前面
        if start_with:
            for i, agent in enumerate(agent_list):
                for r in self.roles:
                    if r["id"] == agent.name and r["display_name"] == start_with:
                        agent_list.insert(0, agent_list.pop(i))
                        break

        if not agent_list:
            return "（没有角色登场）"

        # 构造表演任务
        entries_desc = "\n".join(
            f"  - {name}：{desc}"
            for name, desc in decide_entries.items()
            if name in active_names
        )

        task = f"""【场景指令】
{directive_text}

【登场角色及状态】
{entries_desc}

【写作要求——请严格遵守】
1. Show, don't tell：通过动作、细节、停顿传递情绪，而非直接说出"我很紧张"、"他生气了"
2. 潜台词优先：对话应有弦外之音，角色说的不等于他们真正想的
3. 克制用词：一个精准的细节胜过三句渲染。避免陈词滥调（"空气凝固了"、"死一般的寂静"）
4. 节奏变化：长短句交替，对话应有留白和沉默的空间
5. 角色语言差异化：每个角色的措辞、节奏、用词偏好应不同
6. 避免功能化对话：角色不应为了向读者"解释剧情"而说话
7. 不以"突然"、"忽然"等词制造廉价转折，让转折从情境中自然生长

【开始表演】
"""

        # 运行对话
        team = RoundRobinGroupChat(
            participants=agent_list,
            max_turns=max_turns,
        )

        dialogue_parts = []
        async for msg in team.run_stream(task=task):
            if isinstance(msg, TaskResult):
                continue
            # 只捕获最终文本消息，跳过流式块
            if not hasattr(msg, "content") or not msg.content:
                continue
            if hasattr(msg, "type") and msg.type == "ModelClientStreamingChunkEvent":
                continue

            source_id = getattr(msg, "source", "")
            display_name = source_id
            for r in self.roles:
                if r["id"] == source_id:
                    display_name = r["display_name"]
                    break

            dialogue_parts.append(f"【{display_name}】\n{msg.content}\n")

        return "\n".join(dialogue_parts)

    def _decide_active_roles(self, decide_entries: dict) -> list[str]:
        """根据内核层的人物安排，决定哪些角色在本场登场"""
        if not decide_entries:
            return []

        active = []
        for name, desc in decide_entries.items():
            if "登场" in desc or "出现" in desc or "进入" in desc:
                active.append(name)
            elif "在场" in desc or "背景" in desc:
                pass  # 背景角色不创建Agent
            else:
                active.append(name)  # 默认为登场

        return active
