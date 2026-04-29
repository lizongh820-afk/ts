"""
多角色对话脚本 - AutoGen + DeepSeek
让角色提示词文件中的角色们自动对话
"""

import asyncio
import os
import re
from pathlib import Path

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

# ---------- 配置 ----------
ROLE_DIR = Path(__file__).parent / "角色提示词"

# DeepSeek API (复用已有的 key)
API_KEY = os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("DEEPSEEK_API_KEY")
BASE_URL = "https://api.deepseek.com/v1"
MODEL_NAME = "deepseek-v4-flash"  # 如果不对可改为 deepseek-chat
MAX_TURNS = 15  # 总对话轮次
# -------------------------


def sanitize_text(text: str) -> str:
    """移除非法 Unicode 代理字符（surrogate pairs），防止 API 序列化报错"""
    return text.encode("utf-8", errors="replace").decode("utf-8")


def parse_role_files(directory: Path) -> list[dict]:
    """读取角色提示词目录，返回 [{id, display_name, system_prompt}]"""
    roles = []
    for fpath in sorted(directory.glob("*.md")):
        content = sanitize_text(fpath.read_text(encoding="utf-8").strip())
        stem = fpath.stem

        # 提取 # 角色名
        name_match = re.search(r"#\s*角色[：:]\s*(\S+)", content)
        display_name = name_match.group(1) if name_match else stem.split("-")[0]

        # Agent name 必须为合法 Python 标识符
        agent_id = f"role_{len(roles)}"

        roles.append({
            "id": agent_id,
            "display_name": display_name,
            "system_prompt": content,
        })
    return roles


def create_agents(
    roles: list[dict], model_client: OpenAIChatCompletionClient
) -> list[AssistantAgent]:
    """为每个角色创建 AssistantAgent"""
    agents = []
    for role in roles:
        # 在 system prompt 中注入名字，让模型知道自己的角色名
        system_msg = f"你的名字是{role['display_name']}。\n\n{role['system_prompt']}"
        agent = AssistantAgent(
            name=role["id"],
            model_client=model_client,
            system_message=system_msg,
            model_client_stream=True,
        )
        agents.append(agent)
    return agents


def create_model_client() -> OpenAIChatCompletionClient:
    """创建 DeepSeek 模型客户端"""
    return OpenAIChatCompletionClient(
        model=MODEL_NAME,
        api_key=API_KEY,
        base_url=BASE_URL,
        model_info={
            "vision": True,
            "function_calling": True,
            "json_output": True,
            "family": "deepseek",
            "structured_output": False,
            "multiple_system_messages": True,
        },
    )



async def main():
    if not API_KEY:
        print("错误：请设置 ANTHROPIC_AUTH_TOKEN 或 DEEPSEEK_API_KEY 环境变量")
        return

    # 1. 加载角色
    roles = parse_role_files(ROLE_DIR)
    if not roles:
        print(f"错误：在 {ROLE_DIR} 中未找到任何角色文件")
        return

    print(f"已加载 {len(roles)} 个角色：")
    for r in roles:
        print(f"  - {r['display_name']}")
    print()

    # 2. 输入场景
    topic = sanitize_text(input("请输入对话场景/话题：").strip())
    if not topic:
        topic = "你们在一个雨夜的酒馆里相遇了，随便聊点什么。"

    print(f"\n场景：{topic}")
    print(f"{'='*50}\n")

    # 3. 创建模型和 Agent
    model_client = create_model_client()
    agents = create_agents(roles, model_client)

    # 4. 创建团队（轮询发言）
    team = RoundRobinGroupChat(
        participants=agents,
        max_turns=MAX_TURNS,
    )

    # 5. 构建任务提示（把角色映射传给对话上下文）
    role_map = "\n".join(f"  - {r['display_name']}（Agent ID: {r['id']}）" for r in roles)
    task = f"""请按照以下场景进行角色扮演对话：

参与角色：
{role_map}

场景设定：{topic}

规则：
1. 每个角色严格按照自己的性格设定说话
2. 不要替别人发言
3. 保持角色一致性
4. 自然推进对话，像真人聊天一样

开始吧！"""

    await Console(team.run_stream(task=task), output_stats=True)


if __name__ == "__main__":
    asyncio.run(main())
