"""
多角色对话脚本 - AutoGen + 多模型兼容
让角色提示词文件中的角色们自动对话
"""

import asyncio
import json
import os
import random
import re
from pathlib import Path

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from openai import RateLimitError

# ---------- 配置 ----------
CONFIG_PATH = Path(__file__).parent / "config.json"
_config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

ROLE_DIR = Path(__file__).parent / _config["role_dir"]

PROVIDERS = _config["providers"]

ACTIVE_PROVIDER = os.environ.get("LLM_PROVIDER", _config["active_provider"]).lower()
if ACTIVE_PROVIDER not in PROVIDERS:
    print(f"警告：未知的 LLM_PROVIDER={ACTIVE_PROVIDER}，回退到 {_config['active_provider']}")
    ACTIVE_PROVIDER = _config["active_provider"]

_provider = PROVIDERS[ACTIVE_PROVIDER]
API_KEY = os.environ.get(_provider["env_key"]) or _provider.get("api_key", "")
BASE_URL = _provider["base_url"]
MODEL_NAME = _provider["model"]
MODEL_FAMILY = _provider["family"]
MAX_TURNS = _config.get("max_turns_simple", 15)
MAX_ACTIVE_ROLES = _config.get("max_active_roles", 4)
RETRY_MAX = _config.get("retry_max", 3)
RETRY_DELAY = _config.get("retry_delay", 5)
# -------------------------


def sanitize_text(text: str) -> str:
    return text.encode("utf-8", errors="replace").decode("utf-8")


def parse_role_files(directory: Path) -> list[dict]:
    roles = []
    for fpath in sorted(directory.glob("*.md")):
        content = sanitize_text(fpath.read_text(encoding="utf-8").strip())
        stem = fpath.stem

        name_match = re.search(r"#\s*角色[：:]\s*(\S+)", content)
        display_name = name_match.group(1) if name_match else stem.split("-")[0]

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
    agents = []
    for role in roles:
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
    return OpenAIChatCompletionClient(
        model=MODEL_NAME,
        api_key=API_KEY,
        base_url=BASE_URL,
        max_retries=8,
        model_info={
            "vision": True,
            "function_calling": True,
            "json_output": True,
            "family": MODEL_FAMILY,
            "structured_output": False,
            "multiple_system_messages": True,
        },
    )


async def run_with_retry(team, task):
    for attempt in range(1, RETRY_MAX + 1):
        try:
            await Console(team.run_stream(task=task), output_stats=True)
            return
        except (RateLimitError, RuntimeError) as e:
            if "429" in str(e) or "RateLimit" in str(e):
                wait = RETRY_DELAY * attempt
                print(f"\n⚠ 触发速率限制，{wait}秒后重试（第{attempt}/{RETRY_MAX}次）...")
                await asyncio.sleep(wait)
            else:
                raise
    print(f"\n✗ 已达最大重试次数({RETRY_MAX})，对话终止。")


async def main():
    if not API_KEY:
        provider_keys = " / ".join(f"{p['env_key']}({name})" for name, p in PROVIDERS.items())
        print(f"错误：请设置环境变量：{provider_keys}")
        print(f"当前提供商：{ACTIVE_PROVIDER}，需要设置 {_provider['env_key']}")
        print(f"可通过 LLM_PROVIDER 环境变量切换提供商：{' / '.join(PROVIDERS.keys())}")
        return

    # 1. 加载角色
    all_roles = parse_role_files(ROLE_DIR)
    if not all_roles:
        print(f"错误：在 {ROLE_DIR} 中未找到任何角色文件")
        return

    print(f"已加载 {len(all_roles)} 个角色：")
    for r in all_roles:
        print(f"  - {r['display_name']}")
    print()

    # 2. 选择登场角色
    if len(all_roles) > MAX_ACTIVE_ROLES:
        active_roles = random.sample(all_roles, MAX_ACTIVE_ROLES)
        active_roles.sort(key=lambda r: all_roles.index(r))
        print(f"为避免免费API限流，随机选取 {MAX_ACTIVE_ROLES} 位角色登场：")
        for r in active_roles:
            print(f"  ★ {r['display_name']}")
        print()
    else:
        active_roles = all_roles

    # 3. 输入场景
    topic = sanitize_text(input("请输入对话场景/话题：").strip())
    if not topic:
        topic = "你们在一个雨夜的酒馆里相遇了，随便聊点什么。"

    print(f"\n场景：{topic}")
    print(f"模型：{MODEL_NAME} ({ACTIVE_PROVIDER})")
    print(f"{'='*50}\n")

    # 4. 创建模型和 Agent
    model_client = create_model_client()
    agents = create_agents(active_roles, model_client)

    # 5. 创建团队（轮询发言）
    team = RoundRobinGroupChat(
        participants=agents,
        max_turns=MAX_TURNS,
    )

    # 6. 构建任务提示
    role_map = "\n".join(f"  - {r['display_name']}" for r in active_roles)
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

    await run_with_retry(team, task)


if __name__ == "__main__":
    asyncio.run(main())
