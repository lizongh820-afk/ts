"""
角色对话系统 - 主入口

双层智能体架构：
  内核层（不可见）→ 规划故事 → 角色层（可见）→ 表演对话
                ↕                    ↕
          状态管理器动态更新角色提示词

每场场景的完整产出自动写入 故事存档/ 目录，支持继承续写。

使用方式：
  python main.py              # 新故事
  python main.py --resume     # 续写上次的故事
  python main.py --tree       # 查看存档目录树
"""

import asyncio
import json
import os
import sys
from pathlib import Path

from kernel import KernelPipeline
from character_layer import CharacterLayer
from state_manager import StateManager
from story_archive import StoryArchive

# ---------- 配置 ----------
CONFIG_PATH = Path(__file__).parent.parent / "config.json"
_config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

BASE_DIR = Path(__file__).parent
ROLE_DIR = BASE_DIR.parent / _config["role_dir"]
STATE_DIR = BASE_DIR / _config["state_dir"]
ARCHIVE_DIR = BASE_DIR / _config["archive_dir"]

ACTIVE_PROVIDER = os.environ.get("LLM_PROVIDER", _config["active_provider"]).lower()
_provider = _config["providers"].get(ACTIVE_PROVIDER, _config["providers"][_config["active_provider"]])
API_KEY = os.environ.get(_provider["env_key"]) or _provider.get("api_key", "")
BASE_URL = _provider["base_url"]
MODEL_NAME = _provider["model"]
MODEL_FAMILY = _provider["family"]

MAX_SCENES = _config["max_scenes"]
MAX_TURNS = _config["max_turns"]
# -------------------------


def print_header(text: str, char: str = "="):
    width = 60
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}\n")


def print_kernel_progress(step: str):
    print(f"  > [{step}]", end="", flush=True)


async def run_new_story(archive: StoryArchive):
    """全新故事运行流程"""
    # ---- 初始化 ----
    state_manager = StateManager(STATE_DIR)
    state_manager.reset_all()

    kernel = KernelPipeline(api_key=API_KEY, base_url=BASE_URL, model=MODEL_NAME)
    character_layer = CharacterLayer(
        role_dir=ROLE_DIR,
        api_key=API_KEY,
        base_url=BASE_URL,
        model=MODEL_NAME,
        model_family=MODEL_FAMILY,
        state_manager=state_manager,
    )
    all_character_names = character_layer.get_all_display_names()

    print_header("角色对话系统 v2 - 双层智能体架构")
    print(f"内核层：大纲架构师 -> 分镜设计师 -> 场景塑造师 -> 人物细节师 -> 世界观架构师 -> 总编审")
    print(f"角色层：{', '.join(all_character_names)}")
    print(f"状态管理：内核层动态更新角色提示词 [OK]")
    print(f"存档位置：{ARCHIVE_DIR}")

    print()
    story_premise = input("请输入故事前提：").strip()
    if not story_premise:
        story_premise = "在一个雨夜的酒馆里，形形色色的人因为一场意外被困在一起，各自的秘密逐渐浮出水面。"

    print(f"\n故事前提：{story_premise}")
    print(f"{'=' * 60}\n")

    # 保存元信息
    archive.save_meta(story_premise, all_character_names, MODEL_NAME)

    await run_scene_loop(
        archive=archive,
        kernel=kernel,
        character_layer=character_layer,
        state_manager=state_manager,
        all_character_names=all_character_names,
        story_premise=story_premise,
        previous_summary="故事尚未开始。",
        start_scene=1,
    )


async def resume_story(archive: StoryArchive):
    """续写已有故事"""
    meta = archive.load_meta()
    if not meta:
        print("错误：没有找到可续写的故事存档")
        return

    state_manager = StateManager(STATE_DIR)
    # 加载最后一场结束后的状态
    last_states = archive.get_last_scene_states()
    if last_states:
        for name, state in last_states.items():
            state_manager.save_state(name, state)

    kernel = KernelPipeline(api_key=API_KEY, base_url=BASE_URL, model=MODEL_NAME)
    character_layer = CharacterLayer(
        role_dir=ROLE_DIR,
        api_key=API_KEY,
        base_url=BASE_URL,
        model=MODEL_NAME,
        model_family=MODEL_FAMILY,
        state_manager=state_manager,
    )
    all_character_names = character_layer.get_all_display_names()

    story_premise = meta.get("story_premise", "")
    total_scenes = meta.get("total_scenes", 0)
    summaries = archive.get_all_summaries()

    print_header("续写故事")
    print(f"故事前提：{story_premise}")
    print(f"已完成 {total_scenes} 场")
    print(f"前情提要：")
    for line in summaries.split("\n"):
        print(f"  {line}")
    print(f"\n{'=' * 60}\n")

    await run_scene_loop(
        archive=archive,
        kernel=kernel,
        character_layer=character_layer,
        state_manager=state_manager,
        all_character_names=all_character_names,
        story_premise=story_premise,
        previous_summary=summaries,
        start_scene=total_scenes + 1,
    )


async def run_scene_loop(
    archive: StoryArchive,
    kernel: KernelPipeline,
    character_layer: CharacterLayer,
    state_manager: StateManager,
    all_character_names: list[str],
    story_premise: str,
    previous_summary: str,
    start_scene: int,
):
    """场景循环（新故事和续写共用）"""
    scene_number = start_scene

    while scene_number <= MAX_SCENES:
        print_header(f"第 {scene_number} 场", char="─")

        # ---- 记录场景开始前的状态 ----
        states_before = state_manager.get_all_states()

        # ======== 阶段一：内核层规划 ========
        print("\n  【内核层】故事工厂运作中...")
        print_kernel_progress("大纲架构")

        kernel_result = await kernel.plan_scene(
            scene_number=scene_number,
            story_premise=story_premise,
            previous_summary=previous_summary,
            all_character_names=all_character_names,
            character_states=states_before,
        )
        print(" [done]")

        char_updates = kernel_result.get("character_updates", {})
        state_updates = char_updates.get("state_updates", {})
        decide_entries = char_updates.get("decide_entries", {})
        recent_experiences = char_updates.get("recent_experiences", {})
        relationship_changes = char_updates.get("relationship_changes", {})

        final_directive = kernel_result.get("final_directive", {})
        scene_title = final_directive.get("scene_title", f"第 {scene_number} 场")
        print(f"  【总编审】场景指令已签发：{scene_title}")

        # ======== 阶段二：更新角色状态 ========
        print_kernel_progress("人物细节注入")

        precise_updates: dict[str, dict] = {}
        for name, mood_update in state_updates.items():
            entry = dict(mood_update)
            if name in recent_experiences:
                current = state_manager.load_state(name)
                existing = current.get("recent_experiences", [])
                existing.append(recent_experiences[name])
                entry["recent_experiences"] = existing[-10:]
            precise_updates[name] = entry

        for pair, change in relationship_changes.items():
            for c in all_character_names:
                if c in pair:
                    other = pair.replace(c, "").strip("_")
                    if c not in precise_updates:
                        precise_updates[c] = {}
                    current = state_manager.load_state(c)
                    existing_rels = dict(current.get("relationship_changes", {}))
                    existing_rels[other] = change
                    precise_updates[c]["relationship_changes"] = existing_rels

        state_manager.apply_updates(precise_updates)
        print(" [done]")

        # ======== 阶段三：角色表演 ========
        print(f"\n  * 场景：{scene_title}")
        entries_str = ", ".join(decide_entries.keys()) if decide_entries else "全体"
        print(f"  * 登场角色：{entries_str}")
        print(f"{'-' * 60}\n")

        directive_for_chars = {**final_directive, "decide_entries": decide_entries}
        dialogue = await character_layer.perform_scene(
            scene_directive=directive_for_chars,
            max_turns=MAX_TURNS,
        )

        print(f"\n{'-' * 60}")
        print(dialogue)
        print(f"{'-' * 60}")

        # ======== 阶段四：场景摘要 ========
        print(f"\n  【内核层】场景归档中...", end="", flush=True)
        summary = await kernel.summarize_scene(dialogue)
        print(" [done]")
        print(f"  剧情进度：{summary}")

        # ======== 阶段五：写入存档 ========
        states_after = state_manager.get_all_states()
        archive.save_scene(
            scene_number=scene_number,
            kernel_raw=kernel_result.get("raw_outputs", {}),
            final_directive=final_directive,
            dialogue_text=dialogue,
            summary=summary,
            states_before=states_before,
            states_after=states_after,
        )
        print(f"  【存档】scene_{scene_number:03d}/ 已写入")

        # ---- 更新摘要供下一轮使用 ----
        previous_summary = summary
        scene_number += 1

        # ---- 询问继续 ----
        if scene_number <= MAX_SCENES:
            print()
            cont = input(f"\n是否进入第 {scene_number} 场？(Enter=继续, n=结束): ").strip().lower()
            if cont == "n":
                print("\n故事结束。")
                break

    print(f"\n{'=' * 60}")
    print(f"  故事完结！共完成 {scene_number - 1} 场场景。")
    print(f"  存档位置：{ARCHIVE_DIR}")
    print(f"  续写命令：python main.py --resume")
    print(f"{'=' * 60}")


async def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if not API_KEY:
        print(f"错误：请设置环境变量 {_provider['env_key']}（当前提供商：{ACTIVE_PROVIDER}）")
        print(f"可通过 LLM_PROVIDER 环境变量切换提供商：{', '.join(_config['providers'].keys())}")
        sys.exit(1)

    if not ROLE_DIR.exists() or not list(ROLE_DIR.glob("*.md")):
        print(f"错误：在 {ROLE_DIR} 中未找到角色提示词文件")
        sys.exit(1)

    archive = StoryArchive(ARCHIVE_DIR)

    # 解析命令行参数
    args = [a for a in sys.argv[1:] if not a.startswith("-")]

    if "--tree" in sys.argv:
        archive.print_tree()
        return
    elif "--resume" in sys.argv:
        await resume_story(archive)
    else:
        await run_new_story(archive)


if __name__ == "__main__":
    asyncio.run(main())
