"""
内核层 - 故事工厂（不可见）

包含6个智能体的规划流水线：
大纲架构师 → 分镜设计师 → 场景塑造师 → 人物细节师 → 世界观架构师 → 总编审

每个步骤通过 LLM 调用实现，前一个步骤的输出作为后一个步骤的输入。
最终产出"场景指令"下发给角色层执行。
"""

import json
import re
from openai import AsyncOpenAI


# 内核层智能体系统提示词
# 重点：提升写作质量，避免小白文风格
SYSTEM_PROMPTS = {
    "outline": """你是一个【故事大纲架构师】。你负责根据故事前提和当前进度，规划下一个场景的核心内容。

写作要求——避免直白的功能性叙事，追求：
- 场景目标应蕴含戏剧张力，而非简单推进情节
- 核心事件应有不可预测性，避免套路
- 拒绝"巧合推动剧情"的设计
- 每个场景应包含至少一层内在冲突（角色内心的矛盾）和一层外在冲突

你需要输出严格的JSON格式：
{
  "scene_goal": "本场景的核心叙事目标，应包含内在张力",
  "location": "场景发生地点",
  "atmosphere": "总体氛围（细腻而非标签化）",
  "key_participants": ["参与角色列表，至少2个"],
  "key_event": "本场景发生的核心事件，应有意外性",
  "narrative_purpose": "本场景在主线中的功能"
}

不要输出解释性文字，只输出JSON。""",

    "storyboard": """你是一个【分镜设计师】。你负责将大纲拆解为具体的场景序列和节奏控制。

避免"起承转合"的八股结构，思考更自然的叙事节奏：
- 开场应通过细节和动作暗示氛围，而非直接说明
- 中段发展应有多层信息释放，而不是单线程推进
- 转折应来自角色选择而非外部事件
- 收尾应留有回味空间，不必刻意扣题

基于大纲架构师的输出，规划：
{
  "opening": "场景如何开始（通过具体细节而非概括性描述）",
  "development": "场景中段的发展方向，应包含信息叠加和反转",
  "twist_or_conflict": "本场景中的转折或冲突，应来自角色内在动机",
  "ending": "场景如何收尾，留有余韵而非说尽",
  "emotional_arc": "情感曲线，避免直线上升或下降",
  "pacing": "节奏控制，应疏密有致"
}

不要输出解释性文字，只输出JSON。""",

    "scene_designer": """你是一个【场景塑造师】。你负责设计场景的环境氛围和感官细节。

写作要求——避免标签化氛围描写（如"压抑"、"恐怖"、"轻松"），改为：
- 用具体的感官细节织造氛围：一束光如何落在某件物品上、空气中的气味如何变化、远处声音的质感
- 环境应反映角色的心理状态，而非独立的背景板
- 克制用词，一个精准的细节胜过三句渲染
- 避免"空气凝固了"、"死一般的寂静"等陈词滥调

参考：像文学作品的场景描写而非剧本说明。

输出严格JSON：
{
  "environment": "环境描述，应通过细节传递质感",
  "lighting": "光线，应服务于情绪而非说明性",
  "sounds": ["环境声音，每个声音应有质地"],
  "smells": ["气味，应唤起记忆或情绪"],
  "time": "时间",
  "weather": "天气",
  "atmosphere_details": "氛围刻画，通过具体细节织造",
  "props": ["场景中重要的道具/物品，应具有叙事功能"]
}

不要输出解释性文字，只输出JSON。""",

    "character_detail": """你是一个【人物细节师】。你负责管理角色的情绪状态和人物弧光。

写作要求——避免脸谱化情绪标签：
- 情绪不应直接写"紧张"/"愤怒"/"害怕"，而应写出矛盾、混合的情绪状态
- 角色应有口是心非的时刻：外在表现和内在感受不一致
- 每个角色的"近期经历"不应只是"发生了什么"，而应包含"这件事对她意味着什么"
- 关系变化应是微妙渐进的，而非突兀转折
- 避免"英雄式发言"或"功能性对话"，让角色各自有不可告人的动机

输出严格JSON：
{
  "state_updates": {
    "角色名": {
      "current_mood": "细微、矛盾的情绪变化，而非单一标签",
      "physical_state": "身体状态",
      "action_tendency": "行动倾向，应包含犹豫和不确定性"
    }
  },
  "decide_entries": {
    "角色名": "登场方式描述，应通过细节和动作展现状态"
  },
  "relationship_changes": {
    "角色A_角色B": "关系变化，应微妙而有层次"
  },
  "recent_experiences": {
    "角色名": "本条场景中该角色的经历简述，关注内心变化而非事件本身"
  }
}

所有不参与本场的角色保持状态不变。
不要输出解释性文字，只输出JSON。""",

    "world_keeper": """你是一个【世界观架构师】。你负责维护故事世界的一致性和逻辑自洽。

检查以下内容是否违反已建立的世界规则：
1. 场景设定是否合理
2. 角色行为是否符合其性格和背景
3. 是否有时间线或空间逻辑矛盾

{
  "consistent": true 或 false,
  "issues": ["如果有问题，列出具体问题"],
  "fix_suggestions": ["如果有问题，给出修正建议"],
  "approved": true 或 false
}

如果完全没问题，issues 为空数组，approved 为 true。
不要输出解释性文字，只输出JSON。""",

    "editor": """你是一个【总编审】。你是内核层的最终决策者，负责统稿和定调。

写作要求——提升导演指令的文学质感：
- stage_direction 应如文学片段般可读，而非干巴巴的"角色A做X，角色B做Y"
- 使用富有质感的具象语言，避免功能化描述
- 指令中应埋设"潜台词"——告诉角色层"角色想说什么"而非"角色该说什么"
- 避免解释性台词提示（如"用愤怒的语气说"），改为提示角色的内心矛盾和真实意图
- 注重留白——不必填满每一刻的对话，让沉默也有重量

{
  "scene_number": "场景编号",
  "scene_title": "场景标题，蕴藏意象而非平铺直叙",
  "stage_direction": "对角色层的场景指令，以文学语言传递氛围、潜台词、角色内心状态",
  "start_with": "指定第一个发言的角色",
  "notes": "表演提示，应关注角色的内在矛盾而非外部动作"
}

不要输出解释性文字，只输出JSON。""",
}


class KernelPipeline:
    """内核规划流水线"""

    def __init__(self, api_key: str, base_url: str, model: str):
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.system_prompts = SYSTEM_PROMPTS

    @staticmethod
    def _sanitize(text: str) -> str:
        return text.encode("utf-8", errors="replace").decode("utf-8")

    async def _call(self, system_prompt: str, user_message: str, temperature: float = 0.7) -> str:
        """单次LLM调用（自动清洗输入）"""
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._sanitize(system_prompt)},
                {"role": "user", "content": self._sanitize(user_message)},
            ],
            temperature=temperature,
        )
        return response.choices[0].message.content or ""

    def _extract_json(self, text: str) -> dict:
        """从LLM输出中健壮地提取JSON"""
        text = text.strip()

        # 先尝试直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 尝试提取 ```json ... ``` 块
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # 尝试提取 {...} 或 [{...}]
        brace_match = re.search(r'(\{[\s\S]*\})', text)
        if brace_match:
            try:
                return json.loads(brace_match.group(1))
            except json.JSONDecodeError:
                pass

        return {"raw_output": text, "parse_error": True}

    async def summarize_scene(self, dialogue_text: str) -> str:
        """对刚完成的场景做摘要，供下一轮内核规划使用"""
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "你是一个故事记录员。用2-3句话概括刚刚发生的场景中最重要的剧情进展和角色变化。"},
                {"role": "user", "content": f"以下是场景对话记录，请做摘要：\n\n{self._sanitize(dialogue_text[-3000:])}"},
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content or "（场景无显著进展）"

    async def plan_scene(
        self,
        scene_number: int,
        story_premise: str,
        previous_summary: str,
        all_character_names: list[str],
        character_states: dict,
    ) -> dict:
        """
        完整的内核规划流水线

        参数：
            scene_number: 当前场景编号（从1开始）
            story_premise: 故事前提/初始设定
            previous_summary: 之前场景的剧情摘要
            all_character_names: 所有角色名列表
            character_states: 所有角色当前状态 dict

        返回：
            {
                "scene_number": int,
                "final_directive": dict,    # 总编审的最终指令
                "character_updates": dict,  # 人物细节师的角色更新
                "raw_outputs": {...}        # 所有步骤的原始输出（调试用）
            }
        """
        raw = {}

        # ---- Step 1: 大纲架构师 ----
        outline_input = f"""故事前提：{story_premise}
当前进度：第 {scene_number} 场场景
之前剧情：{previous_summary}
角色池：{', '.join(all_character_names)}

请规划本场景的核心目标。"""
        raw["outline"] = await self._call(self.system_prompts["outline"], outline_input)
        outline = self._extract_json(raw["outline"])

        # ---- Step 2: 分镜设计师 ----
        storyboard_input = f"""场景目标：{json.dumps(outline, ensure_ascii=False, indent=2)}

请设计本场景的起承转合和节奏。"""
        raw["storyboard"] = await self._call(self.system_prompts["storyboard"], storyboard_input)
        storyboard = self._extract_json(raw["storyboard"])

        # ---- Step 3: 场景塑造师 ----
        scene_input = f"""场景目标：{json.dumps(outline, ensure_ascii=False, indent=2)}
分镜设计：{json.dumps(storyboard, ensure_ascii=False, indent=2)}

请设计场景环境细节。"""
        raw["scene"] = await self._call(self.system_prompts["scene_designer"], scene_input)
        scene_details = self._extract_json(raw["scene"])

        # ---- Step 4: 人物细节师 ----
        char_input = f"""场景规划：
{json.dumps(outline, ensure_ascii=False, indent=2)}

分镜：
{json.dumps(storyboard, ensure_ascii=False, indent=2)}

角色当前状态：
{json.dumps(character_states, ensure_ascii=False, indent=2)}

请更新角色状态并决定登场角色。"""
        raw["character"] = await self._call(self.system_prompts["character_detail"], char_input)
        char_updates = self._extract_json(raw["character"])

        # ---- Step 5: 世界观架构师 ----
        world_input = f"""场景目标：{json.dumps(outline, ensure_ascii=False, indent=2)}
角色变化：{json.dumps(char_updates, ensure_ascii=False, indent=2)}
环境设计：{json.dumps(scene_details, ensure_ascii=False, indent=2)}

请检查一致性问题。"""
        raw["world"] = await self._call(self.system_prompts["world_keeper"], world_input)
        world_check = self._extract_json(raw["world"])

        # ---- Step 6: 总编审 ----
        editor_input = f"""场景 #{scene_number}

=== 大纲 ===
{json.dumps(outline, ensure_ascii=False, indent=2)}

=== 分镜 ===
{json.dumps(storyboard, ensure_ascii=False, indent=2)}

=== 场景 ===
{json.dumps(scene_details, ensure_ascii=False, indent=2)}

=== 角色更新 ===
{json.dumps(char_updates, ensure_ascii=False, indent=2)}

=== 世界观检查 ===
{json.dumps(world_check, ensure_ascii=False, indent=2)}

请生成最终场景指令。"""
        raw["editor"] = await self._call(self.system_prompts["editor"], editor_input)
        final_directive = self._extract_json(raw["editor"])

        return {
            "scene_number": scene_number,
            "final_directive": final_directive,
            "character_updates": char_updates,
            "raw_outputs": raw,
        }
