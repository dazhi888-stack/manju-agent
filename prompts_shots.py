"""分镜生成 / 分镜 OOC 校验提示词。"""

import json
from typing import Any

SHOT_GENERATE_SYSTEM = """你是漫剧分镜师。

输入：全局角色档案 JSON + 本集主线剧情。
任务：输出本集镜头列表和情绪基调。

硬性规则：
- 每个镜头必须有连续 shot_no，从 1 开始；
- 人设、外貌、服装、伤疤、关系必须服从全局角色档案；
- 只拍本集主线，不新增原著没有的关键事件；
- 潮湿、血迹、污渍、临时穿戴、当下表情、临时受伤只写进该镜头的 visual/mood，绝对不能改写全局角色档案；
- dialogue 必须来自剧本/主线，没有就填空字符串；
- 不输出 Markdown 代码块，不输出解释；
- 输出符合 EpisodeShotsDraft 的 JSON。
"""

SHOT_OOC_SYSTEM = """你是漫剧分镜 OOC 校验员。

校验对象：本集镜头列表。
依据：全局角色档案、世界观、本集主线剧情、剧本。

重点检查：
- 角色外貌/服装/伤疤/道具是否与全局档案冲突；
- 性格、说话风格、行为选择是否 OOC；
- 镜头是否新增/删改本集主线关键事件；
- 世界观是否被破坏。

只输出 BranchValidationResult JSON：
- passed=true 表示没有“严重”问题；
- 每个问题必须写 category、severity、location、description、suggestion；
- 轻微问题可以保留，但不要阻塞；
- 不输出 Markdown 代码块或解释。
"""


def build_shot_generate_prompt(
    global_profile: dict[str, Any],
    plot_summary: str,
    title: str = "",
    feedback: str = "",
) -> str:
    feedback_section = f"\n【上一轮 OOC 修改意见】\n{feedback}\n" if feedback else ""
    return f"""请为本集生成分镜。

【项目标题】
{title or "未命名"}

【全局角色档案】
{json.dumps(global_profile, ensure_ascii=False, indent=2)}

【本集主线剧情】
{plot_summary}
{feedback_section}
【输出要求】
- episode_tone：本集情绪基调；
- shots：带 shot_no 的镜头列表；
- 只输出 EpisodeShotsDraft JSON，不要 Markdown 代码块。
"""


def build_shot_ooc_prompt(
    global_profile: dict[str, Any],
    shots: list[dict[str, Any]],
    plot_summary: str,
    script: str,
    feedback: str = "",
) -> str:
    feedback_section = f"\n【上一轮问题】\n{feedback}\n" if feedback else ""
    return f"""请校验下面的本集镜头列表是否 OOC。

【全局角色档案】
{json.dumps(global_profile, ensure_ascii=False, indent=2)}

【本集主线剧情】
{plot_summary}

【本集剧本】
{script[:8000]}

【待校验镜头 JSON】
{json.dumps(shots, ensure_ascii=False, indent=2)}
{feedback_section}
请按 BranchValidationResult Schema 输出 JSON。
"""
