"""build_video_prompt_node：纯 Python 按镜头区间拼接提示词，不调用 LLM。"""

from __future__ import annotations

from typing import Any

import config
from services.character_reference import reference_block as character_reference_block
from stores.global_profile import load_global_profile
from services.shot_prompt import parse_storyboard_shots
from stores.shot_store import render_storyboard_markdown
from state import ComicState
from services.video_prompt import build_episode_video_prompt


def build_video_prompt_node(state: ComicState) -> dict[str, Any]:
    """读取 state.shots / storyboard，按 start_shot~end_shot 切片。"""
    story_id = str(state.get("story_id") or "default")
    episode_no = int(state.get("episode_no") or 0)
    shots = state.get("shots") or []
    if not shots:
        shots = parse_storyboard_shots(state.get("storyboard") or "")
    total = len(shots)
    if total <= 0:
        return {
            "video_prompt": "",
            "video_prompt_start": 0,
            "video_prompt_end": 0,
            "trace": ["视频提示词：没有镜头数据，跳过"],
        }

    start = max(1, int(state.get("start_shot") or 1))
    end = int(state.get("end_shot") or 3)
    end = min(max(start, end), total)

    storyboard = str(state.get("storyboard") or "")
    if not storyboard:
        storyboard = render_storyboard_markdown(shots, str(state.get("title") or ""))
    prompt = build_episode_video_prompt(
        summary=str(state.get("plot_summary") or ""),
        storyboard=storyboard,
        style_desc=str(state.get("style_desc") or config.DEFAULT_STYLE_DESC),
        start_shot=start,
        end_shot=end,
        global_profile=state.get("global_profile") or load_global_profile(story_id),
        reference_block=character_reference_block(story_id),
    )
    prompt_dir = config.OUTPUT_DIR / story_id / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    path = prompt_dir / f"第{episode_no}章_视频_镜头{start}-{end}.txt"
    path.write_text(prompt, encoding="utf-8")
    return {
        "video_prompt": prompt,
        "video_prompt_start": start,
        "video_prompt_end": end,
        "trace": [
            f"视频提示词：第{episode_no}集镜头 {start}-{end}/{total}｜纯 Python 拼接"
        ],
    }
