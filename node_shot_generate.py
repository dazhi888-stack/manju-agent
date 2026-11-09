"""shot_generate_node：全局角色 + 本集主线 → 带 shot_no 的镜头 JSON。

本地 ep_XX.json 存在时直接缓存命中，不调用 LLM。
"""

from __future__ import annotations

import time
from typing import Any

import config
from global_profile import load_global_profile
from node_shared import NodeError, _print_header, _print_output, invoke_structured
from prompts_shots import SHOT_GENERATE_SYSTEM, build_shot_generate_prompt
from shot_store import (
    load_episode_shots,
    normalize_shots,
    render_storyboard_markdown,
    save_episode_shots,
)
from state import ComicState, EpisodeShotsDraft


def _episode_no(state: ComicState) -> int:
    episode_no = int(state.get("episode_no") or 0)
    if episode_no > 0:
        return episode_no
    from vector_store import get_story_memory

    story_id = str(state.get("story_id") or "default")
    return get_story_memory().get_latest_episode_no(story_id) + 1


def shot_generate_node(state: ComicState) -> dict[str, Any]:
    """
    输入全局角色档案 + 本集主线剧情，生成分镜列表。

    输出 state：
    - shots：结构化镜头列表
    - episode_tone：本集情绪基调
    - storyboard：兼容现有渲染的 Markdown
    - shot_file / shots_cached
    """
    story_id = str(state.get("story_id") or "default")
    episode_no = _episode_no(state)
    title = str(state.get("title") or "未命名")
    plot_summary = str(state.get("plot_summary") or "")
    feedback = str(state.get("shot_feedback") or state.get("branch_feedback") or "")
    started = time.monotonic()

    # 缓存判断：有 JSON 且没有 OOC 修改意见时，完全跳过 LLM。
    if not feedback and not state.get("shot_force"):
        cached = load_episode_shots(story_id, episode_no)
        if cached and cached.get("shots"):
            shots = normalize_shots(cached.get("shots") or [])
            profile = state.get("global_profile") or load_global_profile(story_id) or {}
            profile_names = [
                str(card.get("name") or "")
                for card in (profile.get("characters") or [])
                if str(card.get("name") or "").strip()
            ]
            for shot in shots:
                visual = str(shot.get("visual") or "")
                shot["characters"] = list(dict.fromkeys(
                    [name for name in profile_names if name and name in visual]
                    + list(shot.get("characters") or [])
                ))
            storyboard = render_storyboard_markdown(shots, title)
            print(
                f"[分镜缓存] 命中 {story_id} 第{episode_no}集 ep_{episode_no:02d}.json，"
                "跳过 LLM。",
                flush=True,
            )
            _print_header(3, "分镜生成（缓存）", f"{story_id} 第{episode_no}集｜{len(shots)} 镜头")
            return {
                "shots": shots,
                "storyboard": storyboard,
                "episode_tone": str(cached.get("episode_tone") or ""),
                "shot_file": str(
                    config.OUTPUT_DIR
                    / story_id
                    / "shots"
                    / f"ep_{episode_no:02d}.json"
                ),
                "shots_cached": True,
                "shot_feedback": "",
                "trace": [
                    f"分镜缓存命中：第{episode_no}集 {len(shots)} 镜头（未调用 LLM）"
                ],
            }

    _print_header(
        3,
        "分镜生成",
        f"{story_id} 第{episode_no}集｜主线 {len(plot_summary)} 字"
        + ("｜带 OOC 修改意见" if feedback else ""),
    )
    profile = state.get("global_profile") or load_global_profile(story_id) or {}
    if not profile.get("characters"):
        # 兼容旧故事：没有冷启动 JSON 时，用已召回的角色卡兜底。
        fallback_cards = state.get("canon_profiles") or state.get("history_roles") or []
        if fallback_cards:
            profile = {"characters": fallback_cards}
    if not profile.get("characters"):
        raise NodeError("分镜生成", "缺少全局角色档案，请先执行冷启动角色抽取。")

    draft: EpisodeShotsDraft = invoke_structured(
        node_name="shot_generate_node",
        schema=EpisodeShotsDraft,
        system_prompt=SHOT_GENERATE_SYSTEM,
        user_prompt=build_shot_generate_prompt(
            global_profile=profile,
            plot_summary=plot_summary,
            title=title,
            feedback=feedback,
        ),
        temperature=config.STORYBOARD_TEMPERATURE,
    )

    shots = normalize_shots([shot.model_dump() for shot in draft.shots])
    profile_names = [
        str(card.get("name") or "")
        for card in (profile.get("characters") or [])
        if str(card.get("name") or "").strip()
    ]
    for shot in shots:
        visual = str(shot.get("visual") or "")
        shot["characters"] = list(dict.fromkeys(
            [name for name in profile_names if name and name in visual]
            + list(shot.get("characters") or [])
        ))
    if not shots:
        raise NodeError("分镜生成", "模型没有返回有效镜头。")
    storyboard = render_storyboard_markdown(shots, title)
    path = save_episode_shots(
        story_id=story_id,
        episode_no=episode_no,
        title=title,
        episode_tone=draft.episode_tone,
        shots=shots,
    )
    _print_output(storyboard)
    elapsed = int((time.monotonic() - started) * 1000)
    return {
        "shots": shots,
        "storyboard": storyboard,
        "episode_no": episode_no,
        "episode_tone": draft.episode_tone,
        "shot_file": str(path),
        "shots_cached": False,
        "shot_feedback": "",
        "trace": [
            f"shot_generate_node：第{episode_no}集 {len(shots)} 镜头｜"
            f"情绪={draft.episode_tone}｜耗时 {elapsed}ms｜已存 {path.name}"
        ],
    }
