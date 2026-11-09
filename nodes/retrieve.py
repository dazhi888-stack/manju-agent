from __future__ import annotations

import json
import re
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import interrupt
from pydantic import BaseModel

import config
from config import *
from prompts import *
from services.render import *
from state import *
from stores.vector_store import *
from .shared import *

from .context import _fallback_compressed_context
from stores.global_profile import global_profile_cards, load_global_profile

def _preferred_roles(story_id: str, memory: Any) -> list[dict[str, Any]]:
    """人设优先读本地全局 JSON；旧故事才回退 Chroma 角色卡。"""
    local = global_profile_cards(load_global_profile(story_id))
    return local or memory.get_roles(story_id)

def retrieve_continue_node(state: ComicState) -> dict[str, Any]:
    """检索角色，并复用前置节点压缩好的最近剧情梗概。"""
    story_id = str(state.get("story_id") or "default").strip()
    _print_stage("续写记忆检索", f"story_id={story_id}")

    try:
        memory = get_story_memory()
        roles = _preferred_roles(story_id, memory)
        summaries = memory.get_recent_summaries(
            story_id, limit=MAX_RECALL_EPISODES
        )
        latest = memory.get_latest_episode_no(story_id)
    except Exception as exc:
        return {
            "error_message": (
                f"读取故事记忆失败：{type(exc).__name__}: {exc}\n"
                "请检查 chroma_db/ 是否可访问，或先用新剧本跑一次初始化。"
            ),
            "trace": [f"续写检索失败：{exc}"],
        }

    if not summaries:
        return {
            "error_message": (
                f"没有找到 story_id={story_id} 的历史剧情摘要。\n"
                "请先用完整小说/剧本跑一次 new_script，或检查 --story-id 是否正确。"
            ),
            "trace": [f"续写检索失败：story_id={story_id} 无历史摘要"],
        }

    compressed = str(state.get("compressed_context") or "").strip()
    if (
        not compressed
        or str(state.get("compressed_story_id") or "") != story_id
    ):
        # 前置压缩没有命中同一故事时，只对最近 3 集做确定性兜底。
        compressed = _fallback_compressed_context(story_id, summaries)
        print(
            "[续写检索] 未命中前置压缩结果，使用最近 3 集摘要兜底。",
            flush=True,
        )

    target = state.get("target_episode")
    if target is not None and int(target) > latest:
        episode_no = int(target)
    else:
        episode_no = latest + 1

    if target is not None and int(target) > latest + 1:
        print(
            f"[续写警告] 用户要求第{int(target)}集，但 story_id={story_id} "
            f"只有到第{latest}集；可能缺少中间集摘要。"
            "请用 --sync-outputs 或 --story-id 检查历史。",
            flush=True,
        )

    print(
        f"[story_memory] story_id={story_id}｜角色 {len(roles)} 个"
        f"｜最近摘要 {len(summaries)} 集（上限 {MAX_RECALL_EPISODES}）"
        f"｜最新第{latest}集｜本次生成第{episode_no}集",
        flush=True,
    )
    print(compressed, flush=True)

    return {
        "story_id": story_id,
        "history_roles": roles,
        "history_summaries": summaries,
        "episode_no": episode_no,
        "canon_profiles": roles,
        "plot_summary": compressed,
        "trace": [
            f"续写检索完成：story_id={story_id}，角色 {len(roles)} 个，"
            f"最近摘要 {len(summaries)} 集，准备生成第{episode_no}集"
        ],
    }

def retrieve_storyboard_node(state: ComicState) -> dict[str, Any]:
    """检索目标集的历史剧本 + 分镜 + 摘要 + 角色。"""
    story_id = str(state.get("story_id") or "default").strip()
    _print_stage("历史分镜检索", f"story_id={story_id}")

    try:
        memory = get_story_memory()
        latest = memory.get_latest_episode_no(story_id)
        target = state.get("target_episode")
        target_episode = int(target) if target is not None else latest
        if target_episode <= 0:
            raise ValueError("该故事还没有任何历史集数")
        docs = memory.get_episode_docs(story_id, target_episode)
        roles = _preferred_roles(story_id, memory)
        summaries = memory.get_recent_summaries(
            story_id, limit=MAX_RECALL_EPISODES
        )
    except Exception as exc:
        return {
            "error_message": (
                f"读取第{state.get('target_episode') or '最新'}集历史失败："
                f"{type(exc).__name__}: {exc}\n"
                f"请检查 story_id={story_id} 和 --story-id。"
            ),
            "trace": [f"历史分镜检索失败：{exc}"],
        }

    if "script" not in docs:
        return {
            "error_message": (
                f"story_id={story_id} 的第{target_episode}集缺少历史剧本，无法局部修改。\n"
                "请先用 new_script 生成该集，或换一个存在的集数。"
            ),
            "trace": [f"历史分镜检索失败：第{target_episode}集剧本缺失"],
        }

    summary_doc = docs.get("summary")
    compressed = str(state.get("compressed_context") or "").strip()
    if (
        not compressed
        or str(state.get("compressed_story_id") or "") != story_id
    ):
        compressed = _fallback_compressed_context(story_id, summaries)
    plot_summary = compressed

    title = str(
        (summary_doc or {}).get("metadata", {}).get("title")
        or state.get("title")
        or f"第{target_episode}集"
    )
    print(
        f"[story_memory] 第{target_episode}集｜角色 {len(roles)} 个"
        f"｜剧本 {len(docs['script']['document'])} 字"
        f"｜最近摘要 {len(summaries)} 集（完整分镜不进入 LLM 上下文）",
        flush=True,
    )

    return {
        "story_id": story_id,
        "episode_no": target_episode,
        "title": title,
        "history_roles": roles,
        "history_summaries": summaries,
        "history_script": str(docs["script"]["document"]),
        "canon_profiles": roles,
        "plot_summary": plot_summary,
        "trace": [
            f"历史分镜检索完成：story_id={story_id}，第{target_episode}集，"
            f"角色 {len(roles)} 个，最近摘要 {len(summaries)} 集"
        ],
    }

