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
from render import *
from state import *
from vector_store import *
from node_shared import *

from node_context import _update_story_bible

def _find_canon_card(name: str, cards: list[dict[str, Any]]) -> dict[str, Any] | None:
    """按姓名/别名找全局角色卡，用于把说话风格等字段合并进故事角色。"""
    normalized = str(name or "").replace(" ", "").strip()
    for card in cards:
        keys = [card.get("name", "")] + list(card.get("aliases") or [])
        if normalized in {str(key).replace(" ", "").strip() for key in keys}:
            return card
    return None

def persist_story_node(state: ComicState) -> dict[str, Any]:
    """
    把最终结果写入 story_memory。

    - new_script：角色 + 本集摘要/剧本/分镜；
    - continue_story：新一集摘要/剧本/分镜；
    - modify_script：覆盖目标集摘要/剧本/分镜；
    - 校验未通过（如果存在 validation）则不保存，避免污染历史。
    """
    validation = state.get("validation")
    if isinstance(validation, dict) and validation and not validation.get("passed"):
        print("[story_memory] 校验未通过，跳过持久化。", flush=True)
        return {
            "persist_status": "skipped_validation_failed",
            "trace": ["故事记忆：校验未通过，跳过保存"],
        }

    branch_validation = state.get("branch_validation")
    if (
        isinstance(branch_validation, dict)
        and branch_validation
        and not branch_validation.get("passed")
    ):
        print("[story_memory] 分支剧情校验未通过，跳过持久化。", flush=True)
        return {
            "persist_status": "skipped_branch_validation_failed",
            "trace": ["故事记忆：分支剧情校验未通过，跳过保存"],
        }

    story_id = str(state.get("story_id") or "default").strip()
    intent_type = str(state.get("intent_type") or "new_script")
    title = str(state.get("title") or "未命名")
    plot_summary = state.get("plot_summary") or ""
    script = state.get("script") or ""
    storyboard = state.get("storyboard") or ""
    if not plot_summary or not script or not storyboard:
        raise NodeError("故事记忆持久化", "缺少摘要/剧本/分镜，拒绝写入。")

    _print_stage("故事记忆持久化", f"story_id={story_id}｜意图={intent_type}")

    memory = get_story_memory()
    episode_no = int(state.get("episode_no") or 0)

    if intent_type == "new_script":
        latest = memory.get_latest_episode_no(story_id)
        if episode_no <= 0:
            same_title_episode = memory.find_episode_by_title(story_id, title)
            if same_title_episode:
                episode_no = same_title_episode
                print(
                    f"[story_memory] 检测到同名章节《{title}》，"
                    f"覆盖第{episode_no}集，而不是新增一集。",
                    flush=True,
                )
            else:
                episode_no = latest + 1

        characters = (state.get("plot_data") or {}).get("characters") or []
        canon_cards = state.get("canon_profiles") or []
        if state.get("global_profile") and config.GLOBAL_PROFILE_PRIMARY:
            print(
                "[本地档案] 全局角色已保存到 JSON；Chroma 只写剧情摘要，"
                "跳过完整角色卡写入。",
                flush=True,
            )
        else:
            for character in characters:
                card = dict(character)
                canon = _find_canon_card(str(card.get("name", "")), canon_cards)
                if canon:
                    for field in ("speech_style", "background", "taboos", "examples", "aliases"):
                        if not card.get(field) and canon.get(field):
                            card[field] = canon[field]
                memory.upsert_role(story_id, card, episode_no=episode_no)
            if characters:
                print(f"[story_memory] 已写入/更新角色 {len(characters)} 个。", flush=True)

    elif episode_no <= 0:
        episode_no = memory.get_latest_episode_no(story_id)

    if episode_no <= 0:
        raise NodeError("故事记忆持久化", "无法确定集数，拒绝写入。")

    memory.replace_episode(
        story_id=story_id,
        episode_no=episode_no,
        title=title,
        plot_summary=plot_summary,
        script=script,
        # 完整分镜只保存在 outputs 的 md 文件里，向量库只留引用标记，
        # 避免后续任何检索把整篇分镜带进 LLM 上下文。
        storyboard="[完整分镜已保存到 outputs 文件，未写入 LLM 上下文]",
    )
    print(f"【已追加第 {episode_no} 章摘要到向量库】", flush=True)
    print(
        f"[story_memory] 已保存第{episode_no}集：summary / script / storyboard。",
        flush=True,
    )

    bible_chars = 0
    bible_trace = ""
    try:
        bible = _update_story_bible(
            memory,
            story_id=story_id,
            title=title,
            episode_no=episode_no,
            plot_summary=plot_summary,
        )
        bible_chars = len(bible)
        print(
            f"[长期记忆] 已更新故事圣经：{bible_chars} 字"
            f"（覆盖第1~{episode_no}集关键铺垫）。",
            flush=True,
        )
        bible_trace = f"长期记忆更新：{bible_chars} 字"
    except Exception as exc:
        print(f"[长期记忆警告] 更新失败，不影响本章：{exc}", flush=True)
        bible_trace = f"长期记忆更新失败：{exc}"

    trace = [
        f"故事记忆保存：story_id={story_id}，第{episode_no}集"
    ]
    if bible_trace:
        trace.append(bible_trace)

    return {
        "episode_no": episode_no,
        "persist_status": f"已保存第{episode_no}集",
        "story_bible_chars": bible_chars,
        "trace": trace,
    }

