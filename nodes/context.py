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

def _format_summaries_for_compress(summaries: list[dict[str, Any]]) -> str:
    """把最近摘要拼成压缩提示词输入，最多 3 条。"""
    parts: list[str] = []
    for doc in summaries:
        metadata = doc.get("metadata") or {}
        episode_no = metadata.get("episode_no", "?")
        title = metadata.get("title", "")
        parts.append(f"### 第{episode_no}集｜{title}\n{doc.get('document', '')}")
    return "\n\n".join(parts)

def _render_compressed_context(result: ContextCompression) -> str:
    lines = [
        "# 前情压缩梗概",
        "",
        result.synopsis.strip(),
        "",
        "## 关键事实",
    ]
    if result.key_facts:
        lines.extend(f"- {item}" for item in result.key_facts)
    else:
        lines.append("- 无")
    lines.append("")
    lines.append("## 未解决悬念")
    if result.open_threads:
        lines.extend(f"- {item}" for item in result.open_threads)
    else:
        lines.append("- 无")
    return "\n".join(lines).strip()

def _fallback_compressed_context(
    story_id: str,
    summaries: list[dict[str, Any]],
    story_bible: str = "",
) -> str:
    """压缩失败时的确定性兜底：长期记忆 + 最近 3 条摘要，限制总长度。"""
    parts = [f"# 前情压缩梗概（兜底）\n\nstory_id={story_id}"]
    if story_bible:
        parts.append(f"## 长期故事圣经\n{story_bible}")
    for doc in summaries:
        metadata = doc.get("metadata") or {}
        episode_no = metadata.get("episode_no", "?")
        title = metadata.get("title", "")
        text = str(doc.get("document", "")).strip()
        parts.append(f"## 第{episode_no}集｜{title}\n{text[:600]}")
    result = "\n\n".join(parts)
    if len(result) > 3_000:
        result = result[:3_000] + "\n……（兜底上下文已截断）"
    return result

def _render_story_bible(result: StoryBible) -> str:
    lines = [
        "# 长期故事圣经",
        "",
        "## 核心主线",
        result.synopsis.strip(),
        "",
        "## 人物状态",
    ]
    lines.extend(f"- {item}" for item in (result.character_states or ["无"]))
    lines.append("")
    lines.append("## 关键物件/设定")
    lines.extend(f"- {item}" for item in (result.key_objects or ["无"]))
    lines.append("")
    lines.append("## 未解决线索")
    lines.extend(f"- {item}" for item in (result.plot_threads or ["无"]))
    lines.append("")
    lines.append("## 时间线")
    lines.extend(f"- {item}" for item in (result.timeline or ["无"]))
    return "\n".join(lines).strip()

def _format_all_summaries_for_bible(
    memory: Any,
    story_id: str,
) -> str:
    parts: list[str] = []
    for doc in memory.get_summaries(story_id):
        metadata = doc.get("metadata") or {}
        parts.append(
            f"### 第{metadata.get('episode_no', '?')}集｜"
            f"{metadata.get('title', '')}\n{doc.get('document', '')}"
        )
    return "\n\n".join(parts)

def _update_story_bible(
    memory: Any,
    story_id: str,
    title: str,
    episode_no: int,
    plot_summary: str,
) -> str:
    """
    更新跨集长期记忆。

    没有旧 bible 时从全部已有摘要重建；有旧 bible 时做增量合并。
    失败时用确定性文本兜底，不影响本章已经完成的持久化。
    """
    previous = memory.get_story_bible(story_id)
    summaries_text = _format_all_summaries_for_bible(memory, story_id)
    try:
        result = invoke_structured(
            node_name="长期记忆更新",
            schema=StoryBible,
            system_prompt=STORY_BIBLE_SYSTEM,
            user_prompt=build_story_bible_prompt(
                previous_bible=previous,
                summaries_text=summaries_text,
                new_summary=plot_summary,
                title=title,
                episode_no=episode_no,
            ),
            temperature=0.1,
        )
        bible = _render_story_bible(result)
    except Exception as exc:
        print(f"[长期记忆警告] 结构化更新失败，使用兜底：{exc}", flush=True)
        chunks = [previous] if previous else []
        chunks.append(f"## 第{episode_no}集｜{title}\n{plot_summary}")
        bible = "# 长期故事圣经（兜底）\n\n" + "\n\n".join(chunks)
        if len(bible) > 4_000:
            bible = bible[:4_000] + "\n……（长期记忆已截断）"

    memory.upsert_story_bible(story_id, bible, updated_episode=episode_no)
    return bible

def context_compress_node(state: ComicState) -> dict[str, Any]:
    """
    续写前的前置压缩节点。

    - 双层记忆：长期故事圣经 + 最近 3 集摘要；
    - 长期记忆不占最近 3 集名额，用来保留早期铺垫；
    - 让 LLM 合并成短梗概，后续节点只使用 compressed_context；
    - 完整分镜/剧本不进入这个节点的上下文；
    - 压缩失败时退化为长期记忆 + 最近 3 集摘要拼接，不让主流程崩掉。
    """
    text = (state.get("raw_text") or "").strip()
    if len(text) > 1_200:
        # 长文本大概率是 new_script，不做续写上下文压缩，避免额外调用。
        return {
            "trace": ["上下文压缩：输入较长，按 new_script 候选跳过"],
        }

    explicit_story_id = str(state.get("story_id") or "").strip()
    story_id = explicit_story_id
    if not story_id:
        try:
            known = get_story_memory().list_story_ids()
        except Exception as exc:
            print(f"[上下文压缩警告] 读取故事列表失败：{exc}", flush=True)
            known = []
        if len(known) == 1:
            story_id = known[0]

    if not story_id:
        return {"trace": ["上下文压缩：未确定 story_id，跳过"]}

    try:
        memory = get_story_memory()
        summaries = memory.get_recent_summaries(story_id, limit=MAX_RECALL_EPISODES)
        story_bible = memory.get_story_bible(story_id)
    except Exception as exc:
        print(f"[上下文压缩警告] 读取摘要失败：{exc}", flush=True)
        return {"trace": [f"上下文压缩失败：{exc}"]}

    if not summaries and not story_bible:
        return {
            "trace": [
                f"上下文压缩：story_id={story_id} 暂无历史摘要，跳过"
            ]
        }

    _print_stage(
        "前置剧情压缩",
        f"story_id={story_id}｜长期记忆 {len(story_bible)} 字"
        f"｜召回最近 {len(summaries)} 集（上限 {MAX_RECALL_EPISODES}）",
    )

    summaries_text = _format_summaries_for_compress(summaries)
    if not summaries_text:
        summaries_text = "（最近 3 集无独立摘要，使用长期故事圣经。）"
    try:
        result = invoke_structured(
            node_name="前置剧情压缩",
            schema=ContextCompression,
            system_prompt=CONTEXT_COMPRESS_SYSTEM,
            user_prompt=build_context_compress_prompt(
                summaries_text,
                story_id,
                story_bible=story_bible,
            ),
            temperature=0.1,
        )
        compressed = _render_compressed_context(result)
        trace = [
            f"上下文压缩：story_id={story_id}，长期记忆 {len(story_bible)} 字，"
            f"最近 {len(summaries)} 集，压缩为 {len(compressed)} 字"
        ]
    except Exception as exc:
        compressed = _fallback_compressed_context(story_id, summaries, story_bible)
        trace = [
            f"上下文压缩失败，使用长期记忆 + 最近 {len(summaries)} 集兜底：{exc}"
        ]
        print(f"[上下文压缩警告] {type(exc).__name__}: {exc}", flush=True)

    print(compressed, flush=True)
    return {
        "compressed_context": compressed,
        "compressed_story_id": story_id,
        "compressed_source_count": len(summaries),
        "long_term_context": story_bible,
        "story_bible_chars": len(story_bible),
        "history_summaries": summaries,
        "trace": trace,
    }

