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
from global_profile import ensure_global_profile, global_profile_cards

def parse_node(state: ComicState) -> dict[str, Any]:
    """
    节点1：提取剧情摘要 + 冷启动全局角色/世界观。

    输入：state["raw_text"]
    输出：title / plot_data / plot_summary / global_profile / canon_profiles

    人设优先读取本地 global_profile.json；Chroma 只作为旧故事/摘要校验的兜底。
    """
    raw_text = (state.get("raw_text") or "").strip()
    if not raw_text:
        raise NodeError("节点1 原文解析", "输入文本为空。")

    truncated = len(raw_text) > MAX_INPUT_CHARS
    text_for_llm = raw_text[:MAX_INPUT_CHARS]

    library = get_character_library()
    try:
        seeded = library.ensure_seeded()
    except Exception as exc:
        raise NodeError(
            "节点1 原文解析",
            f"初始化 Chroma 人物库失败：{type(exc).__name__}: {exc}。"
            "可尝试删除 manju_agent/chroma_db/ 后重新运行。",
        ) from exc

    if seeded:
        print(f"[Chroma] 人物库为空，已自动注入示例人物 {seeded} 条。", flush=True)

    _print_header(
        1,
        "原文解析",
        f"原文 {len(raw_text)} 字"
        + ("（超过上限，将截断）" if truncated else "")
        + f"｜人物库 {library.get_character_count()} 人",
    )

    plot = invoke_structured(
        node_name="节点1 原文解析",
        schema=PlotExtraction,
        system_prompt=PARSE_SYSTEM,
        user_prompt=build_parse_prompt(text_for_llm, truncated=truncated),
        temperature=PARSE_TEMPERATURE,
    )

    names = [character.name for character in plot.characters if character.name.strip()]
    story_id = str(state.get("story_id") or "default")
    profile: dict[str, Any] = {}
    profile_path = ""
    profile_cached = False
    try:
        profile, saved_path, profile_cached = ensure_global_profile(
            story_id=story_id,
            raw_text=raw_text,
            title=plot.title or "",
        )
        profile_path = str(saved_path)
        local_cards = global_profile_cards(profile)
        if local_cards:
            # 人设优先使用本地全局 JSON；Chroma 只保留剧情摘要用于校验。
            matched = local_cards
            local_names = {
                str(card.get("name") or "") for card in local_cards
            } | {
                str(alias)
                for card in local_cards
                for alias in (card.get("aliases") or [])
            }
            missed = [name for name in names if name not in local_names]
            print(
                f"[全局档案] 命中本地 JSON：{profile_path}"
                f"｜角色 {len(matched)} 人"
                + ("（缓存，未调用 LLM）" if profile_cached else "（首次抽取）"),
                flush=True,
            )
        else:
            matched, missed = library.match_characters(names)
            missed = list(dict.fromkeys(missed))
            print(
                f"[Chroma] 命中 {len(matched)} 人"
                + (f"：{'、'.join(str(card.get('name')) for card in matched)}" if matched else "")
                + (f"｜未命中：{'、'.join(missed)}（按原文推断）" if missed else ""),
                flush=True,
            )
    except Exception as exc:
        print(f"[全局档案警告] 抽取失败，回退 Chroma：{exc}", flush=True)
        matched, missed = library.match_characters(names)
        missed = list(dict.fromkeys(missed))

    plot_data = plot.model_dump()
    plot_summary = render_plot_summary(
        plot_data,
        canon_profiles=matched,
        canon_missed=missed,
        truncated=truncated,
    )
    _print_output(plot_summary)

    return {
        "title": plot.title or "未命名章节",
        "plot_data": plot_data,
        "plot_summary": plot_summary,
        "canon_profiles": matched,
        "canon_missed": missed,
        "global_profile": profile,
        "global_profile_path": profile_path,
        "global_profile_cached": profile_cached,
        "input_chars": len(raw_text),
        "input_truncated": truncated,
        "trace": [
            f"节点1 原文解析完成：{len(plot.characters)} 人 / {len(plot.scenes)} 场景，"
            f"人物库命中 {len(matched)} 人"
            + ("，原文已截断" if truncated else "")
        ],
    }

