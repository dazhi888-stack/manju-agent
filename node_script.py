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

def script_node(state: ComicState) -> dict[str, Any]:
    """
    节点2：把小说叙事改写成漫剧剧本。

    输入：plot_summary + raw_text + canon_profiles（重写时再加 feedback / 上一版剧本）
    输出：script（带【旁白】【台词】【动作描述】标签的 Markdown）
    """
    plot_summary = state.get("plot_summary") or ""
    raw_text = state.get("raw_text") or ""
    if not plot_summary:
        raise NodeError("节点2 剧本改写", "没有收到节点1的剧情摘要，无法改写。")

    feedback = (state.get("feedback") or "").strip()
    previous_script = state.get("script") or ""
    is_rewrite = bool(feedback)

    # revision_count 表示“已经完成的剧本重写次数”。
    # 首次进入节点2不加；带 feedback 进来才加 1。
    revision_count = int(state.get("revision_count", 0)) + (1 if is_rewrite else 0)

    canon_text = format_character_cards(state.get("canon_profiles") or [])
    _print_header(
        2,
        "剧本改写" + ("（重写）" if is_rewrite else ""),
        f"摘要 {len(plot_summary)} 字｜原文 {min(len(raw_text), MAX_INPUT_CHARS)} 字"
        f"｜人物卡 {len(state.get('canon_profiles') or [])} 张"
        f"｜重写次数 {revision_count}",
    )

    draft = invoke_structured(
        node_name="节点2 剧本改写",
        schema=ScriptDraft,
        system_prompt=SCRIPT_SYSTEM,
        user_prompt=build_script_prompt(
            plot_summary=plot_summary,
            raw_text=raw_text[:MAX_INPUT_CHARS],
            canon_text=canon_text,
            feedback=feedback,
            previous_script=previous_script,
            truncated=bool(state.get("input_truncated")),
        ),
        temperature=SCRIPT_TEMPERATURE,
        use_script_llm=not is_rewrite,
    )

    script = _clean_rendered_script(render_script(draft))
    beat_count = sum(len(scene.beats) for scene in draft.scenes)
    _print_output(script)

    return {
        "script": script,
        "revision_count": revision_count,
        "trace": [
            f"节点2 剧本改写完成：{len(draft.scenes)} 场 / {beat_count} 条"
            + (f"（第 {revision_count} 次重写）" if is_rewrite else "")
        ],
    }

