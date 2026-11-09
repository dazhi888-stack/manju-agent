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

def modify_script_node(state: ComicState) -> dict[str, Any]:
    """根据用户指令，对历史剧本做局部修改。"""
    history_script = state.get("history_script") or ""
    history_summary = (
        state.get("compressed_context")
        or state.get("plot_summary")
        or ""
    )
    instruction = state.get("user_instruction") or (state.get("raw_text") or "")
    episode_no = int(state.get("episode_no") or 0)
    branch_feedback = (state.get("branch_feedback") or "").strip()
    previous_script = state.get("script") or ""
    branch_revision_count = int(state.get("branch_revision_count", 0)) + (
        1 if branch_feedback else 0
    )
    if not history_script:
        raise NodeError("局部修改", "缺少历史剧本，无法修改。")
    if not instruction.strip():
        raise NodeError("局部修改", "缺少修改指令。")
    if episode_no <= 0:
        raise NodeError("局部修改", "集数无效，无法修改。")

    canon_text = format_character_cards(state.get("canon_profiles") or [])
    _print_stage(
        "局部修改剧本" + ("（校验后重写）" if branch_feedback else ""),
        f"第{episode_no}集｜历史剧本 {len(history_script)} 字"
        f"｜分支重写 {branch_revision_count}",
    )

    draft = invoke_structured(
        node_name="局部修改",
        schema=ModifyScriptDraft,
        system_prompt=MODIFY_SCRIPT_SYSTEM,
        user_prompt=build_modify_script_prompt(
            history_summary=history_summary,
            canon_text=canon_text,
            history_script=history_script,
            instruction=instruction,
            episode_no=episode_no,
            feedback=branch_feedback,
            previous_script=previous_script if branch_feedback else "",
        ),
        temperature=MODIFY_SCRIPT_TEMPERATURE,
        use_script_llm=not bool(branch_feedback),
    )

    script = _clean_rendered_script(render_script(draft.script))
    plot_summary = render_episode_summary(
        title=draft.title,
        episode_no=episode_no,
        main_plot=draft.plot_summary,
        key_events=[],
        change_summary=draft.change_summary,
    )
    feedback = (
        f"【局部修改要求】{instruction}\n"
        f"【修改说明】{draft.change_summary}\n"
        "请只调整与修改相关的镜头台词/画面，其他镜头尽量保持历史分镜结构。"
    )
    _print_output(script)

    return {
        "title": draft.title,
        "plot_summary": plot_summary,
        "script": script,
        "feedback": feedback,
        "branch_feedback": "",
        "branch_validation": {},
        "branch_revision_count": branch_revision_count,
        "change_summary": draft.change_summary,
        "episode_no": episode_no,
        "trace": [
            f"局部修改完成：第{episode_no}集《{draft.title}》：{draft.change_summary}"
            + (f"（第{branch_revision_count}次分支重写）" if branch_feedback else "")
        ],
    }

