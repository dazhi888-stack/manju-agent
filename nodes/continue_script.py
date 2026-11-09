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

def continue_script_node(state: ComicState) -> dict[str, Any]:
    """基于历史记忆 + 用户续写指令，生成新一集剧本。"""
    history_context = (
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
    if not history_context:
        raise NodeError("续写剧本", "缺少历史剧情上下文。")
    if not instruction.strip():
        raise NodeError("续写剧本", "缺少续写指令。")
    if episode_no <= 0:
        raise NodeError("续写剧本", "集数无效，无法续写。")

    canon_text = format_character_cards(state.get("canon_profiles") or [])
    _print_stage(
        "续写剧本生成" + ("（校验后重写）" if branch_feedback else ""),
        f"第{episode_no}集｜历史 {len(history_context)} 字"
        f"｜人物卡 {len(state.get('canon_profiles') or [])} 张"
        f"｜分支重写 {branch_revision_count}",
    )

    draft = invoke_structured(
        node_name="续写剧本",
        schema=ContinueScriptDraft,
        system_prompt=CONTINUE_SCRIPT_SYSTEM,
        user_prompt=build_continue_script_prompt(
            history_context=history_context,
            canon_text=canon_text,
            instruction=instruction,
            episode_no=episode_no,
            feedback=branch_feedback,
            previous_script=previous_script if branch_feedback else "",
        ),
        temperature=CONTINUE_SCRIPT_TEMPERATURE,
        use_script_llm=not bool(branch_feedback),
    )

    script = _clean_rendered_script(render_script(draft.script))
    plot_summary = render_episode_summary(
        title=draft.title,
        episode_no=episode_no,
        main_plot=draft.plot_summary,
        key_events=draft.key_events,
    )
    _print_output(script)

    return {
        "title": draft.title,
        "plot_summary": plot_summary,
        "script": script,
        "episode_no": episode_no,
        "feedback": "",
        "branch_feedback": "",
        "branch_validation": {},
        "branch_revision_count": branch_revision_count,
        "trace": [
            f"续写剧本完成：第{episode_no}集《{draft.title}》，"
            f"{len(draft.script.scenes)} 场"
            + (f"（第{branch_revision_count}次分支重写）" if branch_feedback else "")
        ],
    }

