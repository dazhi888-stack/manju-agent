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

def storyboard_node(state: ComicState) -> dict[str, Any]:
    """
    节点3：把剧本拆成固定 6 列的分镜表。

    输入：script（重写时再加 feedback）
    输出：storyboard（Markdown 表格）
    """
    script = state.get("script") or ""
    if not script:
        raise NodeError("节点3 分镜生成", "没有收到节点2的剧本，无法分镜。")

    feedback = (state.get("feedback") or "").strip()
    _print_header(
        3,
        "分镜生成",
        f"剧本 {len(script)} 字"
        + (f"｜本轮带校验意见" if feedback else ""),
    )

    draft = invoke_structured(
        node_name="节点3 分镜生成",
        schema=StoryboardDraft,
        system_prompt=STORYBOARD_SYSTEM,
        user_prompt=build_storyboard_prompt(script, feedback=feedback),
        temperature=STORYBOARD_TEMPERATURE,
    )

    warnings: list[str] = []
    storyboard = render_storyboard(
        draft,
        title=state.get("title") or "",
        warnings=warnings,
    )
    for warning in warnings:
        print(f"[分镜警告] {warning}", flush=True)
    _print_output(storyboard)

    next_step = (
        "persist_story_node"
        if state.get("intent_type") in {"continue_story", "modify_script"}
        else "validate_node"
    )
    return {
        "storyboard": storyboard,
        "trace": [
            f"节点3 分镜生成完成：{len(draft.shots)} 个镜头 → {next_step}"
        ],
    }

