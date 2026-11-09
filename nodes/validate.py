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

def _build_feedback(
    validation: dict[str, Any],
    round_number: int,
    revision_count: int,
    max_revisions: int,
    history: list[dict[str, Any]] | None = None,
) -> str:
    """
    把“历史驳回问题 + 本轮问题”一起整理成下一轮修改意见。

    只传本轮问题会让模型原地换皮；这里必须累计历史问题，并要求逐条解决。
    """
    history = history or []
    rounds = [(index + 1, item) for index, item in enumerate(history)]
    rounds.append((round_number, validation))

    lines = [
        f"【第{round_number}轮校验未通过】",
        f"当前已重写 {revision_count} 次，最多允许 {max_revisions} 次。",
        "这是 DeepSeek 修复轮：请按问题逐条做实质性修改，"
        "不能只换同义表达/换皮，也不能把已解决的问题重新引入。",
        "",
    ]

    index = 0
    truncated = False
    for round_no, item in rounds:
        for issue in item.get("issues") or []:
            index += 1
            if index > MAX_ACCUMULATED_ISSUES:
                truncated = True
                break
            prefix = "历史" if round_no < round_number else "本轮"
            lines.append(
                f"{index}. 【{prefix}·第{round_no}轮】"
                f"[{issue.get('category')}/{issue.get('severity')}] "
                f"{issue.get('location')}"
            )
            lines.append(f"   问题：{issue.get('description')}")
            lines.append(f"   建议：{issue.get('suggestion')}")
        if truncated:
            break

    if truncated:
        lines.append(
            f"……累计问题超过 {MAX_ACCUMULATED_ISSUES} 条，其余未展开；"
            "请优先解决上面列出的重复问题和严重问题。"
        )

    feedback = "\n".join(lines)
    max_chars = max(MAX_FEEDBACK_CHARS, 4_000)
    if len(feedback) > max_chars:
        feedback = feedback[:max_chars] + "\n……（反馈过长，已截断）"
    return feedback

def validate_node(state: ComicState) -> dict[str, Any]:
    """
    节点4：只做两条校验——剧情是否偏离、人物是否 OOC。

    输入：script + storyboard + plot_summary + canon_profiles
    输出：validation / validation_history / feedback
    """
    script = state.get("script") or ""
    storyboard = state.get("storyboard") or ""
    plot_summary = state.get("plot_summary") or ""
    if not script or not storyboard:
        raise NodeError("节点4 校验", "缺少剧本或分镜表，无法校验。")
    if not plot_summary:
        raise NodeError("节点4 校验", "缺少节点1的剧情摘要，无法判断剧情忠实度。")

    canon_text = format_character_cards(state.get("canon_profiles") or [])
    history_before = state.get("validation_history") or []
    round_number = len(history_before) + 1

    _print_header(
        4,
        f"校验（第{round_number}轮）",
        f"剧本 {len(script)} 字｜分镜 {len(storyboard)} 字｜事实基准 {len(plot_summary)} 字",
    )

    result = invoke_structured(
        node_name="节点4 校验",
        schema=ValidationResult,
        system_prompt=VALIDATE_SYSTEM,
        user_prompt=build_validate_prompt(
            plot_summary=plot_summary,
            canon_text=canon_text,
            script=script,
            storyboard=storyboard,
        ),
        temperature=VALIDATE_TEMPERATURE,
    )

    validation = result.model_dump()
    issues = validation.get("issues") or []

    # 任何实质问题都算不通过；布尔值和问题列表取“更严格”的一方。
    passed = bool(
        validation.get("plot_faithful")
        and validation.get("character_consistent")
        and not issues
    )

    # 模型说“不通过”却没列问题：补一条可执行的通用问题，避免反馈为空。
    if not passed and not issues:
        category = (
            "人物OOC"
            if validation.get("plot_faithful") and not validation.get("character_consistent")
            else "剧情偏离"
        )
        issues = [
            {
                "category": category,
                "severity": "严重",
                "location": "全局",
                "description": "模型判定未通过，但没有列出具体问题。",
                "suggestion": "请重新对照剧情摘要和人物设定，逐场检查关键事件、人物关系、说话风格是否一致。",
            }
        ]
        validation["issues"] = issues

    # 模型说“通过”却列了问题：仍然按不通过处理。
    if issues and validation.get("plot_faithful") and validation.get("character_consistent"):
        print(
            "[校验提示] 模型同时给出了“通过”和问题清单，按更严格的结果处理：不通过。",
            flush=True,
        )

    validation["passed"] = passed

    revision_count = int(state.get("revision_count", 0))
    max_revisions = int(state.get("max_revisions", 0))
    feedback = (
        _build_feedback(
            validation,
            round_number,
            revision_count,
            max_revisions,
            history_before,
        )
        if not passed
        else ""
    )

    report = render_validation(
        validation,
        revision_count=revision_count,
        max_revisions=max_revisions,
        history=None,
    )
    _print_output(report)

    if passed:
        next_route = "persist_node"
        print("[路由] 校验通过 → persist_node（写入 story_memory 后结束）。", flush=True)
    elif revision_count < max_revisions:
        next_route = "script_node"
        print(
            f"[路由] 校验未通过，将回到节点2重写（已重写 {revision_count}/{max_revisions} 次）。",
            flush=True,
        )
    else:
        next_route = "end"
        print(
            f"[路由] 校验未通过，但已达到重写上限（{revision_count}/{max_revisions}），"
            "输出最后一版并标记为需人工复核。",
            flush=True,
        )

    return {
        "validation": validation,
        "validation_history": [validation],
        "feedback": feedback,
        "trace": [
            f"节点4 校验第{round_number}轮："
            f"{'通过' if passed else '未通过'}"
            f"（问题 {len(issues)} 条）→ {next_route}"
        ],
    }

