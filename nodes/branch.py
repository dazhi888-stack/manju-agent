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
from .shot_validate import validate_shot_ooc

def _format_history_summaries(summaries: list[dict[str, Any]]) -> str:
    """把历史摘要文档拼成校验提示词需要的文本。"""
    if not summaries:
        return "（没有历史摘要）"
    parts: list[str] = []
    for doc in summaries:
        metadata = doc.get("metadata") or {}
        episode_no = metadata.get("episode_no", "?")
        title = metadata.get("title", "")
        content = doc.get("document", "")
        parts.append(f"### 第{episode_no}集｜{title}\n{content}")
    return "\n\n".join(parts)

def _normalize_issue_part(value: Any) -> str:
    text = str(value or "").lower()
    text = re.sub(r"[\s，。；：、,.;:!?！？（）()【】\[\]“”\"'`]+", "", text)
    return text[:160]

def _ledger_from_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """兼容旧 state：没有台账时，从历史校验记录生成初始台账。"""
    ledger: list[dict[str, Any]] = []
    for round_no, item in enumerate(history, start=1):
        for issue in item.get("issues") or []:
            ledger.append(
                {
                    "id": f"I{round_no}-{len(ledger) + 1}",
                    "status": "open",
                    "first_round": round_no,
                    "last_round": round_no,
                    "repeat_count": 1,
                    "category": issue.get("category", ""),
                    "severity": issue.get("severity", ""),
                    "location": issue.get("location", ""),
                    "description": issue.get("description", ""),
                    "suggestion": issue.get("suggestion", ""),
                    "location_key": _normalize_issue_part(issue.get("location")),
                    "description_key": _normalize_issue_part(
                        issue.get("description")
                    ),
                }
            )
    return ledger

def _match_ledger_entry(
    issue: dict[str, Any],
    ledger: list[dict[str, Any]],
) -> dict[str, Any] | None:
    issue_id = str(issue.get("issue_id") or "").strip()
    if issue_id:
        for entry in ledger:
            if str(entry.get("id")) == issue_id:
                return entry

    category = str(issue.get("category") or "")
    location_key = _normalize_issue_part(issue.get("location"))
    description_key = _normalize_issue_part(issue.get("description"))
    for entry in ledger:
        if str(entry.get("category") or "") != category:
            continue
        if location_key and location_key == entry.get("location_key"):
            return entry
        previous_description = str(entry.get("description_key") or "")
        if (
            description_key
            and previous_description
            and (
                description_key[:48] in previous_description
                or previous_description[:48] in description_key
            )
        ):
            return entry
    return None

def _reconcile_issue_ledger(
    previous: list[dict[str, Any]],
    issues: list[dict[str, Any]],
    round_number: int,
    passed: bool,
    resolved_ids: list[str] | None = None,
    repeated_ids: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    维护跨轮问题台账。

    - 同一问题复用原 ID，status=repeated，repeat_count+1；
    - 本轮未再出现的旧问题标记 resolved，后续反馈不再重复；
    - 新问题分配新 ID，status=open。
    """
    ledger = [dict(entry) for entry in previous]
    resolved_set = {str(item) for item in (resolved_ids or [])}
    repeated_set = {str(item) for item in (repeated_ids or [])}
    matched_ids: set[str] = set()

    for issue in issues:
        entry = _match_ledger_entry(issue, ledger)
        if entry is None:
            entry = {
                "id": f"I{round_number}-{len(ledger) + 1}",
                "status": "open",
                "first_round": round_number,
                "last_round": round_number,
                "repeat_count": 1,
                "category": issue.get("category", ""),
                "severity": issue.get("severity", ""),
                "location": issue.get("location", ""),
                "description": issue.get("description", ""),
                "suggestion": issue.get("suggestion", ""),
                "location_key": _normalize_issue_part(issue.get("location")),
                "description_key": _normalize_issue_part(
                    issue.get("description")
                ),
            }
            ledger.append(entry)
        else:
            entry["status"] = "resolved" if passed else "repeated"
            entry["last_round"] = round_number
            entry["repeat_count"] = int(entry.get("repeat_count", 1)) + 1
            entry.update(
                {
                    "category": issue.get("category", ""),
                    "severity": issue.get("severity", ""),
                    "location": issue.get("location", ""),
                    "description": issue.get("description", ""),
                    "suggestion": issue.get("suggestion", ""),
                    "location_key": _normalize_issue_part(issue.get("location")),
                    "description_key": _normalize_issue_part(
                        issue.get("description")
                    ),
                }
            )
        issue["issue_id"] = entry["id"]
        issue["status"] = entry["status"]
        matched_ids.add(str(entry["id"]))
        if passed:
            entry["status"] = "resolved"
            issue["status"] = "resolved"

    for entry in ledger:
        entry_id = str(entry.get("id"))
        if passed:
            entry["status"] = "resolved"
            continue
        if entry_id in resolved_set and entry_id not in matched_ids:
            entry["status"] = "resolved"
            continue
        if entry_id in repeated_set and entry_id not in matched_ids:
            entry["status"] = "repeated"
            entry["last_round"] = round_number
            continue
        if entry_id not in matched_ids and entry.get("status") != "resolved":
            # 校验器本轮没有再报告：按已解决处理，避免下一轮反复追同一条。
            entry["status"] = "resolved"
            entry["resolution_note"] = "本轮未再出现"

    return ledger, issues

def _format_ledger_for_prompt(ledger: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    resolved: list[str] = []
    for entry in ledger:
        if entry.get("status") == "resolved":
            resolved.append(str(entry.get("id")))
            continue
        status = "重复驳回" if entry.get("status") == "repeated" else "未解决"
        repeat_count = int(entry.get("repeat_count", 1))
        lines.append(
            f"- [{entry.get('id')}][{status}][重复{repeat_count}次]"
            f"[{entry.get('category')}/{entry.get('severity')}] "
            f"{entry.get('location')}：{entry.get('description')}"
            f"（建议：{entry.get('suggestion')}）"
        )
        if len(lines) >= MAX_ACCUMULATED_ISSUES:
            break
    if resolved:
        lines.append(f"已解决，不要重新引入：{'、'.join(resolved[:12])}")
    return "\n".join(lines)

def _build_branch_feedback(
    validation: dict[str, Any],
    round_number: int,
    revision_count: int,
    max_revisions: int,
    history: list[dict[str, Any]] | None = None,
    ledger: list[dict[str, Any]] | None = None,
) -> str:
    """
    生成 DeepSeek 修复轮的台账反馈。

    只把未解决/重复驳回的问题交给修复模型；已解决问题单独列出但禁止重新引入。
    """
    history = history or []
    ledger = ledger or _ledger_from_history(history + [validation])

    lines = [
        f"【第{round_number}轮剧情忠实度校验未通过】",
        f"当前已重写 {revision_count} 次，最多允许 {max_revisions} 次。",
        "这是 DeepSeek 修复轮，不是重新创作。请按问题 ID 逐条做实质性修改；"
        "只换同义词/换句式/换皮不算解决，禁止把已解决问题重新引入。",
        "",
        "【必须处理的问题台账】",
    ]

    open_entries = [
        entry
        for entry in ledger
        if entry.get("status") in {"open", "repeated"}
    ][:MAX_ACCUMULATED_ISSUES]
    if open_entries:
        for entry in open_entries:
            status = "重复驳回" if entry.get("status") == "repeated" else "未解决"
            repeat_count = int(entry.get("repeat_count", 1))
            lines.append(
                f"- [{entry.get('id')}][{status}][第{repeat_count}次]"
                f"[{entry.get('category')}/{entry.get('severity')}] "
                f"{entry.get('location')}"
            )
            lines.append(f"  问题：{entry.get('description')}")
            lines.append(f"  建议：{entry.get('suggestion')}")
            if status == "重复驳回":
                lines.append(
                    "  要求：上一轮没有实质修正；本轮必须改变动作/结果/信息，"
                    "不能只改措辞。"
                )
    else:
        for index, issue in enumerate(validation.get("issues") or [], start=1):
            lines.append(
                f"- [{issue.get('issue_id') or f'R{index}'}]"
                f"[{issue.get('category')}/{issue.get('severity')}] "
                f"{issue.get('location')}：{issue.get('description')}"
            )

    resolved_entries = [
        entry for entry in ledger if entry.get("status") == "resolved"
    ]
    if resolved_entries:
        lines.append("")
        lines.append("【已解决，禁止重新引入】")
        for entry in resolved_entries[:12]:
            lines.append(f"- [{entry.get('id')}] {entry.get('description')}")

    feedback = "\n".join(lines)
    if len(feedback) > MAX_BRANCH_FEEDBACK_CHARS:
        feedback = (
            feedback[:MAX_BRANCH_FEEDBACK_CHARS]
            + "\n……（分支反馈过长，已截断）"
        )
    return feedback

def branch_validate_node(state: ComicState) -> dict[str, Any]:
    """
    分镜 OOC 校验（new/continue/modify 通用）：
    生成镜头之后、进入文生图和持久化之前执行。

    校验依据：
    - 本地全局角色档案（global_profile）
    - 本集主线/剧本
    - shot_generate_node 产出的镜头列表

    失败时返回修改意见并回到 shot_generate_node；达到上限进 error_node。
    """
    script = state.get("script") or ""
    intent_type = str(state.get("intent_type") or "")
    if state.get("shots"):
        # new/continue/modify 的镜头生成后，都先做分镜 OOC 校验。
        return validate_shot_ooc(state)
    if intent_type == "new_script":
        # new_script 的分镜先由 shot_generate_node 生成，再做 OOC 校验。
        return validate_shot_ooc(state)
    if not script:
        raise NodeError("分支剧情校验", "缺少待校验剧本。")
    if intent_type not in {"continue_story", "modify_script"}:
        raise NodeError("分支剧情校验", f"不支持的意图：{intent_type}")

    history_context = str(state.get("compressed_context") or "").strip()
    if not history_context:
        history_context = _format_history_summaries(
            state.get("history_summaries") or []
        )
    canon_text = format_character_cards(state.get("canon_profiles") or [])
    instruction = state.get("user_instruction") or (state.get("raw_text") or "")
    change_summary = state.get("change_summary") or ""

    history_before = state.get("branch_validation_history") or []
    round_number = len(history_before) + 1
    revision_count = int(state.get("branch_revision_count", 0))
    raw_max_revisions = state.get("branch_max_revisions")
    max_revisions = (
        int(raw_max_revisions)
        if raw_max_revisions is not None
        else MAX_BRANCH_REVISIONS
    )

    _print_stage(
        f"剧情忠实度校验（第{round_number}轮）",
        f"意图={intent_type}｜剧本 {len(script)} 字"
        f"｜角色卡 {len(state.get('canon_profiles') or [])} 张"
        f"｜历史摘要 {len(state.get('history_summaries') or [])} 集"
        f"｜台账问题 {sum(1 for item in (state.get('branch_issue_ledger') or []) if item.get('status') != 'resolved')} 条",
    )

    previous_ledger = state.get("branch_issue_ledger") or _ledger_from_history(
        history_before
    )
    previous_issues_text = _format_ledger_for_prompt(previous_ledger)

    result = invoke_structured(
        node_name="分支剧情校验",
        schema=BranchValidationResult,
        system_prompt=BRANCH_VALIDATE_SYSTEM,
        user_prompt=build_branch_validate_prompt(
            intent_type=intent_type,
            script=script,
            canon_text=canon_text,
            history_context=history_context,
            instruction=instruction,
            change_summary=change_summary,
            previous_issues=previous_issues_text,
        ),
        temperature=BRANCH_VALIDATE_TEMPERATURE,
    )

    validation = result.model_dump()
    issues = validation.get("issues") or []
    # 只有“严重”问题阻塞分镜；轻微语气/节奏问题作为提示，避免无限重写。
    severe_issues = [
        issue for issue in issues if str(issue.get("severity")) == "严重"
    ]
    passed = not severe_issues
    # 模型明确说没通过、又没给任何问题：保守起见仍然按不通过处理。
    if not result.passed and not issues:
        passed = False
        issues = [
            {
                "category": "剧情矛盾",
                "severity": "严重",
                "location": "全局",
                "description": "模型判定未通过，但没有列出具体问题。",
                "suggestion": "请重新对照角色人设和历史摘要，检查是否有 OOC、吃设定或剧情矛盾。",
            }
        ]
        validation["issues"] = issues
        severe_issues = issues
    validation["passed"] = passed

    ledger, issues = _reconcile_issue_ledger(
        previous_ledger,
        issues,
        round_number,
        passed,
        list(result.resolved_issue_ids or []),
        list(result.repeated_issue_ids or []),
    )
    validation["issues"] = issues
    validation["issue_ledger"] = ledger
    validation["resolved_issue_ids"] = [
        entry["id"] for entry in ledger if entry.get("status") == "resolved"
    ]
    validation["repeated_issue_ids"] = [
        entry["id"] for entry in ledger if entry.get("status") == "repeated"
    ]

    lines = [
        f"**结论**：{'✅ 通过' if passed else '❌ 未通过'}",
        f"**问题数**：{len(issues)}",
        f"**台账未解决**：{sum(1 for entry in ledger if entry.get('status') != 'resolved')} 条",
        f"**总结**：{validation.get('summary', '')}",
    ]
    for index, issue in enumerate(issues, start=1):
        lines.append(
            f"{index}. [{issue.get('issue_id') or '-'}]"
            f"[{issue.get('category')}/{issue.get('severity')}/"
            f"{issue.get('status') or 'open'}] "
            f"{issue.get('location')}：{issue.get('description')} "
            f"→ {issue.get('suggestion')}"
        )
    _print_output("\n".join(lines))

    if passed:
        print("[路由] 剧情忠实度校验通过，进入分镜。", flush=True)
        return {
            "branch_validation": validation,
            "branch_validation_history": [validation],
            "branch_issue_ledger": ledger,
            "branch_feedback": "",
            "trace": [
                f"分支剧情校验第{round_number}轮：通过（问题 0 条）→ storyboard_node"
            ],
        }

    feedback = _build_branch_feedback(
        validation,
        round_number,
        revision_count,
        max_revisions,
        history_before,
        ledger,
    )

    if revision_count >= max_revisions:
        message = (
            f"续写/修改剧本连续 {round_number} 轮未通过剧情忠实度校验，"
            "已达到重写上限；本次结果不会写入 story_memory。\n"
            "请查看上方问题清单，调整指令后重试。"
        )
        print(f"[路由] {message}", flush=True)
        return {
            "branch_validation": validation,
            "branch_validation_history": [validation],
            "branch_issue_ledger": ledger,
            "branch_feedback": feedback,
            "error_message": message,
            "trace": [
                f"分支剧情校验第{round_number}轮：未通过，已达上限 → error_node（不写 story_memory）"
            ],
        }

    print(
        f"[路由] 校验未通过，返回"
        f"{'续写' if intent_type == 'continue_story' else '修改'}节点重写"
        f"（已重写 {revision_count}/{max_revisions} 次）。",
        flush=True,
    )
    return {
        "branch_validation": validation,
        "branch_validation_history": [validation],
        "branch_issue_ledger": ledger,
        "branch_feedback": feedback,
        "trace": [
            f"分支剧情校验第{round_number}轮：未通过（问题 {len(issues)} 条），"
            f"返回 {'continue_script_node' if intent_type == 'continue_story' else 'modify_script_node'} 重写"
        ],
    }

