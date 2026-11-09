"""分镜 OOC 校验：由 branch_validate_node 在 new_script 路径调用。"""

from __future__ import annotations

from typing import Any

import config
from global_profile import load_global_profile
from node_shared import NodeError, _print_stage, _print_output, invoke_structured
from prompts_shots import SHOT_OOC_SYSTEM, build_shot_ooc_prompt
from state import BranchValidationResult, ComicState


def _feedback(validation: dict[str, Any]) -> str:
    lines = ["【分镜 OOC 校验未通过】"]
    for index, issue in enumerate(validation.get("issues") or [], start=1):
        lines.append(
            f"{index}. [{issue.get('category')}/{issue.get('severity')}] "
            f"{issue.get('location')}：{issue.get('description')}"
            f" → {issue.get('suggestion')}"
        )
    return "\n".join(lines)


def validate_shot_ooc(state: ComicState) -> dict[str, Any]:
    """对 shot_generate_node 输出的镜头列表做 OOC 校验。"""
    story_id = str(state.get("story_id") or "default")
    shots = state.get("shots") or []
    if not shots:
        raise NodeError("分镜 OOC 校验", "没有 shot 列表，无法校验。")
    profile = state.get("global_profile") or load_global_profile(story_id) or {}
    if not profile.get("characters"):
        fallback_cards = state.get("canon_profiles") or state.get("history_roles") or []
        if fallback_cards:
            profile = {"characters": fallback_cards}
    if not profile.get("characters"):
        raise NodeError("分镜 OOC 校验", "缺少全局角色档案。")

    history_before = state.get("branch_validation_history") or []
    round_number = len(history_before) + 1
    revision_count = int(state.get("branch_revision_count", 0))
    max_revisions = int(
        state.get("branch_max_revisions") or config.MAX_BRANCH_REVISIONS
    )
    _print_stage(
        f"分镜 OOC 校验（第{round_number}轮）",
        f"{story_id} 第{state.get('episode_no')}集｜{len(shots)} 镜头",
    )

    result: BranchValidationResult = invoke_structured(
        node_name="分镜 OOC 校验",
        schema=BranchValidationResult,
        system_prompt=SHOT_OOC_SYSTEM,
        user_prompt=build_shot_ooc_prompt(
            global_profile=profile,
            shots=shots,
            plot_summary=str(state.get("plot_summary") or ""),
            script=str(state.get("script") or ""),
            feedback=str(state.get("shot_feedback") or ""),
        ),
        temperature=0.2,
    )
    validation = result.model_dump()
    issues = validation.get("issues") or []
    severe = [item for item in issues if str(item.get("severity")) == "严重"]
    passed = not severe
    if not result.passed and not issues:
        issues = [
            {
                "category": "人物OOC",
                "severity": "严重",
                "location": "全局",
                "description": "模型判定未通过，但没有列出问题。",
                "suggestion": "对照全局角色档案重新检查镜头。",
            }
        ]
    validation["passed"] = passed
    validation["issues"] = issues

    lines = [
        f"**结论**：{'✅ 通过' if passed else '❌ 未通过'}",
        f"**问题数**：{len(issues)}",
        f"**总结**：{validation.get('summary', '')}",
    ]
    for index, issue in enumerate(issues, start=1):
        lines.append(
            f"{index}. [{issue.get('category')}/{issue.get('severity')}] "
            f"{issue.get('location')}：{issue.get('description')}"
        )
    _print_output("\n".join(lines))

    if passed:
        return {
            "branch_validation": validation,
            "branch_validation_history": [validation],
            "shot_feedback": "",
            "trace": [f"分镜 OOC 校验第{round_number}轮：通过"],
        }

    if revision_count >= max_revisions:
        message = (
            f"分镜连续 {round_number} 轮未通过 OOC 校验，已达到重写上限；"
            "本次结果不会写入 story_memory。"
        )
        return {
            "branch_validation": validation,
            "branch_validation_history": [validation],
            "shot_feedback": _feedback(validation),
            "branch_feedback": _feedback(validation),
            "error_message": message,
            "trace": [f"分镜 OOC 校验第{round_number}轮：达上限 → error_node"],
        }

    feedback = _feedback(validation)
    return {
        "branch_validation": validation,
        "branch_validation_history": [validation],
        "shot_feedback": feedback,
        "branch_feedback": feedback,
        "branch_revision_count": revision_count + 1,
        "trace": [
            f"分镜 OOC 校验第{round_number}轮：未通过 → shot_generate_node 重写"
        ],
    }
