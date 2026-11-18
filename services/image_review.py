"""
分镜图审核（branch_validate_node 的图片层）：

三层检查，全部通过才算审核通过：
1. 人物数检查（规则）：镜头描述里只提到一个角色时，prompt/负面词必须强制单人画面；
2. 角色一致性（CLIP）：有基准图时，生成图与角色库最佳匹配必须属于镜头出场的角色；
3. 内容 OOC（LLM，可选）：DeepSeek 对照角色档案检查 prompt 是否偏离镜头描述/人设。

返回统一结构：
{
  "passed": bool,
  "person_check": {...}, "character_check": {...}, "ooc_check": {...},
  "issues": [str, ...],
  "suggested_prompt_fix": str   # 需人工修改 prompt 时的修改建议
}
"""

from __future__ import annotations

from typing import Any

import config

try:
    from langsmith import traceable
except ImportError:  # pragma: no cover
    def traceable(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

from pydantic import BaseModel, Field

from services.character_reference import image_similarity
from state import ComicState


class ImageReviewResult(BaseModel):
    """LLM 审核的结构化输出。"""

    passed: bool = Field(description="整体是否通过")
    person_count_ok: bool = Field(description="画面人物数量是否符合镜头描述")
    character_match_ok: bool = Field(description="出场角色是否与镜头描述一致")
    ooc_ok: bool = Field(description="角色行为/外貌是否符合人设")
    issues: list[str] = Field(default_factory=list, description="发现的问题列表")
    suggested_prompt_fix: str = Field(default="", description="建议如何修改绘图 prompt")


# ---------------------------------------------------------------- 人物数检查
def expected_characters(shot_text: str, profiles: list[dict[str, Any]]) -> list[str]:
    """从镜头描述里找出出场的角色名（按角色档案名单匹配）。

    匹配规则：全名命中，或 3 字以上名字的"去姓氏称呼"命中（如 韩月瑶 -> 月瑶）。
    """
    text = str(shot_text or "")
    names: list[str] = []
    for card in profiles or []:
        name = str(card.get("name") or "").strip()
        if not name:
            continue
        if name in text or (len(name) >= 3 and name[1:] in text):
            names.append(name)
    return names


def _single_person_guard(
    expected: list[str],
    positive: str,
    negative: str,
) -> tuple[list[str], str, str, list[str]]:
    """
    单人镜头时强化 prompt：
    追加"仅一人出镜"正向约束 + 多人负面词。返回 (issues, positive, negative, expected)。
    """
    issues: list[str] = []
    if len(expected) == 1:
        name = expected[0]
        if f"仅{name}" not in positive and f"只有{name}" not in positive:
            positive += f"。单人画面：画面中只有{name}一人出镜，不要出现其他人物。"
            issues.append(
                f"镜头只提到{name}，已在 prompt 追加单人约束（防止画出多人）"
            )
        if "多人" not in negative:
            negative += "，多人，多个人物，双人，两个角色，群体"
    elif len(expected) > 1:
        positive += f"。画面人物仅限：{'、'.join(expected)}，不要出现其他人物。"
    return issues, positive, negative, expected


# ---------------------------------------------------------------- CLIP 检查
def _clip_character_check(
    story_id: str,
    image_path: str,
    expected: list[str],
    similarity_threshold: float = 0.0,
) -> dict[str, Any]:
    """生成图与角色库基准图比对：最佳匹配角色必须在镜头出场名单里。"""
    check = image_similarity(story_id, image_path)
    result = {
        "similarity": check.get("similarity"),
        "best_match": check.get("character") or "",
        "threshold": check.get("threshold"),
        "backend": check.get("embedding_backend") or "",
        "passed": True,
        "issues": [],
    }
    if check.get("embedding_backend") == "unavailable" or check.get("error"):
        # 嵌入不可用时不能把链路判死，只提示。
        return result
    if check.get("embedding_backend") in ("unavailable", "perceptual_fallback"):
        # CLIP 不可用时的兜底向量没有语义可比性（相似度恒接近 0），
        # 不能拿它做相似度判定，否则审核永远失败、白白烧出图额度。
        result["skipped"] = True
        result["issues"] = []
        return result
    best = result["best_match"]
    if expected and best and best not in expected:
        result["passed"] = False
        result["issues"].append(
            f"画面主体更像「{best}」，但镜头出场角色是：{'、'.join(expected)}"
        )
    # image_similarity 自带的相似度阈值判定（与角色基准库一致）
    if check.get("passed") is False:
        result["passed"] = False
        result["issues"].append(
            f"与基准图相似度 {check.get('similarity')} "
            f"低于阈值 {check.get('threshold')}（{best or '-'}）"
        )
    return result


# ---------------------------------------------------------------- LLM 检查
_REVIEW_SYSTEM = (
    "你是漫剧画面的图片审核节点。给定：镜头描述、角色档案、将要使用的绘图 prompt。"
    "请检查：1) prompt 画出的人物数量是否与镜头一致（镜头只有一人时 prompt 不得引导多人）；"
    "2) 出场角色是否与镜头描述一致，没有混入其他镜头/其他角色；"
    "3) 角色行为与外貌是否符合人设（OOC）。"
    "只依据文本判断，输出 ImageReviewResult 结构。轻微问题也必须列出，"
    "只有会影响成图的问题才令 passed=false。"
)


def _llm_ooc_check(
    shot_text: str,
    profiles: list[dict[str, Any]],
    positive: str,
) -> dict[str, Any] | None:
    """LLM 文本层审核；未启用/失败返回 None（不阻塞）。"""
    if not config.IMAGE_REVIEW_USE_LLM:
        return None
    try:
        from nodes.shared import invoke_structured

        cards = "\n".join(
            f"- {c.get('name')}：{c.get('appearance') or '原文未说明'}"
            for c in (profiles or [])[:8]
        )
        user_prompt = (
            f"【镜头描述】\n{shot_text}\n\n【角色档案】\n{cards or '（无）'}\n\n"
            f"【绘图 prompt】\n{positive}"
        )
        result = invoke_structured(
            node_name="分镜图审核",
            schema=ImageReviewResult,
            system_prompt=_REVIEW_SYSTEM,
            user_prompt=user_prompt,
            temperature=0.1,
        )
        data = result.model_dump()
        return {
            "passed": bool(data.get("passed")),
            "issues": list(data.get("issues") or []),
            "suggested_prompt_fix": str(data.get("suggested_prompt_fix") or ""),
        }
    except Exception as exc:
        print(f"[图片审核] LLM 审核跳过：{type(exc).__name__}: {exc}", flush=True)
        return None


# ---------------------------------------------------------------- 总入口
def review_shot_image(
    state: ComicState,
    shot_text: str,
    positive: str,
    negative: str,
    image_path: str = "",
) -> dict[str, Any]:
    """
    审核一次生成结果。image_path 为空时只做文本层检查（prompt 审核）。
    返回 {passed, person_check, character_check, ooc_check, issues, prompt}。
    prompt 字段是（可能被单人约束强化过的）正向 prompt，调用方应采用它重绘。
    """
    profiles = list(state.get("canon_profiles") or [])
    story_id = str(state.get("story_id") or "default")

    expected = expected_characters(shot_text, profiles)
    issues, new_positive, new_negative, expected = _single_person_guard(
        expected, positive, negative
    )
    person_check = {
        "expected": expected,
        "multi_person": len(expected) != 1,
        "passed": True,
        "issues": issues,
    }

    character_check: dict[str, Any] = {"passed": True, "issues": [], "skipped": True}
    if image_path:
        character_check = _clip_character_check(
            story_id, image_path, expected
        )

    ooc_check = _llm_ooc_check(shot_text, profiles, new_positive) or {
        "passed": True,
        "issues": [],
        "suggested_prompt_fix": "",
        "skipped": True,
    }

    all_issues = list(person_check["issues"]) + list(character_check["issues"]) + list(
        ooc_check["issues"]
    )
    passed = person_check["passed"] and character_check["passed"] and ooc_check["passed"]

    return {
        "passed": passed,
        "person_check": person_check,
        "character_check": character_check,
        "ooc_check": ooc_check,
        "issues": all_issues,
        "suggested_prompt_fix": str(ooc_check.get("suggested_prompt_fix") or ""),
        "prompt": new_positive,
        "negative": new_negative,
    }
