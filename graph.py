"""
LangGraph 状态机：前置剧情压缩 + 意图识别 + 三路分流 + 文生图/视频预留。

图结构：
    START → context_compress_node → intent_node
      ├─ new_script → parse → script → shot_generate → branch_validate(OOC)
      │                                  → build_video_prompt → image → video → persist
      ├─ continue_story → retrieve_continue → continue_script → shot_generate
      │                                  → branch_validate(OOC) → build_video_prompt → image → video → persist
      ├─ modify_script → retrieve_storyboard → modify_script → shot_generate
      │                                  → branch_validate(OOC) → build_video_prompt → image → video → persist
      └─ unknown → error_node
"""

from __future__ import annotations

from typing import Any, Literal

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from config import INTENT_HIGH_CONFIDENCE, INTENT_LOW_CONFIDENCE
from image_gen import image_gen_node
from node_shot_generate import shot_generate_node
from node_video_prompt import build_video_prompt_node
from nodes import (
    branch_validate_node,
    confirm_node,
    context_compress_node,
    continue_script_node,
    error_node,
    intent_node,
    modify_script_node,
    parse_node,
    persist_story_node,
    retrieve_continue_node,
    retrieve_storyboard_node,
    script_node,
    storyboard_node,
    validate_node,
)
from state import ComicState
from video_gen import video_graph_node


def route_after_intent(
    state: ComicState,
) -> Literal[
    "parse_node",
    "retrieve_continue_node",
    "retrieve_storyboard_node",
    "confirm_node",
    "error_node",
]:
    """
    意图识别后的置信度路由。

    不能只看 intent，必须同时读取 confidence：
    - intent=unknown 或 confidence <= 0.60 → error_node；
    - 0.60 < confidence < 0.85 → confirm_node；
    - confidence >= 0.85 → 按 intent 分流。
    """
    intent = str(state.get("intent_type") or "unknown")
    confidence = float(state.get("intent_confidence") or 0.0)

    if intent == "unknown" or confidence <= INTENT_LOW_CONFIDENCE:
        return "error_node"
    if confidence < INTENT_HIGH_CONFIDENCE:
        return "confirm_node"

    mapping = {
        "new_script": "parse_node",
        "continue_story": "retrieve_continue_node",
        "modify_script": "retrieve_storyboard_node",
    }
    return mapping.get(intent, "error_node")


def route_after_confirm(
    state: ComicState,
) -> Literal[
    "parse_node",
    "retrieve_continue_node",
    "retrieve_storyboard_node",
    "error_node",
]:
    """二次确认后的路由：确认则按 intent 走，取消则 error_node。"""
    if not state.get("confirmed"):
        return "error_node"
    mapping = {
        "new_script": "parse_node",
        "continue_story": "retrieve_continue_node",
        "modify_script": "retrieve_storyboard_node",
    }
    return mapping.get(str(state.get("intent_type") or "unknown"), "error_node")


def route_after_continue_retrieval(
    state: ComicState,
) -> Literal["continue_script_node", "error_node"]:
    """续写检索失败（没有历史/Chroma 异常）时进入错误节点。"""
    return "error_node" if state.get("error_message") else "continue_script_node"


def route_after_storyboard_retrieval(
    state: ComicState,
) -> Literal["modify_script_node", "error_node"]:
    """修改检索失败（目标集不存在/缺剧本）时进入错误节点。"""
    return "error_node" if state.get("error_message") else "modify_script_node"


def route_after_storyboard(
    state: ComicState,
) -> Literal["validate_node", "image_gen_node"]:
    """
    分镜之后：
    - 旧路径 new_script 走原有校验（当前主链已改为 shot_generate）；
    - continue/modify 已经过了分支剧情校验，直接进入文生图。
    """
    if state.get("intent_type") in {"continue_story", "modify_script"}:
        return "image_gen_node"
    return "validate_node"


def route_after_branch_validation(
    state: ComicState,
) -> Literal[
    "storyboard_node",
    "shot_generate_node",
    "build_video_prompt_node",
    "continue_script_node",
    "modify_script_node",
    "error_node",
]:
    """
    new_script：分镜已经生成，branch_validate 做 OOC 校验；
    - 通过 → build_video_prompt_node；
    - 未通过 → shot_generate_node 重写；
    continue/modify：仍在剧本阶段校验，通过后去 storyboard。
    """
    validation = state.get("branch_validation") or {}
    has_shots = bool(state.get("shots"))
    if validation.get("passed"):
        if has_shots:
            return "build_video_prompt_node"
        return "storyboard_node"
    if state.get("error_message"):
        return "error_node"
    if has_shots:
        return "shot_generate_node"
    if state.get("intent_type") == "continue_story":
        return "continue_script_node"
    return "modify_script_node"


def route_after_validation(
    state: ComicState,
) -> Literal["script_node", "image_gen_node", "end"]:
    """
    校验后的路由。

    - 通过：进入文生图 → 文生视频预留 → 持久化；
    - 不通过且还能重写：回节点2；
    - 不通过但达到上限：结束。
    """
    validation = state.get("validation") or {}
    if validation.get("passed"):
        return "image_gen_node"

    revision_count = int(state.get("revision_count", 0))
    max_revisions = int(state.get("max_revisions", 0))
    if revision_count >= max_revisions:
        return "end"

    return "script_node"


def route_after_image(
    state: ComicState,
) -> Literal["video_gen_node", "error_node"]:
    """角色基准图相似度校验失败时终止，不进入视频和持久化。"""
    if state.get("image_status") == "character_ooc_failed":
        return "error_node"
    return "video_gen_node"


def build_graph() -> Any:
    """组装并编译状态图。"""
    builder = StateGraph(ComicState)

    # 新增分支节点
    builder.add_node("context_compress_node", context_compress_node)
    builder.add_node("intent_node", intent_node)
    builder.add_node("confirm_node", confirm_node)
    builder.add_node("retrieve_continue_node", retrieve_continue_node)
    builder.add_node("continue_script_node", continue_script_node)
    builder.add_node("retrieve_storyboard_node", retrieve_storyboard_node)
    builder.add_node("modify_script_node", modify_script_node)
    builder.add_node("branch_validate_node", branch_validate_node)
    builder.add_node("shot_generate_node", shot_generate_node)
    builder.add_node("build_video_prompt_node", build_video_prompt_node)
    builder.add_node("image_gen_node", image_gen_node)
    builder.add_node("video_gen_node", video_graph_node)
    builder.add_node("persist_story_node", persist_story_node)
    builder.add_node("error_node", error_node)

    # 原有 4 个节点
    builder.add_node("parse_node", parse_node)
    builder.add_node("script_node", script_node)
    builder.add_node("storyboard_node", storyboard_node)
    builder.add_node("validate_node", validate_node)

    # 图的最开始：续写前压缩最近剧情 → 意图识别
    builder.add_edge(START, "context_compress_node")
    builder.add_edge("context_compress_node", "intent_node")
    builder.add_conditional_edges(
        "intent_node",
        route_after_intent,
        {
            "parse_node": "parse_node",
            "retrieve_continue_node": "retrieve_continue_node",
            "retrieve_storyboard_node": "retrieve_storyboard_node",
            "confirm_node": "confirm_node",
            "error_node": "error_node",
        },
    )

    # 中等置信度：二次确认后再分流
    builder.add_conditional_edges(
        "confirm_node",
        route_after_confirm,
        {
            "parse_node": "parse_node",
            "retrieve_continue_node": "retrieve_continue_node",
            "retrieve_storyboard_node": "retrieve_storyboard_node",
            "error_node": "error_node",
        },
    )

    # new_script：剧本 → 结构化分镜 → 分镜 OOC 校验 → 视频提示词 → 文生图
    builder.add_edge("parse_node", "script_node")
    builder.add_edge("script_node", "shot_generate_node")
    builder.add_edge("shot_generate_node", "branch_validate_node")

    # continue_story：检索 → 续写剧本 → 剧情忠实度校验 → 分镜
    builder.add_conditional_edges(
        "retrieve_continue_node",
        route_after_continue_retrieval,
        {
            "continue_script_node": "continue_script_node",
            "error_node": "error_node",
        },
    )
    builder.add_edge("continue_script_node", "shot_generate_node")

    # modify_script：检索 → 局部修改剧本 → 分镜 → OOC 校验
    builder.add_conditional_edges(
        "retrieve_storyboard_node",
        route_after_storyboard_retrieval,
        {
            "modify_script_node": "modify_script_node",
            "error_node": "error_node",
        },
    )
    builder.add_edge("modify_script_node", "shot_generate_node")

    # 分支校验：通过才进分镜；失败回剧本；超限进错误节点且不持久化
    builder.add_conditional_edges(
        "branch_validate_node",
        route_after_branch_validation,
        {
            "storyboard_node": "storyboard_node",
            "shot_generate_node": "shot_generate_node",
            "build_video_prompt_node": "build_video_prompt_node",
            "continue_script_node": "continue_script_node",
            "modify_script_node": "modify_script_node",
            "error_node": "error_node",
        },
    )

    # 分镜之后：new_script 去原有校验；续写/修改已通过分支校验，直接文生图
    builder.add_conditional_edges(
        "storyboard_node",
        route_after_storyboard,
        {
            "validate_node": "validate_node",
            "image_gen_node": "image_gen_node",
        },
    )

    # 校验之后：通过进入文生图；失败回剧本或结束
    builder.add_conditional_edges(
        "validate_node",
        route_after_validation,
        {
            "script_node": "script_node",
            "image_gen_node": "image_gen_node",
            "end": END,
        },
    )

    # 新剧本/续写不再自动生图；图片由“图片生成”独立接口触发。
    builder.add_edge("build_video_prompt_node", "video_gen_node")
    builder.add_edge("video_gen_node", "persist_story_node")
    builder.add_edge("persist_story_node", END)
    builder.add_edge("error_node", END)

    # InMemorySaver 只保存在当前进程内，适合 Demo；
    # 它能让我们在异常时通过 graph.get_state(config) 查看已经跑到哪一步。
    return builder.compile(checkpointer=InMemorySaver())


# 供 main.py 直接导入。
graph = build_graph()
