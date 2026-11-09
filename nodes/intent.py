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

INTENT_HELP = (
    "无法识别你的意图。请使用更明确的输入：\n"
    "  1) 粘贴完整小说章节/剧本 → 生成新剧本；\n"
    "  2) “续写第四夜”/“写下一集” → 续写已有故事；\n"
    "  3) “把第三夜结局改掉”/“调整镜头台词” → 局部修改；\n"
    "  可用 --story-id 指定故事ID。"
)
_CN_NUM = {
    "零": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}

def _unknown_intent_state(message: str, error: str = "", trace: str = "") -> dict[str, Any]:
    return {
        "intent_type": "unknown",
        "intent_error": error,
        "error_message": message,
        "trace": [trace or f"意图识别失败：{message}"],
    }

def _cn_number_to_int(value: str) -> int | None:
    """把“三 / 十二 / 二十四 / 4”转成整数；失败返回 None。"""
    value = value.strip()
    if not value:
        return None
    if value.isdigit():
        return int(value)
    if value == "十":
        return 10
    if "十" in value:
        left, _, right = value.partition("十")
        tens = _CN_NUM.get(left, 1) if left else 1
        ones = _CN_NUM.get(right, 0) if right else 0
        return tens * 10 + ones
    if len(value) == 1:
        return _CN_NUM.get(value)
    return None

def _parse_story_id(text: str) -> str:
    """从用户原文中提取显式 story_id，不依赖模型输出。"""
    patterns = [
        r"story[_-]?id\s*[=:：]\s*([A-Za-z0-9_\-\u4e00-\u9fff]+)",
        r"故事\s*id\s*[=:：]\s*([A-Za-z0-9_\-\u4e00-\u9fff]+)",
        r"故事id\s*是\s*([A-Za-z0-9_\-\u4e00-\u9fff]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return ""

def _parse_target_episode(text: str) -> int | None:
    """从“第三夜 / 第4集 / 第12话”中解析目标集数。"""
    match = re.search(
        r"第\s*([0-9零一二两三四五六七八九十百]+)\s*[夜集话回章]",
        text,
    )
    if not match:
        return None
    number = _cn_number_to_int(match.group(1))
    if number is None or number <= 0:
        return None
    return number

def _normalize_yes_no(answer: Any) -> bool:
    """确认节点只接受明确的肯定；其他一律视为取消。"""
    text = str(answer or "").strip().lower()
    return text in {"y", "yes", "确认", "是", "对", "继续", "1", "true"}

def intent_node(state: ComicState) -> dict[str, Any]:
    """
    图的开始：DeepSeek 只输出 intent / confidence / reason 三个字段。

    置信度路由：
    - confidence >= 0.85：放行；
    - 0.60 < confidence < 0.85：标记 needs_confirmation，去 confirm_node；
    - confidence <= 0.60：直接 unknown，去 error_node。
    """
    text = (state.get("raw_text") or "").strip()
    if not text:
        return _unknown_intent_state("输入为空，请粘贴小说/剧本，或输入续写/修改指令。")

    known_story_ids: list[str] = []
    try:
        known_story_ids = get_story_memory().list_story_ids()
    except Exception as exc:
        print(f"[意图识别警告] 读取故事记忆失败：{exc}", flush=True)

    _print_stage(
        "意图识别",
        f"输入 {len(text)} 字｜已有故事ID：{'、'.join(known_story_ids) if known_story_ids else '无'}",
    )

    try:
        result = invoke_structured(
            node_name="意图识别",
            schema=IntentResult,
            system_prompt=INTENT_SYSTEM,
            user_prompt=build_intent_prompt(text, known_story_ids),
            temperature=INTENT_TEMPERATURE,
        )
    except Exception as exc:
        print(f"[意图识别失败] {type(exc).__name__}: {exc}", flush=True)
        return _unknown_intent_state(
            INTENT_HELP,
            error=f"{type(exc).__name__}: {exc}",
            trace=f"意图识别调用/解析失败：{exc}",
        )

    intent = result.intent
    confidence = float(result.confidence)
    reason = result.reason

    # 低置信度直接判 unknown，绝不进入流水线。
    if intent == "unknown":
        print(f"[意图识别] unknown｜置信度={confidence:.2f}｜{reason}", flush=True)
        return _unknown_intent_state(INTENT_HELP, error=reason)

    if confidence <= INTENT_LOW_CONFIDENCE:
        print(
            f"[意图识别] 低置信 {confidence:.2f} <= {INTENT_LOW_CONFIDENCE}，强制 unknown。",
            flush=True,
        )
        return _unknown_intent_state(
            "意图识别置信度过低，无法安全选择流水线。\n"
            f"模型判断：{intent}（置信度 {confidence:.2f}）\n"
            "请用更明确的指令，例如：\n"
            "  - “续写第四夜”\n"
            "  - “把第一集结局改掉”\n"
            "  - “调整第7个镜头的台词”",
            error=f"low confidence: {confidence:.2f}",
            trace=f"意图识别低置信：{intent} {confidence:.2f}",
        )

    # story_id / 目标集数由代码解析，模型只负责 intent/confidence/reason。
    state_story_id = str(state.get("story_id") or "").strip()
    parsed_story_id = _parse_story_id(text)
    if intent == "new_script":
        story_id = state_story_id or parsed_story_id or "default"
    else:
        story_id = state_story_id or parsed_story_id
        if not story_id:
            if len(known_story_ids) == 1:
                story_id = known_story_ids[0]
            elif len(known_story_ids) > 1:
                return _unknown_intent_state(
                    "检测到多个故事，无法确定要操作哪一个。\n"
                    f"已有故事ID：{'、'.join(known_story_ids)}\n"
                    "请用 --story-id 指定，例如 --story-id demo_story。",
                    error="multiple story ids",
                )
            else:
                story_id = "default"

    target_episode = _parse_target_episode(text)
    instruction = text if intent in {"continue_story", "modify_script"} else ""
    if intent in {"continue_story", "modify_script"} and not instruction:
        return _unknown_intent_state(INTENT_HELP, error="续写/修改指令为空")

    needs_confirmation = INTENT_LOW_CONFIDENCE < confidence < INTENT_HIGH_CONFIDENCE
    confirmation_question = ""
    if needs_confirmation:
        action = "续写" if intent == "continue_story" else "局部修改"
        confirmation_question = (
            f"我理解你想对 story_id={story_id} 执行【{action}】"
            f"（目标集：{target_episode or '自动'}，置信度 {confidence:.2f}）。\n"
            f"原始指令：{text}\n"
            "请确认：回复 y 继续，回复 n 取消。"
        )

    print(
        f"[意图识别] {intent}｜story_id={story_id}"
        f"｜目标集={target_episode or '自动'}"
        f"｜置信度={confidence:.2f}｜{reason}"
        f"｜{'需二次确认' if needs_confirmation else '高置信放行'}",
        flush=True,
    )

    return {
        "user_query": text,
        "intent_type": intent,
        "intent_confidence": confidence,
        "intent_reason": reason,
        "intent_error": "",
        "story_id": story_id,
        "target_episode": target_episode,
        "user_instruction": instruction,
        "pending_intent": intent if needs_confirmation else "",
        "pending_confidence": confidence if needs_confirmation else 0.0,
        "needs_confirmation": needs_confirmation,
        "confirmed": not needs_confirmation,
        "confirmation_question": confirmation_question,
        "trace": [
            f"意图识别：{intent}"
            f"（story_id={story_id}，目标集={target_episode or '自动'}，"
            f"置信度 {confidence:.2f}，"
            f"{'待确认 → confirm_node' if needs_confirmation else '高置信放行'}）"
        ],
    }

def confirm_node(state: ComicState) -> dict[str, Any]:
    """
    中等置信度二次确认。

    用 LangGraph interrupt 暂停图，main.py 通过 Command(resume=用户回答) 恢复。
    只有明确的 y/yes/确认/是 才放行，其余全部取消并进入 error_node。
    """
    intent = str(state.get("intent_type") or "unknown")
    user_query = str(state.get("user_query") or state.get("raw_text") or "")
    confidence = float(state.get("intent_confidence") or 0)
    question = state.get("confirmation_question") or (
        f"我理解你的意图是 {intent}，请确认。回复 y 继续，回复 n 取消。"
    )
    _print_stage(
        "意图二次确认",
        f"intent={intent}｜confidence={confidence:.2f}｜user_query={user_query[:40]}",
    )

    answer = interrupt(
        {
            "question": question,
            "intent": intent,
            "confidence": confidence,
            "user_query": user_query,
            "raw_text": state.get("raw_text") or "",
        }
    )
    answer_text = str(answer or "").strip()

    if _normalize_yes_no(answer_text):
        print("[确认] 用户确认放行。", flush=True)
        return {
            "confirmed": True,
            "needs_confirmation": False,
            "confirm_answer": answer_text,
            "intent_confidence": 1.0,
            "trace": [
                f"二次确认：用户回复 {answer_text!r}，确认执行 {intent} → 对应业务分支"
            ],
        }

    print("[确认] 用户取消。", flush=True)
    return {
        "confirmed": False,
        "needs_confirmation": False,
        "confirm_answer": answer_text,
        "intent_type": "unknown",
        "error_message": "用户取消了本次操作。请重新输入更明确的续写或修改指令。",
        "trace": [
            f"二次确认：用户回复 {answer_text!r}，取消 {intent} → error_node"
        ],
    }

def error_node(state: ComicState) -> dict[str, Any]:
    """意图/历史检索失败时的兜底节点。"""
    message = state.get("error_message") or INTENT_HELP
    print(f"\n{SEPARATOR}")
    print("【错误提示】")
    print("-" * 72)
    print(message, flush=True)
    return {
        "error_message": message,
        "trace": [f"错误节点：{message.splitlines()[0]}"],
    }

