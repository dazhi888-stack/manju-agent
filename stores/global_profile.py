"""冷启动全局角色/世界观档案：本地 JSON 优先，只抽取一次。"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import config
from stores.json_store import read_json, write_json
from nodes.shared import invoke_structured
from state import GlobalProfileDraft
from prompts.profile import GLOBAL_PROFILE_SYSTEM, build_global_profile_prompt


def _safe_name(value: str) -> str:
    name = re.sub(r'[\\/:*?"<>|\r\n\t]', "_", str(value or "")).strip(" ._")
    return name[:60] or "default"


def profile_path(story_id: str) -> Path:
    return config.DATA_DIR / "stories" / _safe_name(story_id) / "global_profile.json"


# 这些只属于单镜头临时状态，不能进入全局角色档案。
_TEMP_MARKERS = (
    "湿", "血", "污渍", "泥", "雨水", "汗", "泪",
    "受伤", "伤口", "包扎", "绷带", "临时", "此刻", "当前",
    "表情", "蓑衣", "斗笠", "淌水", "灰尘", "脏",
    "紧张", "惊讶", "愤怒", "冷笑", "害怕", "疲惫",
    "带笑", "笑意", "当场", "犹豫后", "交出",
    "出手", "刀绳", "不伤人",
)


def _clean_permanent_text(text: str) -> str:
    """按分句删除临时状态；永久旧疤/无鞘短剑等保留。"""
    value = str(text or "")
    # 只删除“含临时状态的括号补充”，保留“谨慎（长期）→ 谨慎”。
    value = re.sub(
        r"[（(]([^）)]*)[）)]",
        lambda match: "" if any(
            marker in match.group(1) for marker in _TEMP_MARKERS
        ) else match.group(0),
        value,
    )
    clauses = re.split(r"[，,；;、\n]+", value)
    kept = [
        clause.strip()
        for clause in clauses
        if clause.strip() and not any(marker in clause for marker in _TEMP_MARKERS)
    ]
    return "，".join(kept)


def sanitize_profile(profile: dict[str, Any] | None) -> dict[str, Any] | None:
    """读取/保存时兜底清洗，防止旧档案把临时状态带进提示词。"""
    if not profile:
        return profile
    cleaned = dict(profile)
    characters: list[dict[str, Any]] = []
    for raw in profile.get("characters") or []:
        if not isinstance(raw, dict):
            continue
        card = dict(raw)
        for field in ("appearance", "personality", "background", "relationships"):
            card[field] = _clean_permanent_text(str(card.get(field) or ""))
        characters.append(card)
    cleaned["characters"] = characters
    return cleaned


def load_global_profile(story_id: str) -> dict[str, Any] | None:
    return sanitize_profile(read_json(profile_path(story_id), default=None))


def global_profile_cards(profile: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not profile:
        return []
    cards: list[dict[str, Any]] = []
    for raw in profile.get("characters") or []:
        if not isinstance(raw, dict) or not raw.get("name"):
            continue
        cards.append(
            {
                "name": str(raw.get("name") or ""),
                "aliases": list(raw.get("aliases") or []),
                "role": str(raw.get("role") or ""),
                "gender": str(raw.get("gender") or "未说明"),
                "age": str(raw.get("age") or "未说明"),
                "appearance": str(raw.get("appearance") or ""),
                "personality": str(raw.get("personality") or ""),
                "speech_style": str(raw.get("speech_style") or ""),
                "background": str(raw.get("background") or ""),
                "relationships": str(raw.get("relationships") or ""),
                "taboos": str(raw.get("taboos") or ""),
            }
        )
    return cards


def ensure_global_profile(
    story_id: str,
    raw_text: str,
    title: str = "",
    *,
    force: bool = False,
) -> tuple[dict[str, Any], Path, bool]:
    """
    返回 (profile, path, cached)。

    文件存在且非 force：直接读本地 JSON，不调用 DeepSeek。
    文件不存在：调用 DeepSeek 抽取一次并保存。
    """
    path = profile_path(story_id)
    if path.exists() and not force:
        existing = sanitize_profile(read_json(path, default={}) or {})
        if existing.get("characters"):
            return existing, path, True

    draft: GlobalProfileDraft = invoke_structured(
        node_name="全局角色/世界观抽取",
        schema=GlobalProfileDraft,
        system_prompt=GLOBAL_PROFILE_SYSTEM,
        user_prompt=build_global_profile_prompt(
            raw_text=str(raw_text or "")[:12000],
            title=title,
        ),
        temperature=0.2,
    )
    data = {
        "story_id": story_id,
        "title": title,
        "worldview": draft.worldview,
        "characters": [card.model_dump() for card in draft.characters],
        "version": 1,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
    }
    data = sanitize_profile(data) or data
    write_json(path, data)
    return data, path, False
