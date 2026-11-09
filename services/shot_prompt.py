"""从 story_memory + 当前集分镜生成单镜头绘图提示词。"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

import config
from services.output_importer import parse_output_markdown
from stores.shot_store import load_episode_shots, render_storyboard_markdown
from prompts import SHOT_PROMPT_SYSTEM, build_shot_prompt_prompt
from state import ShotPromptDraft
from stores.vector_store import (
    format_character_cards,
    get_character_library,
    get_story_memory,
)


def resolve_active_story_id() -> str:
    """
    菜单 11 自动选当前故事：
    - 显式 ACTIVE_STORY_ID 优先；
    - 只有一个故事直接用；
    - 多个故事时取集数最多的那个（通常是主故事）。
    """
    explicit = str(getattr(config, "ACTIVE_STORY_ID", "") or "").strip()
    memory = get_story_memory()
    story_ids = memory.list_story_ids()
    if explicit and explicit in story_ids:
        return explicit
    if not story_ids:
        raise ValueError("story_memory 里还没有故事，请先跑一集。")
    if len(story_ids) == 1:
        return story_ids[0]
    selected = max(story_ids, key=lambda item: memory.get_latest_episode_no(item))
    print(f"[提示词] 自动选择故事：{selected}（可用 ACTIVE_STORY_ID 指定）")
    return selected


def select_key_shot(
    shots: list[dict[str, Any]],
    characters: list[dict[str, Any]],
) -> dict[str, Any]:
    """自动选一个有人物动作的关键镜头；没有人物时取第二个镜头。"""
    names = [
        str(card.get("name") or "").strip()
        for card in characters
        if str(card.get("name") or "").strip()
    ]
    for shot in shots:
        visual = str(shot.get("visual") or "")
        if any(name in visual for name in names):
            return shot
    if len(shots) > 1:
        return shots[1]
    return shots[0]


def _find_output_file(story_id: str, episode_no: int) -> dict[str, Any] | None:
    """从 outputs 里找该集完整分镜；向量库里的 storyboard 只是文件引用。"""
    candidates: list[tuple[float, dict[str, Any]]] = []
    for path in config.OUTPUT_DIR.rglob("*.md"):
        if path.name == "latest_result.md":
            continue
        try:
            data = parse_output_markdown(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not data:
            continue
        if (
            str(data.get("story_id") or "") == story_id
            and int(data.get("episode_no") or 0) == episode_no
        ):
            candidates.append((path.stat().st_mtime, data))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _merge_roles(
    story_id: str,
    episode_no: int,
    parsed_characters: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    memory = get_story_memory()
    roles: dict[str, dict[str, Any]] = {}
    for card in memory.get_roles(story_id):
        name = str(card.get("name") or "").strip()
        if name:
            roles[name] = dict(card)
    for card in parsed_characters:
        name = str(card.get("name") or "").strip()
        if not name:
            continue
        merged = roles.setdefault(name, {})
        for key, value in card.items():
            if value and str(value) != "原文未说明":
                merged[key] = value
    if roles:
        try:
            matched, _ = get_character_library().match_characters(list(roles))
        except Exception:
            matched = []
        for card in matched:
            name = str(card.get("name") or "").strip()
            if name in roles:
                for key in ("appearance", "personality", "speech_style", "relationships", "taboos"):
                    if card.get(key) and not roles[name].get(key):
                        roles[name][key] = card[key]
    return list(roles.values())


def load_episode_context(
    story_id: str,
    episode_no: int | None = None,
) -> dict[str, Any]:
    """读取本集摘要/剧本/分镜和角色卡；不硬编码任何剧情。"""
    memory = get_story_memory()
    latest = memory.get_latest_episode_no(story_id)
    episode = int(episode_no or latest)
    if episode <= 0:
        raise ValueError(f"story_id={story_id} 没有历史集数。")

    docs = memory.get_episode_docs(story_id, episode)
    output = _find_output_file(story_id, episode)
    stored = load_episode_shots(story_id, episode)
    if output is None and not (stored and stored.get("shots")):
        raise ValueError(
            f"找不到 {story_id} 第{episode}集的完整分镜 md；"
            "请先运行该集或确认 outputs 路径。"
        )

    summary_doc = docs.get("summary") or {}
    summary = (
        str(summary_doc.get("document") or "")
        or str((output or {}).get("summary") or "")
    )
    script = str((docs.get("script") or {}).get("document") or "")
    storyboard = str((output or {}).get("storyboard") or "")
    if not storyboard and stored and stored.get("shots"):
        storyboard = render_storyboard_markdown(
            stored.get("shots") or [],
            str(stored.get("title") or ""),
        )
    if not storyboard:
        raise ValueError(f"{story_id} 第{episode}集分镜为空。")

    characters = _merge_roles(
        story_id,
        episode,
        list((output or {}).get("characters") or []),
    )
    return {
        "story_id": story_id,
        "episode_no": episode,
        "title": str((output or {}).get("title") or (stored or {}).get("title") or ""),
        "summary": summary,
        "script": script,
        "storyboard": storyboard,
        "characters": characters,
        "character_text": format_character_cards(characters),
    }


def parse_storyboard_shots(storyboard: str) -> list[dict[str, Any]]:
    """解析 6 列分镜 Markdown 表格。"""
    rows: list[list[str]] = []
    for line in str(storyboard or "").splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if not cells or "镜头序号" in cells[0]:
            continue
        if all(set(cell) <= {"-", ":", " "} for cell in cells):
            continue
        if re.fullmatch(r"\d+", cells[0] or ""):
            rows.append(cells)

    shots: list[dict[str, Any]] = []
    for cells in rows:
        padded = cells + [""] * (6 - len(cells))
        shots.append(
            {
                "shot_no": int(padded[0]),
                "duration": padded[1],
                "visual": padded[2],
                "dialogue": padded[3],
                "camera": padded[4],
                "mood": padded[5],
                "row": "| " + " | ".join(padded[:6]) + " |",
            }
        )
    return shots


def _shot_context(shots: list[dict[str, Any]], index: int) -> str:
    start = max(0, index - 2)
    end = min(len(shots), index + 3)
    lines = []
    for item in shots[start:end]:
        mark = "→ " if item["shot_no"] == shots[index]["shot_no"] else "  "
        lines.append(
            f"{mark}镜头 #{item['shot_no']}：{item['visual']}"
            f"（氛围：{item['mood']}）"
        )
    return "\n".join(lines)


def _clean_positive_prompt(text: str) -> str:
    """去掉运镜/景别/时长词，只保留画面本身。"""
    value = str(text or "").strip()
    # 近景构图、特写、镜头缓慢推近、时长4秒等都不进入绘图 prompt。
    value = re.sub(
        r"(近景|中景|远景|全景|特写|大特写|镜头[^，。；,;]*)"
        r"(构图)?[，,。；;]?",
        "",
        value,
    )
    value = re.sub(r"\d+\s*(秒|s|S)\b", "", value)
    value = re.sub(r"时长\s*\d*\s*(秒|s|S)?", "", value)
    value = re.sub(r"[，,]\s*[，,]+", "，", value)
    value = re.sub(r"\s{2,}", " ", value).strip(" ：:，,。；;")
    return value


def render_shot_prompt(positive: str, negative: str) -> str:
    """用户要求的精确输出格式。"""
    return f"正向提示词\n{positive.strip()}\n---\n负面提示词\n{negative.strip()}"


async def generate_shot_prompt(
    story_id: str,
    shot_no: int,
    episode_no: int | None = None,
    style_desc: str = "",
    extra_negative: str = "",
) -> dict[str, Any]:
    """
    读取向量库角色卡 + 本集完整分镜 + 指定镜头，生成绘图提示词。

    不硬编码剧本；story_id / episode_no / shot_no 全部来自调用方。
    """
    context = load_episode_context(story_id, episode_no)
    shots = parse_storyboard_shots(context["storyboard"])
    index = next(
        (i for i, item in enumerate(shots) if int(item["shot_no"]) == int(shot_no)),
        None,
    )
    if index is None:
        raise ValueError(
            f"{story_id} 第{context['episode_no']}集没有镜头 #{shot_no}；"
            f"可用镜头：{', '.join(str(item['shot_no']) for item in shots)}"
        )

    style = (
        style_desc.strip()
        if str(style_desc or "").strip()
        else "自动判断：根据本集剧情、角色卡和分镜选择最合适的漫剧画风"
    )
    target = shots[index]
    draft: ShotPromptDraft = await asyncio.to_thread(
        __import__("nodes").invoke_structured,
        node_name="单镜头绘图提示词",
        schema=ShotPromptDraft,
        system_prompt=SHOT_PROMPT_SYSTEM,
        user_prompt=build_shot_prompt_prompt(
            storyboard=context["storyboard"],
            shot_row=target["row"],
            shot_context=_shot_context(shots, index),
            character_cards=context["character_text"],
            style_desc=style,
            episode_summary=context["summary"][:3000],
            extra_negative=extra_negative,
        ),
        temperature=0.3,
    )

    positive = _clean_positive_prompt(draft.positive_prompt)
    negative = draft.negative_prompt.strip()
    if config.IMAGE_NEGATIVE_BASE not in negative:
        negative = f"{config.IMAGE_NEGATIVE_BASE}，{negative}"
    text = render_shot_prompt(positive, negative)

    directory = (
        config.OUTPUT_DIR
        / str(story_id or "default")
        / "prompts"
    )
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"第{context['episode_no']}章_镜头{shot_no}.txt"
    path.write_text(text, encoding="utf-8")

    return {
        "story_id": story_id,
        "episode_no": context["episode_no"],
        "shot_no": int(shot_no),
        "style_desc": style,
        "positive_prompt": positive,
        "negative_prompt": negative,
        "text": text,
        "path": str(path),
    }


async def generate_episode_prompt(
    episode_no: int | None = None,
) -> dict[str, Any]:
    """
    菜单 11 简化入口：不询问 story_id，自动选当前故事，
    自动挑一个有人物动作的关键镜头，风格和负面词由模型按剧情判断。
    """
    story_id = resolve_active_story_id()
    context = load_episode_context(story_id, episode_no)
    shots = parse_storyboard_shots(context["storyboard"])
    if not shots:
        raise ValueError(f"{story_id} 第{context['episode_no']}集没有分镜镜头。")
    key_shot = select_key_shot(shots, context["characters"])
    result = await generate_shot_prompt(
        story_id=story_id,
        shot_no=int(key_shot["shot_no"]),
        episode_no=context["episode_no"],
        style_desc="",
        extra_negative="",
    )
    result["selection_reason"] = "自动选择含人物动作的关键镜头"
    return result
