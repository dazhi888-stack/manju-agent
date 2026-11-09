"""单集视频提示词拼接：只使用本集指定镜头区间的分镜，不读取角色库。"""

from __future__ import annotations

import re
from typing import Any

import config


def _storyboard_shots(storyboard: str) -> list[dict[str, Any]]:
    """读取 6 列分镜表的镜头序号和画面描述。"""
    shots: list[dict[str, Any]] = []
    for line in str(storyboard or "").splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if not cells or "镜头序号" in cells[0]:
            continue
        if all(set(cell) <= {"-", ":", " "} for cell in cells):
            continue
        if cells[0].isdigit() and len(cells) >= 3:
            visual = cells[2].strip()
            if visual:
                shots.append(
                    {
                        "shot_no": int(cells[0]),
                        "visual": visual,
                        "camera": cells[4].strip() if len(cells) >= 5 else "",
                        "mood": cells[5].strip() if len(cells) >= 6 else "",
                    }
                )
    return shots


def storyboard_shots(storyboard: str) -> list[dict[str, Any]]:
    """公开的镜头列表读取，供 API 返回总镜头数。"""
    return _storyboard_shots(storyboard)


_DIALOGUE_MARKERS = (
    "说话", "台词", "句子", "感叹号", "反问", "流行语", "语气", "口吻",
    "长篇大论", "解释", "交心", "示弱", "出手", "刀绳", "不伤人",
    "带笑", "笑意", "当场", "犹豫后", "交出",
)


def _visual_personality(text: str) -> str:
    """只保留能影响画面的气质/长期行为，不把台词标点规则塞进视频提示词。"""
    clauses = re.split(r"[，,；;、\n]+", str(text or ""))
    kept = [
        clause.strip()
        for clause in clauses
        if clause.strip()
        and not any(marker in clause for marker in _DIALOGUE_MARKERS)
    ]
    return "，".join(kept)


def _character_block(profile: dict[str, Any] | None) -> str:
    """视频提示词只保留影响画面的永久外貌、道具、气质。"""
    if not profile:
        return ""
    characters = profile.get("characters") or []
    lines: list[str] = []
    for card in characters:
        name = str(card.get("name") or "").strip()
        if not name:
            continue
        personality = _visual_personality(str(card.get("personality") or ""))
        parts = [
            f"角色：{card.get('role')}" if card.get("role") else "",
            f"外貌：{card.get('appearance')}" if card.get("appearance") else "",
            f"气质：{personality}" if personality else "",
        ]
        detail = "；".join(part for part in parts if part)
        lines.append(f"- {name}：{detail}" if detail else f"- {name}")
    if not lines:
        return ""
    return "【全局角色档案】\n" + "\n".join(lines)


def build_episode_video_prompt(
    summary: str,
    storyboard: str,
    style_desc: str = "",
    start_shot: int = 1,
    end_shot: int = 3,
    global_profile: dict[str, Any] | None = None,
    reference_block: str = "",
) -> str:
    """
    基于本集指定镜头区间的分镜 + 全局角色档案拼接视频提示词。

    - 只读取 [start_shot, end_shot] 的镜头；
    - 不加载整集主线/关键事件；
    - 人物锚点来自 global_profile.json（本地全局档案），不读角色库。
    """
    style = (style_desc or config.DEFAULT_STYLE_DESC).strip()
    shots = _storyboard_shots(storyboard)
    if not shots:
        raise ValueError("该集暂无分镜数据，请先生成分镜。")

    try:
        start = int(start_shot)
        end = int(end_shot)
    except (TypeError, ValueError) as exc:
        raise ValueError("起始镜头/结束镜头必须是整数") from exc

    total = len(shots)
    if start < 1 or end < start:
        raise ValueError("起始镜头不能大于结束镜头，请重新选择区间")
    if start > total or end > total:
        raise ValueError(f"本集一共{total}个镜头，请重新选择区间")

    selected = [item for item in shots if start <= item["shot_no"] <= end]
    if not selected:
        raise ValueError(f"本集一共{total}个镜头，请重新选择区间")

    shot_lines = "\n".join(
        f"镜头{item['shot_no']}：{item['visual']}"
        f"（运镜：{item.get('camera') or '自然镜头'}；"
        f"氛围：{item.get('mood') or '按画面自然光'}）"
        for item in selected
    )
    character_block = _character_block(global_profile)
    lines = [
        f"{style}，竖屏漫剧视频，统一人物形象与服装，镜头之间动作连贯。",
    ]
    if reference_block:
        lines.append(reference_block)
    if character_block:
        lines.append(character_block)
    lines.extend(
        [
            "【选中镜头】",
            shot_lines,
            (
                "画面要求：保留选中镜头中的场景、时间、天气和关键道具氛围；"
                "电影感光影，细节清晰，画面稳定；不要文字、字幕、水印、logo。"
            ),
        ]
    )
    return "\n".join(lines).strip()
