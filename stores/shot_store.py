"""按集镜头 JSON 的保存/读取/渲染（不调用 LLM）。"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import config
from stores.json_store import read_json, write_json


def _safe_name(value: str) -> str:
    name = re.sub(r'[\\/:*?"<>|\r\n\t]', "_", str(value or "")).strip(" ._")
    return name[:60] or "default"


def shots_path(story_id: str, episode_no: int) -> Path:
    return (
        config.OUTPUT_DIR
        / _safe_name(story_id)
        / "shots"
        / f"ep_{int(episode_no):02d}.json"
    )


def load_episode_shots(story_id: str, episode_no: int) -> dict[str, Any] | None:
    return read_json(shots_path(story_id, episode_no), default=None)


def save_episode_shots(
    story_id: str,
    episode_no: int,
    title: str,
    episode_tone: str,
    shots: list[dict[str, Any]],
) -> Path:
    data = {
        "story_id": story_id,
        "episode_no": int(episode_no),
        "title": title,
        "episode_tone": episode_tone,
        "shots": shots,
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    return write_json(shots_path(story_id, episode_no), data)


def normalize_shots(shots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """统一 shot_no 为 1..N，并清理 Markdown 围栏/空字段。"""
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(shots, start=1):
        visual = re.sub(r"```.*?```", "", str(raw.get("visual") or ""), flags=re.DOTALL)
        normalized.append(
            {
                "shot_no": index,
                "duration_seconds": float(raw.get("duration_seconds") or 4),
                "visual": visual.strip(),
                "dialogue": str(raw.get("dialogue") or "").strip(),
                "camera": str(raw.get("camera") or "中景固定").strip(),
                "mood": str(raw.get("mood") or "").strip(),
                "characters": [
                    str(item) for item in (raw.get("characters") or []) if str(item).strip()
                ],
            }
        )
    return [item for item in normalized if item["visual"]]


def render_storyboard_markdown(
    shots: list[dict[str, Any]],
    title: str = "",
) -> str:
    lines = [
        f"# 分镜表｜{title or '未命名'}",
        "",
        "| 镜头序号 | 时长 | 画面描述 | 台词/旁白 | 镜头运镜 | 氛围备注 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for shot in shots:
        duration = f"{float(shot.get('duration_seconds') or 4):g}秒"
        row = [
            str(shot.get("shot_no")),
            duration,
            str(shot.get("visual") or "").replace("|", "／"),
            str(shot.get("dialogue") or "—").replace("|", "／") or "—",
            str(shot.get("camera") or "").replace("|", "／"),
            str(shot.get("mood") or "").replace("|", "／"),
        ]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)
