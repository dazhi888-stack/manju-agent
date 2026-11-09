"""Web API 最终 Markdown 落盘：保持 outputs/<story_id>/第N章_标题.md 约定。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import config
from render import render_final


def _safe(value: str, fallback: str = "未命名") -> str:
    text = re.sub(r'[\\/:*?"<>|\r\n\t]', "_", str(value or "")).strip(" ._")
    return text[:60] or fallback


def _chapter_name(state: dict[str, Any]) -> str:
    title = str(state.get("title") or "未命名")
    title = re.sub(r"^第[0-9一二三四五六七八九十百零]+[章集]\s*", "", title)
    title = title.strip("《》 ")
    return _safe(title)


def save_state_markdown(state: dict[str, Any]) -> Path | None:
    story_id = _safe(str(state.get("story_id") or "default"))
    episode_no = int(state.get("episode_no") or 0)
    if episode_no <= 0:
        return None
    markdown = render_final(state, model_name=config.MODEL_NAME)
    directory = config.OUTPUT_DIR / story_id
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"第{episode_no}章_{_chapter_name(state)}.md"
    path.write_text(markdown, encoding="utf-8", newline="\n")
    (config.OUTPUT_DIR / "latest_result.md").write_text(
        markdown, encoding="utf-8", newline="\n"
    )
    return path
