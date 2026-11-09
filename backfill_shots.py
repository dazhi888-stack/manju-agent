"""一次性回填旧故事的 global_profile.json 和 shots/ep_XX.json。

只读已有 Markdown/视频提示词，不调用 LLM，适合修复历史数据。
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import config
from stores.global_profile import profile_path, sanitize_profile
from stores.json_store import write_json
from services.output_importer import parse_output_markdown
from services.shot_prompt import parse_storyboard_shots
from stores.shot_store import load_episode_shots, save_episode_shots
from stores.vector_store import get_story_memory


def _from_output_files(story_id: str, episode_no: int) -> dict[str, Any] | None:
    for path in config.OUTPUT_DIR.rglob("*.md"):
        if path.name == "latest_result.md":
            continue
        try:
            data = parse_output_markdown(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if (
            data
            and str(data.get("story_id") or "") == story_id
            and int(data.get("episode_no") or 0) == episode_no
        ):
            return data
    return None


def _from_video_prompt(story_id: str, episode_no: int) -> dict[str, Any] | None:
    path = config.OUTPUT_DIR / story_id / f"第{episode_no}集_video_prompt.txt"
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    rows = parse_storyboard_shots(text)
    if not rows:
        return None
    return {"storyboard": text, "title": f"第{episode_no}集", "characters": []}


def _tone(shots: list[dict[str, Any]]) -> str:
    moods = [str(item.get("mood") or "").strip() for item in shots]
    moods = [item for item in moods if item]
    return "、".join(dict.fromkeys(moods[:3])) or "剧情基调"


def _duration(value: str) -> float:
    match = re.search(r"([\d.]+)", str(value))
    return float(match.group(1)) if match else 4.0


def backfill_story(story_id: str) -> dict[str, Any]:
    memory = get_story_memory()
    latest = memory.get_latest_episode_no(story_id)
    results = []
    all_characters: list[dict[str, Any]] = []
    for episode_no in range(1, latest + 1):
        source = _from_output_files(story_id, episode_no)
        if source is None:
            source = _from_video_prompt(story_id, episode_no)
        if source is None:
            results.append({"episode_no": episode_no, "status": "no_source"})
            continue
        if source.get("characters"):
            all_characters = source["characters"]
        rows = parse_storyboard_shots(source.get("storyboard", ""))
        if not rows:
            results.append({"episode_no": episode_no, "status": "no_storyboard"})
            continue
        existing = load_episode_shots(story_id, episode_no)
        if existing and existing.get("shots"):
            results.append({"episode_no": episode_no, "status": "cached"})
            continue
        char_names = [str(item.get("name")) for item in (source.get("characters") or [])]
        shots = []
        for index, row in enumerate(rows, start=1):
            visual = row.get("visual", "")
            shots.append(
                {
                    "shot_no": index,
                    "duration_seconds": _duration(row.get("duration", "")),
                    "visual": visual,
                    "dialogue": row.get("dialogue", ""),
                    "camera": row.get("camera", ""),
                    "mood": row.get("mood", ""),
                    "characters": [name for name in char_names if name in visual],
                }
            )
        save_episode_shots(
            story_id=story_id,
            episode_no=episode_no,
            title=str(source.get("title") or f"第{episode_no}集"),
            episode_tone=_tone(shots),
            shots=shots,
        )
        results.append(
            {"episode_no": episode_no, "status": "saved", "shots": len(shots)}
        )

    # 旧故事没有 global_profile 时，用已有角色卡/设定做一个本地档案。
    path = profile_path(story_id)
    if not path.exists() and all_characters:
        write_json(
            path,
            sanitize_profile(
                {
                    "story_id": story_id,
                    "title": str(memory.get_summaries(story_id)[0]["metadata"].get("title") or ""),
                    "worldview": memory.get_story_bible(story_id)[:2000] or "根据已有故事回填。",
                    "characters": all_characters,
                    "version": 1,
                    "backfilled": True,
                }
            ),
        )
    return {"story_id": story_id, "latest": latest, "results": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--story-id", default="demo_story")
    args = parser.parse_args()
    print(backfill_story(args.story_id))
