"""
分镜图入库清单：一张有效图绑定一个 shot_id。

manifest.json 存在 outputs/<story_id>/images/ 下，结构：
{
  "e1_s3": {                       # shot_id = 第1集 镜头3（s0 = 整集）
      "file_name": "第1章_镜头3_写实.png",
      "episode_no": 1,
      "shot_no": 3,
      "style_desc": "写实",
      "confirmed_at": "2026-11-18 16:50:00"
  }
}

规则：
- 每个 shot_id 只保留一张有效图；重新采用会替换旧图并删除旧文件；
- 待定区（_pending）的图不出现在任何镜头绑定里；
- 删除/丢弃图片时同步解绑。
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import config


def _manifest_path(story_id: str) -> Path:
    directory = config.OUTPUT_DIR / _safe_dir(story_id) / "images"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / "manifest.json"


def _safe_dir(story_id: str) -> str:
    value = re.sub(r'[\\/:*?"<>|\r\n\t]', "_", str(story_id or "default")).strip(" ._")
    return value[:40] or "default"


def load_manifest(story_id: str) -> dict[str, dict[str, Any]]:
    """读取 manifest；损坏时返回空（不阻塞主流程）。"""
    path = _manifest_path(story_id)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_manifest(story_id: str, manifest: dict[str, dict[str, Any]]) -> None:
    path = _manifest_path(story_id)
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def bind_shot(
    story_id: str,
    episode_no: int,
    shot_no: int,
    file_name: str,
    style_desc: str = "",
) -> dict[str, Any]:
    """
    把一张确认过的图绑定到 shot_id；该镜头已有绑定时替换并删除旧文件。
    返回绑定后的条目。
    """
    manifest = load_manifest(story_id)
    shot_id = f"e{int(episode_no)}_s{int(shot_no)}"
    images_dir = _manifest_path(story_id).parent

    old = manifest.get(shot_id)
    if old and old.get("file_name") != file_name:
        old_path = images_dir / str(old.get("file_name"))
        if old_path.is_file():
            old_path.unlink()

    entry = {
        "file_name": file_name,
        "episode_no": int(episode_no),
        "shot_no": int(shot_no),
        "style_desc": style_desc,
        "confirmed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    manifest[shot_id] = entry
    save_manifest(story_id, manifest)
    return {"shot_id": shot_id, **entry}


def unbind_file(story_id: str, file_name: str) -> str:
    """图片被删除/丢弃时解绑；返回被解绑的 shot_id（没有则空串）。"""
    manifest = load_manifest(story_id)
    target = str(file_name or "")
    for shot_id, entry in list(manifest.items()):
        if str(entry.get("file_name")) == target:
            del manifest[shot_id]
            save_manifest(story_id, manifest)
            return shot_id
    return ""


def entries_for_episode(
    story_id: str, episode_no: int
) -> dict[int, dict[str, Any]]:
    """返回 {shot_no: entry}，供分镜卡片按镜头展示绑定图。"""
    manifest = load_manifest(story_id)
    result: dict[int, dict[str, Any]] = {}
    for entry in manifest.values():
        if int(entry.get("episode_no") or 0) == int(episode_no):
            result[int(entry.get("shot_no") or 0)] = entry
    return result
