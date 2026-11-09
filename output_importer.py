"""
从 outputs/ 下已经生成的 Markdown 结果中恢复故事记忆。

用途：
- 以前跑过 new_script / continue / modify，但 story_memory 里没有对应章节；
- 不想重新调用 LLM 完整重跑，只需要把已有结果补进 Chroma；
- 支持单文件或整个 outputs 目录。

每个 md 文件需要包含现有最终报告格式中的：
- `- **故事 ID**：...`
- `- **章节/集数**：...（第N集）`
- `# 剧情摘要...`
- `# 漫剧剧本...`
- `# 分镜表...`
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from config import OUTPUT_DIR, PROJECT_DIR, configure_console
from vector_store import get_character_library, get_story_memory


def _section(text: str, heading_prefix: str) -> str:
    """截取一个一级标题区块；优先到 --- 分隔线，没有就截到下一个一级标题。"""
    pattern = re.compile(
        rf"^{re.escape(heading_prefix)}.*?(?=^---\s*$|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(text)
    if not match:
        return ""
    return match.group(0).strip()


def _metadata(text: str, label: str) -> str:
    pattern = re.compile(rf"^- \*\*{re.escape(label)}\*\*：(.+)$", re.MULTILINE)
    match = pattern.search(text)
    return match.group(1).strip() if match else ""


def _parse_episode(text: str) -> int:
    match = re.search(r"第\s*(\d+)\s*集", text)
    if not match:
        return 0
    return int(match.group(1))


def _parse_script_names(script: str) -> list[str]:
    """从剧本的“出场人物”和台词 speaker 中提取角色名。"""
    names: set[str] = set()
    for line in script.splitlines():
        match = re.match(r"^\*\*出场人物\*\*：(.+)$", line.strip())
        if match:
            for name in re.split(r"[、,，\s]+", match.group(1)):
                if name.strip():
                    names.add(name.strip())
    for match in re.finditer(r"【台词】([^：:\n]+)[：:]", script):
        name = match.group(1).strip()
        if name:
            names.add(name)
    return sorted(names)


def _merge_with_global_canon(characters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """用全局人物库补全说话风格/禁忌/台词示例，不覆盖已解析字段。"""
    if not characters:
        return characters
    names = [item["name"] for item in characters]
    try:
        matched, _ = get_character_library().match_characters(names)
    except Exception:
        matched = []
    by_name = {str(card.get("name")): card for card in matched}
    for character in characters:
        canon = by_name.get(character["name"])
        if not canon:
            continue
        for key in ("speech_style", "background", "taboos", "examples", "aliases"):
            if not character.get(key) and canon.get(key):
                character[key] = canon[key]
    return characters


def _parse_characters(summary: str) -> list[dict[str, Any]]:
    """从摘要的“出场人物”段落恢复简化角色卡。"""
    section_match = re.search(
        r"^## 出场人物\s*(.*?)(?=^## 场景|\Z)",
        summary,
        flags=re.MULTILINE | re.DOTALL,
    )
    if not section_match:
        return []

    section = section_match.group(1)
    blocks = re.split(r"\n(?=- \*\*)", section)
    characters: list[dict[str, Any]] = []

    for block in blocks:
        name_match = re.match(r"- \*\*(.+?)\*\*（(.+?)）", block.strip())
        if not name_match:
            continue
        name = name_match.group(1).strip()
        role = name_match.group(2).strip()

        def field(label: str) -> str:
            match = re.search(rf"^\s*-\s*{re.escape(label)}：(.+)$", block, re.MULTILINE)
            return match.group(1).strip() if match else "原文未说明"

        characters.append(
            {
                "name": name,
                "role": role,
                "personality": field("性格"),
                "appearance": field("外貌"),
                "relationships": field("关系"),
            }
        )

    return characters


def parse_output_markdown(text: str) -> dict[str, Any] | None:
    """解析一份最终 Markdown，返回可写入 story_memory 的结构。"""
    summary = _section(text, "# 剧情摘要")
    script = _section(text, "# 漫剧剧本")
    storyboard = _section(text, "# 分镜表")
    if not summary or not script or not storyboard:
        return None

    story_id = _metadata(text, "故事 ID")
    chapter_text = _metadata(text, "章节/集数")
    title = re.sub(r"（第\s*\d+\s*集）", "", chapter_text).strip()
    episode_no = _parse_episode(chapter_text) or _parse_episode(text)
    validation_text = _metadata(text, "校验结论")

    characters = _parse_characters(summary)
    known_names = {item["name"] for item in characters}
    for name in _parse_script_names(script):
        if name not in known_names:
            characters.append(
                {
                    "name": name,
                    "role": "出场人物",
                    "personality": "原文未说明",
                    "appearance": "原文未说明",
                    "relationships": "原文未说明",
                }
            )
            known_names.add(name)
    characters = _merge_with_global_canon(characters)

    return {
        "story_id": story_id,
        "title": title,
        "episode_no": episode_no,
        "summary": summary,
        "script": script,
        "storyboard": storyboard,
        "characters": characters,
        "validation_text": validation_text,
    }


def _read_text(path: Path) -> str:
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"无法识别文件编码：{path}")


def import_outputs(
    path: str | Path,
    story_id_override: str = "",
    story_id_filter: str = "",
) -> list[dict[str, Any]]:
    """
    导入单个 md 或整个目录。

    - story_id_filter：只导入该 story_id 的文件（--sync-outputs 用）；
    - story_id_override：强制改写文件里的 story_id（单文件迁移用）；
    - 同一个 (story_id, episode_no) 如果有多份文件，只取修改时间最新的一份；
    - `latest_result.md` 是副本，直接跳过。
    """
    target = Path(path).expanduser()
    if not target.is_absolute() and not target.exists():
        candidate = PROJECT_DIR / target
        if candidate.exists():
            target = candidate
        else:
            target = OUTPUT_DIR / target
    if not target.exists():
        print(f"[导入] 路径不存在：{target}")
        return []

    files = sorted(target.rglob("*.md")) if target.is_dir() else [target]
    parsed: list[dict[str, Any]] = []

    for file_path in files:
        if file_path.name == "latest_result.md":
            continue
        try:
            data = parse_output_markdown(_read_text(file_path))
        except Exception as exc:
            print(f"[导入] 跳过 {file_path.name}：{type(exc).__name__}: {exc}")
            continue
        if not data:
            print(f"[导入] 跳过 {file_path.name}：缺少摘要/剧本/分镜区块")
            continue

        if story_id_filter and str(data.get("story_id") or "") != story_id_filter:
            print(
                f"[导入] 跳过 {file_path.name}：story_id="
                f"{data.get('story_id') or '?'}，不是 --story-id {story_id_filter}"
            )
            continue
        if story_id_override:
            data["story_id"] = story_id_override
        if not data.get("story_id"):
            print(f"[导入] 跳过 {file_path.name}：没有故事 ID，可用 --story-id 指定")
            continue
        if int(data.get("episode_no") or 0) <= 0:
            print(f"[导入] 跳过 {file_path.name}：无法解析集数")
            continue
        if "未通过" in str(data.get("validation_text") or ""):
            print(f"[导入] 跳过 {file_path.name}：校验结论为未通过")
            continue

        data["path"] = file_path
        data["mtime"] = file_path.stat().st_mtime
        parsed.append(data)

    if not parsed:
        print("[导入] 没有可导入的结果。")
        return []

    # 同故事同集只保留最新版
    latest_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for item in parsed:
        key = (str(item["story_id"]), int(item["episode_no"]))
        previous = latest_by_key.get(key)
        if previous is None or item["mtime"] > previous["mtime"]:
            latest_by_key[key] = item

    memory = get_story_memory()
    imported: list[dict[str, Any]] = []

    for (story_id, episode_no), item in sorted(
        latest_by_key.items(), key=lambda pair: (pair[0][0], pair[0][1])
    ):
        memory.replace_episode(
            story_id=story_id,
            episode_no=episode_no,
            title=item["title"],
            plot_summary=item["summary"],
            script=item["script"],
            # 完整分镜只留在 md 文件；向量库只留引用，避免进入 LLM 上下文。
            storyboard=(
                f"[完整分镜见 {Path(item['path']).name}，未写入 LLM 上下文]"
            ),
        )
        for character in item.get("characters") or []:
            memory.upsert_role(story_id, character, episode_no=episode_no)

        # 用户要求的固定确认日志
        print(f"【已追加第 {episode_no} 章摘要到向量库】", flush=True)
        print(
            f"[导入] story_id={story_id}｜第{episode_no}集《{item['title']}》"
            f"｜角色 {len(item.get('characters') or [])} 个"
            f"｜来源 {Path(item['path']).name}",
            flush=True,
        )
        imported.append(item)

    print(f"[导入] 完成：共写入 {len(imported)} 集。", flush=True)
    return imported


def _chapter_core(title: str) -> str:
    """去掉标题前的“第N章/第N夜/第N集”，只保留章节名称。"""
    value = str(title or "").strip()
    value = re.sub(
        r"^第\s*[0-9零一二两三四五六七八九十百]+\s*[章集夜回话]",
        "",
        value,
    )
    value = re.sub(r"^[\s:：|·_.-]+", "", value).strip()
    return value


def _safe_file_part(value: str, fallback: str = "故事") -> str:
    value = re.sub(r'[\\/:*?"<>|\r\n\t]', "_", str(value or "")).strip(" ._")
    value = re.sub(r"\s+", "_", value)
    return value[:50] or fallback


def cleanup_outputs(
    path: str | Path = OUTPUT_DIR,
    only_vector_memory: bool = True,
) -> list[Path]:
    """
    清理重复输出，并统一为：outputs/<story_id>/第N章_章节名称.md。

    同一 story_id + episode_no 只保留最新文件；如果 only_vector_memory=True，
    只保留当前 story_memory 中已经存在的章节。latest_result.md 作为别名保留。
    """
    root = Path(path).expanduser()
    if not root.is_absolute() and not root.exists():
        root = PROJECT_DIR / root
    if not root.exists():
        print(f"[整理] outputs 路径不存在：{root}")
        return []

    files = sorted(root.rglob("*.md"))
    parsed_items: list[dict[str, Any]] = []
    latest_alias: Path | None = None

    for file_path in files:
        if file_path.name == "latest_result.md":
            latest_alias = file_path
        try:
            data = parse_output_markdown(_read_text(file_path))
        except Exception as exc:
            print(f"[整理] 跳过 {file_path.name}：{type(exc).__name__}: {exc}")
            continue
        if not data or not data.get("story_id") or not data.get("episode_no"):
            continue
        data["path"] = file_path
        data["mtime"] = file_path.stat().st_mtime
        data["is_latest_alias"] = file_path.name == "latest_result.md"
        parsed_items.append(data)

    memory = get_story_memory()
    valid_keys: set[tuple[str, int]] = set()
    if only_vector_memory:
        for story_id in memory.list_story_ids():
            for summary in memory.get_summaries(story_id):
                valid_keys.add(
                    (story_id, int(summary["metadata"].get("episode_no", 0)))
                )

    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    story_names: dict[str, str] = {}
    story_name_episodes: dict[str, int] = {}
    for item in parsed_items:
        key = (str(item["story_id"]), int(item["episode_no"]))
        if only_vector_memory and key not in valid_keys:
            if not item["is_latest_alias"]:
                item["path"].unlink(missing_ok=True)
                print(f"[整理] 删除向量记忆中不存在的文件：{item['path'].name}")
            continue
        groups.setdefault(key, []).append(item)
        core = _chapter_core(item.get("title", ""))
        story_id = str(item["story_id"])
        episode_no = int(item["episode_no"])
        previous_episode = story_name_episodes.get(story_id)
        if core and (previous_episode is None or episode_no < previous_episode):
            story_names[story_id] = core
            story_name_episodes[story_id] = episode_no

    kept: list[Path] = []
    for (story_id, episode_no), items in sorted(
        groups.items(), key=lambda pair: (pair[0][0], pair[0][1])
    ):
        chapter_name = ""
        for item in items:
            chapter_name = _chapter_core(item.get("title", ""))
            if chapter_name:
                break
        if not chapter_name:
            chapter_name = story_names.get(story_id, "") or _safe_file_part(story_id)

        latest_item = max(items, key=lambda item: item["mtime"])
        target_dir = root / _safe_file_part(story_id)
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"第{episode_no}章_{_safe_file_part(chapter_name)}.md"

        target.write_text(_read_text(Path(latest_item["path"])), encoding="utf-8")
        for item in items:
            source = Path(item["path"])
            if source.name == "latest_result.md":
                continue
            if source.resolve() != target.resolve() and source.exists():
                source.unlink()
                print(f"[整理] 删除旧版本：{source.name}")

        kept.append(target)
        print(f"[整理] 保留：{target.relative_to(root)}")

    if latest_alias is not None:
        print(f"[整理] 保留别名：{latest_alias.relative_to(root)}")
    print(f"[整理] 完成：保留 {len(kept)} 个规范章节文件。")
    return kept


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="从 outputs 的 Markdown 恢复/整理 story_memory")
    parser.add_argument("--path", type=str, default=str(OUTPUT_DIR), help="md 文件或目录")
    parser.add_argument("--story-id", type=str, default="", help="覆盖文件里的故事 ID")
    parser.add_argument("--cleanup", action="store_true", help="按 story_memory 整理并重命名 outputs")
    return parser


def main() -> int:
    configure_console()
    args = _build_parser().parse_args()
    if args.cleanup:
        cleanup_outputs(args.path)
    else:
        import_outputs(args.path, story_id_override=args.story_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
