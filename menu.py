"""交互式测试菜单：不用记长命令，直接选 1/2/3 跑流程。"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import config

STYLE_PRESETS = [
    "日系漫画",
    "厚涂",
    "写实",
    "赛璐璐",
    "水墨",
    "像素风",
]


def _ask(label: str, default: str = "") -> str:
    suffix = f"（默认 {default}）" if default else ""
    print(f"{label}{suffix}：", end="", flush=True)
    try:
        value = input().strip()
    except EOFError:
        value = ""
    print()
    return value or default


def _ask_multiline(label: str) -> str:
    """
    多行输入，单独一行输入 EOF 结束。

    不能用“空行结束”：用户粘贴的文本末尾通常自带空行，
    会把后面的画风/确认输入提前吃掉，导致直接默认取消。
    """
    print(f"{label}（可多行粘贴；单独一行输入 EOF 结束）：")
    lines: list[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip().upper() == "EOF":
            break
        lines.append(line)
    print()
    return "\n".join(lines).strip()


def _choose_style() -> str:
    """用编号选画风，避免直接输入时提示行挤在一起。"""
    print("可选画风：")
    for index, name in enumerate(STYLE_PRESETS, start=1):
        print(f"  {index}. {name}")
    print("  7. 自定义")
    choice = _ask("请选择画风", "1")
    if choice.isdigit():
        index = int(choice)
        if 1 <= index <= len(STYLE_PRESETS):
            return STYLE_PRESETS[index - 1]
        if index == 7:
            return _ask("输入自定义画风", config.DEFAULT_STYLE_DESC)
    return choice or config.DEFAULT_STYLE_DESC


def _base_args() -> argparse.Namespace:
    return argparse.Namespace(
        file="",
        text="",
        story_id="",
        max_revisions=None,
        model="",
        seed_only=False,
        reset_db=False,
        sync_outputs=False,
        import_output="",
        cleanup_outputs=False,
        check_langsmith=False,
        debug=False,
        no_pause=True,
        style="",
        negative="",
        menu=True,
    )


def _run(args: argparse.Namespace) -> int:
    # 延迟导入，避免 main.py 启动时就加载整张 LangGraph。
    from main import run_demo

    return run_demo(args)


def _show_stories() -> None:
    try:
        from vector_store import get_story_memory

        story_ids = get_story_memory().list_story_ids()
    except Exception as exc:
        print(f"[菜单] 读取故事列表失败：{exc}")
        return
    if not story_ids:
        print("[菜单] 暂无故事。")
        return
    print("[菜单] 已有故事：")
    for story_id in story_ids:
        try:
            latest = get_story_memory().get_latest_episode_no(story_id)
        except Exception:
            latest = 0
        print(f"  - {story_id}（最新第{latest}集）")


def _new_script() -> int:
    args = _base_args()
    args.story_id = _ask("story_id", "demo_story")
    path = _ask("章节文件路径（留空使用 data/sample_chapter.txt）")
    if path:
        args.file = path
    else:
        args.file = str(config.SAMPLE_CHAPTER_PATH)
    args.style = _choose_style()
    args.negative = _ask("额外负面词（可留空）")
    return _run(args)


def _continue_story() -> int:
    args = _base_args()
    args.story_id = _ask("story_id", "demo_story")
    args.text = _ask("续写指令", "续写下一章")
    args.style = _choose_style()
    args.negative = _ask("额外负面词（可留空）")
    return _run(args)


def _modify_story() -> int:
    args = _base_args()
    args.story_id = _ask("story_id", "demo_story")
    args.text = _ask("修改指令", "把最后一集结尾改掉")
    args.style = _choose_style()
    args.negative = _ask("额外负面词（可留空）")
    return _run(args)


def _sync_outputs() -> int:
    args = _base_args()
    args.story_id = _ask("只同步哪个 story_id（留空=全部，可用 7 查看）")
    args.sync_outputs = True
    return _run(args)


def _image_from_text() -> int:
    text = _ask_multiline("画面描述/一段文本")
    if not text:
        print("[菜单] 文本为空。")
        return 1
    style = _choose_style()
    negative = _ask("额外负面词（可留空）")
    from image_gen import generate_image_from_text

    result = asyncio.run(
        generate_image_from_text(
            text=text,
            style_desc=style,
            story_id="demo_story",
            negative_extra=negative,
        )
    )
    print(f"[菜单] 文生图状态：{result.get('image_status')}")
    print(f"[菜单] 图片路径：{result.get('image_path') or '（未配置图片 API）'}")
    return 0


def _video_from_text() -> int:
    text = _ask_multiline("视频画面描述/一段文本")
    if not text:
        print("[菜单] 文本为空。")
        return 1
    style = _choose_style()
    print(f"[菜单] 已选画风：{style}")
    confirm = _ask("调用 Seedance 可能耗时/收费，确认生成？y/N", "n").lower()
    if confirm not in {"y", "yes", "是", "确认"}:
        print("[菜单] 已取消视频生成，回到主菜单。")
        return 0
    from video_gen import video_gen_node

    result = asyncio.run(
        video_gen_node(
            {
                "video_prompt": text,
                "image_prompt_positive": text,
                "style_desc": style,
                "story_id": "demo_story",
                "episode_no": 1,
            },
            enable_video=True,
        )
    )
    print(f"[菜单] 文生视频状态：{result.get('status')}")
    print(f"[菜单] {result.get('msg')}")
    if result.get("video_path"):
        print(f"[菜单] 视频路径：{result['video_path']}")
    return 0


def _shot_prompt() -> int:
    episode_text = _ask("第几集（留空=最新）")
    episode_no = int(episode_text) if episode_text.isdigit() else None
    from shot_prompt import (
        generate_shot_prompt,
        load_episode_context,
        parse_storyboard_shots,
        resolve_active_story_id,
    )

    try:
        story_id = resolve_active_story_id()
        context = load_episode_context(story_id, episode_no)
        shots = parse_storyboard_shots(context["storyboard"])
    except Exception as exc:
        print(f"[菜单] 读取分镜失败：{type(exc).__name__}: {exc}")
        return 1

    print(f"\n[菜单] story_id={story_id}｜第{context['episode_no']}集分镜：")
    for shot in shots:
        visual = str(shot.get("visual") or "").replace("\n", " ")
        print(f"  #{shot['shot_no']}  {visual[:70]}")

    shot_text = _ask("选择镜头序号", str(shots[0]["shot_no"]) if shots else "")
    if not shot_text.isdigit():
        print("[菜单] 镜头序号必须是数字。")
        return 1
    try:
        result = asyncio.run(
            generate_shot_prompt(
                story_id=story_id,
                shot_no=int(shot_text),
                episode_no=context["episode_no"],
                style_desc="",
                extra_negative="",
            )
        )
    except Exception as exc:
        print(f"[菜单] 生成提示词失败：{type(exc).__name__}: {exc}")
        return 1
    print(
        f"[菜单] story_id={result['story_id']}｜第{result['episode_no']}集"
        f"｜镜头 #{result['shot_no']}｜画风：{result['style_desc']}"
    )
    print("\n" + result["text"])
    print(f"\n[菜单] 已保存：{result['path']}")
    return 0


def run_menu() -> int:
    config.configure_console()
    while True:
        print("\n" + "=" * 72)
        print("AI 漫剧 Agent 测试菜单")
        print("=" * 72)
        print("1. 新剧本（new_script）")
        print("2. 续写（continue_story）")
        print("3. 修改（modify_script）")
        print("4. 从 outputs 同步故事记忆")
        print("5. 清理/规范化 outputs")
        print("6. 检查 LangSmith")
        print("7. 查看已有故事")
        print("8. 只初始化人物库")
        print("9. 文生图（直接输入一段文本）")
        print("10. 文生视频（Seedance，可能收费）")
        print("11. 选择集数+分镜，生成绘图提示词")
        print("0. 退出")
        choice = _ask("请选择", "0")

        if choice == "0":
            return 0
        if choice == "1":
            _new_script()
        elif choice == "2":
            _continue_story()
        elif choice == "3":
            _modify_story()
        elif choice == "4":
            _sync_outputs()
        elif choice == "5":
            args = _base_args()
            args.cleanup_outputs = True
            _run(args)
        elif choice == "6":
            args = _base_args()
            args.check_langsmith = True
            _run(args)
        elif choice == "7":
            _show_stories()
        elif choice == "8":
            args = _base_args()
            args.seed_only = True
            _run(args)
        elif choice == "9":
            _image_from_text()
        elif choice == "10":
            _video_from_text()
        elif choice == "11":
            _shot_prompt()
        else:
            print("[菜单] 无效选项。")


if __name__ == "__main__":
    raise SystemExit(run_menu())
