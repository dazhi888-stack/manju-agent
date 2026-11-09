"""
AI 漫剧 Agent Demo 的命令行入口。

用法示例：
    python main.py
    python main.py --story-id demo_story --file data\\sample_chapter.txt
    python main.py --story-id demo_story --text "续写第二夜"
    python main.py --story-id demo_story --text "把第一夜结局改掉"
    python main.py --seed-only
    python main.py --reset-db

第一次使用建议直接回车，使用内置示例章节先跑通链路。
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import os
import re
import sys
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path

# 只做标准库操作；业务模块在依赖预检通过后再导入。
PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"

REQUIRED_MODULES = {
    "langgraph": "langgraph",
    "langchain_core": "langchain-core",
    "langchain_deepseek": "langchain-deepseek",
    "chromadb": "chromadb",
    "dotenv": "python-dotenv",
    "pydantic": "pydantic",
    "numpy": "numpy",
    "openai": "openai",
}


# ===========================================================================
# 一、依赖预检
# ===========================================================================
def configure_console_basic() -> None:
    """
    在导入业务模块之前先把控制台切到 UTF-8。

    这样连“缺少依赖”的提示也能正常显示中文；
    config.configure_console() 后面还会再做一次（幂等）。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            if stream is not None and (getattr(stream, "encoding", "") or "").lower() != "utf-8":
                stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def check_dependencies() -> bool:
    """
    在导入业务模块前检查依赖。

    这样即使 PyCharm 选错了解释器，也能看到“缺什么 + 怎么换环境”，
    而不是一长串 ImportError 堆栈。
    """
    missing = [
        pip_name
        for module_name, pip_name in REQUIRED_MODULES.items()
        if importlib.util.find_spec(module_name) is None
    ]
    if not missing:
        return True

    print("=" * 72)
    print("依赖检查失败：当前 Python 解释器缺少以下包")
    for name in missing:
        print(f"  - {name}")
    print()
    print(f"当前解释器：{sys.executable}")
    print()
    print("推荐直接使用已装齐依赖的环境：")
    print(r"  C:\Users\Administrator\miniconda3\envs\langchain1.2\python.exe main.py")
    print()
    print("或者在 PyCharm 中把 Project Interpreter 切换到：")
    print(r"  C:\Users\Administrator\miniconda3\envs\langchain1.2\python.exe")
    print()
    print("如果要在当前环境安装：")
    print("  pip install -r requirements.txt")
    print("=" * 72)
    return False


# ===========================================================================
# 二、命令行参数
# ===========================================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AI 漫剧 Agent：意图识别 -> 新剧本 / 续写 / 局部修改",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--file", type=str, default="", help="从文件读取小说章节（UTF-8/GB18030）")
    source.add_argument("--text", type=str, default="", help="直接传入章节文本")

    parser.add_argument(
        "--story-id",
        type=str,
        default="",
        help="故事ID，用于续写/修改时检索 story_memory；多个故事时必须指定",
    )
    parser.add_argument(
        "--style",
        type=str,
        default="",
        help="文生图画风，例如 日系漫画/厚涂/写实/赛璐璐/水墨/像素风",
    )
    parser.add_argument(
        "--negative",
        type=str,
        default="",
        help="文生图额外负面词，会追加到基础负面模板",
    )
    parser.add_argument("--menu", action="store_true", help="打开交互式测试菜单")
    parser.add_argument(
        "--max-revisions",
        type=int,
        default=None,
        help="校验不通过时最多重写几次剧本；默认读取 .env（2），0 表示只跑一轮",
    )
    parser.add_argument("--model", type=str, default="", help="临时覆盖模型名，例如 deepseek-chat")
    parser.add_argument("--seed-only", action="store_true", help="只把 data/demo_characters.json 注入人物库")
    parser.add_argument("--reset-db", action="store_true", help="清空人物库并重新注入示例人物（只初始化，不生成剧本）")
    parser.add_argument(
        "--sync-outputs",
        action="store_true",
        help="扫描 outputs/ 已有 md，把剧情摘要/剧本/分镜追加到对应 story_id 的向量记忆",
    )
    parser.add_argument(
        "--import-output",
        type=str,
        default="",
        help="导入单个 md 或目录到 story_memory；不重新跑 LLM",
    )
    parser.add_argument(
        "--cleanup-outputs",
        action="store_true",
        help="只保留向量记忆中存在的章节，并重命名为 第N章_章节名称.md",
    )
    parser.add_argument("--check-langsmith", action="store_true", help="检查 LangSmith 配置和连接")
    parser.add_argument("--debug", action="store_true", help="打印原始结构化输出，便于调提示词")
    parser.add_argument("--no-pause", action="store_true", help="运行结束后不等待回车（适合自动化）")
    return parser


# ===========================================================================
# 三、读取输入
# ===========================================================================
def read_text_file(raw_path: str) -> str:
    """读取文本文件，优先 UTF-8，失败后尝试 GB18030（兼容常见中文 GBK 文件）。"""
    path = Path(raw_path).expanduser()
    if not path.is_absolute() and not path.exists():
        # 允许从工作区根目录执行 python manju_agent/main.py --file data/xxx.txt
        candidate = PROJECT_DIR / path
        if candidate.exists():
            path = candidate

    if not path.exists():
        raise FileNotFoundError(f"找不到文件：{path}")

    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            text = path.read_text(encoding=encoding)
            print(f"[输入] 已读取 {path}（编码：{encoding}，{len(text)} 字）", flush=True)
            return text.strip()
        except UnicodeDecodeError as exc:
            last_error = exc

    raise ValueError(
        f"无法识别文件编码：{path}。请另存为 UTF-8 或 GB18030。"
        f"最后一次错误：{last_error}"
    )


def read_interactive_text() -> tuple[str, str]:
    """
    交互式读取多行文本。

    - 直接回车：使用内置示例；
    - 粘贴多行后，单独一行输入 EOF 结束。
    """
    print("请粘贴小说章节文本。")
    print("  - 直接按回车：使用内置示例 data/sample_chapter.txt")
    print("  - 粘贴多行后，单独一行输入 EOF 回车结束")
    print()

    try:
        first_line = input("> ")
    except EOFError:
        first_line = ""

    if first_line.strip() == "":
        sample_path = PROJECT_DIR / "data" / "sample_chapter.txt"
        return sample_path.read_text(encoding="utf-8").strip(), "内置示例"

    lines = [first_line]
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "EOF":
            break
        lines.append(line)

    text = "\n".join(lines).strip()
    if not text:
        sample_path = PROJECT_DIR / "data" / "sample_chapter.txt"
        return sample_path.read_text(encoding="utf-8").strip(), "内置示例"
    return text, "交互输入"


def get_input_text(args: argparse.Namespace) -> tuple[str, str]:
    """根据参数选择输入来源，返回 (文本, 来源说明)。"""
    if args.text:
        return args.text.strip(), "命令行 --text"
    if args.file:
        return read_text_file(args.file), f"文件 {args.file}"

    # 纯本地管理命令不需要读小说。
    if args.seed_only:
        return "", "无需文本（--seed-only）"

    return read_interactive_text()


# ===========================================================================
# 四、运行前准备
# ===========================================================================
def print_banner(args: argparse.Namespace, config) -> None:
    max_revisions = (
        args.max_revisions
        if args.max_revisions is not None
        else config.MAX_REVISIONS
    )
    max_revisions = max(0, max_revisions)
    key_status = "已配置" if (os.getenv("DEEPSEEK_API_KEY") or "").strip() else "未配置"
    tracing_status = (
        "开启"
        if (os.getenv("LANGSMITH_TRACING") or "").strip().lower()
        in {"1", "true", "yes", "on"}
        else "关闭"
    )
    story_id = args.story_id.strip() or "自动识别/默认"
    print("\n" + "=" * 72)
    print("AI 漫剧 Agent Demo｜小说章节 -> 漫剧剧本 + 分镜表")
    print("=" * 72)
    print("流程：前置剧情压缩 -> 意图识别")
    print("      ├─ new_script     → 原文解析 -> 剧本改写 -> 分镜 -> 校验 -> 入库")
    print("      ├─ continue_story → 检索角色/摘要 -> 续写剧本 -> 分镜 -> 入库")
    print("      ├─ modify_script  → 检索历史分镜 -> 局部修改 -> 分镜 -> 入库")
    print("      └─ unknown        → 错误提示")
    print("意图置信度：>=0.85 放行｜0.60~0.85 二次确认｜<=0.60 直接 unknown")
    print("-" * 72)
    print(f"模型：{config.MODEL_NAME}")
    print(
        f"剧本模型：{config.SCRIPT_MODEL_FREE}"
        f"（429 → {config.SCRIPT_MODEL_PRO}）"
    )
    print(f"API Key：{key_status}")
    print(f"LangSmith：{tracing_status}")
    print(f"故事 ID：{story_id}")
    print(f"文生图画风：{args.style.strip() or config.DEFAULT_STYLE_DESC}")
    print(f"人物库：{config.CHROMA_DIR}")
    print(f"输出目录：{config.OUTPUT_DIR}")
    print(f"最大重写：{max_revisions}（仅 new_script 校验循环）")
    print(f"输入上限：{config.MAX_INPUT_CHARS} 字")
    print("=" * 72 + "\n", flush=True)


def prepare_library(args: argparse.Namespace, config) -> None:
    """处理 --reset-db / --seed-only / 首次自动注入。"""
    from stores.vector_store import get_character_library

    library = get_character_library()

    if args.reset_db:
        library.reset()
        count = library.seed_from_file(config.DEMO_CHARACTERS_PATH)
        print(f"[Chroma] 已重置人物库，并注入示例人物 {count} 条。", flush=True)
    elif args.seed_only:
        count = library.seed_from_file(config.DEMO_CHARACTERS_PATH)
        print(f"[Chroma] 已注入示例人物 {count} 条：{config.DEMO_CHARACTERS_PATH}", flush=True)
    else:
        seeded = library.ensure_seeded()
        if seeded:
            print(f"[Chroma] 人物库为空，已自动注入示例人物 {seeded} 条。", flush=True)


# ===========================================================================
# 五、保存与暂停
# ===========================================================================
def sanitize_title(title: str) -> str:
    """把章节标题变成 Windows 上安全的文件名片段。"""
    value = re.sub(r'[\\/:*?"<>|\r\n\t]', "_", title or "").strip(" ._")
    value = re.sub(r"\s+", "_", value)
    return value[:50] or "chapter"


def _chapter_core(title: str) -> str:
    """去掉“第N章/第N夜/第N集”前缀，只保留章节名称。"""
    value = str(title or "").strip()
    value = re.sub(
        r"^第\s*[0-9零一二两三四五六七八九十百]+\s*[章集夜回话]",
        "",
        value,
    )
    value = re.sub(r"^[\s:：|·_.-]+", "", value).strip()
    return value


def _chapter_name_for_file(state: dict) -> str:
    """优先使用本集标题；只有“第四夜”这种编号时，退回故事第一章的章节名。"""
    core = _chapter_core(str(state.get("title") or ""))
    if core:
        return sanitize_title(core)
    for summary in state.get("history_summaries") or []:
        metadata = summary.get("metadata") if isinstance(summary, dict) else {}
        core = _chapter_core(str((metadata or {}).get("title") or ""))
        if core:
            return sanitize_title(core)
    return sanitize_title(str(state.get("story_id") or "故事"))


def save_result(
    markdown: str,
    state: dict,
    run_id: str,
) -> tuple[Path, Path]:
    """
    保存为 outputs/<story_id>/第N章_章节名称.md。

    不再使用时间戳 + “第四夜”这种双重编号；同一章重复生成会覆盖同名文件。
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    story_id = sanitize_title(str(state.get("story_id") or "default"))
    episode_no = int(state.get("episode_no") or 0)
    chapter_name = _chapter_name_for_file(state)

    if episode_no > 0:
        output_dir = OUTPUT_DIR / story_id
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"第{episode_no}章_{chapter_name}.md"
        path = output_dir / filename
    else:
        # 正常流程不会走到这里；只作为异常兜底，避免覆盖第0章。
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = OUTPUT_DIR / f"{timestamp}_{chapter_name}_{run_id[-6:]}.md"

    with path.open("w", encoding="utf-8", newline="\n") as file:
        file.write(markdown)

    latest = OUTPUT_DIR / "latest_result.md"
    with latest.open("w", encoding="utf-8", newline="\n") as file:
        file.write(markdown)

    return path, latest


def pause_if_needed(args: argparse.Namespace) -> None:
    if args.no_pause:
        return
    try:
        input("\n按回车退出...")
    except EOFError:
        pass


# ===========================================================================
# 六、主流程
# ===========================================================================
def run_demo(args: argparse.Namespace) -> int:
    import config

    config.configure_console()
    config.set_debug(args.debug)
    config.set_model_override(args.model or None)

    # 在导入 graph 之前补齐 LangChain/LangGraph 能识别的 LangSmith 环境变量。
    from services.tracing import configure_langsmith

    configure_langsmith()

    print_banner(args, config)

    if args.max_revisions is not None and args.max_revisions < 0:
        print("[参数错误] --max-revisions 不能小于 0。")
        return 2

    # 纯人物库操作：不调用 LLM，也不需要输入小说。
    # --reset-db 只负责清空并重建人物库，然后退出；需要生成时再去掉该参数运行。
    if args.seed_only or args.reset_db:
        prepare_library(args, config)
        if args.reset_db and not args.seed_only:
            print("[提示] 人物库已重置。本次只做初始化，没有生成剧本；如需生成请重新运行 main.py。", flush=True)
        return 0

    # 从已有 outputs/*.md 恢复 story_memory，不调用 LLM。
    if args.sync_outputs or args.import_output:
        prepare_library(args, config)
        from services.output_importer import import_outputs

        if args.sync_outputs:
            # 注意：--sync-outputs 时 --story-id 只做过滤，不把别的故事改写成当前故事。
            results = import_outputs(
                str(config.OUTPUT_DIR),
                story_id_filter=args.story_id,
            )
        else:
            # 单文件迁移时才允许覆盖 story_id。
            results = import_outputs(
                args.import_output,
                story_id_override=args.story_id,
            )
        print(
            f"[导入] 共处理 {len(results)} 集；"
            "现在可以继续用同一 story_id 续写下一章。",
            flush=True,
        )
        return 0

    if args.check_langsmith:
        from services.tracing import check_langsmith

        return 0 if check_langsmith() else 2

    if args.cleanup_outputs:
        prepare_library(args, config)
        from services.output_importer import cleanup_outputs

        cleanup_outputs(config.OUTPUT_DIR)
        return 0

    text, source = get_input_text(args)
    if not text.strip():
        print("[输入错误] 没有读到小说文本。")
        return 2

    print(f"[输入] 来源：{source}｜长度：{len(text)} 字", flush=True)

    prepare_library(args, config)

    # 延迟导入：确保依赖预检已经通过。
    from graph import graph
    from langgraph.types import Command
    from nodes import NodeError
    from services.render import render_final

    max_revisions = (
        args.max_revisions
        if args.max_revisions is not None
        else config.MAX_REVISIONS
    )
    run_id = f"manju-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
    initial_state = {
        "raw_text": text,
        "user_query": text,
        "story_id": (args.story_id or "").strip(),
        "intent_type": "",
        "pending_intent": "",
        "pending_confidence": 0.0,
        "compressed_context": "",
        "compressed_story_id": "",
        "compressed_source_count": 0,
        "long_term_context": "",
        "story_bible_chars": 0,
        "style_desc": (args.style or config.DEFAULT_STYLE_DESC).strip(),
        "image_negative_extra": (args.negative or "").strip(),
        "image_prompt_positive": "",
        "image_prompt_negative": "",
        "image_path": "",
        "image_status": "",
        "image_gen_ms": 0.0,
        "image_records": [],
        "video_prompt": "",
        "video_path": "",
        "video_status": "",
        "video_gen_ms": 0.0,
        "episode_no": 0,
        "user_instruction": "",
        "target_episode": None,
        "max_revisions": max(0, max_revisions),
        "revision_count": 0,
        "feedback": "",
        "validation_history": [],
        "branch_validation": {},
        "branch_validation_history": [],
        "branch_issue_ledger": [],
        "branch_feedback": "",
        "branch_revision_count": 0,
        "branch_max_revisions": config.MAX_BRANCH_REVISIONS,
        "trace": [],
    }
    run_config = {
        "configurable": {"thread_id": run_id},
        # LangSmith：run_name / tags / metadata 会随整条 LangGraph trace 展示。
        "run_name": "manju-agent",
        "tags": ["manju-agent", "langgraph", "intent-routing"],
        "metadata": {
            "story_id": (args.story_id or "auto"),
            "model": config.MODEL_NAME,
            "source": source,
        },
        # 给原有校验循环 + 分支校验循环都留足余量。
        "recursion_limit": max(40, 15 + max_revisions * 3 + config.MAX_BRANCH_REVISIONS * 4),
    }

    print(f"[运行] thread_id={run_id}", flush=True)
    started = time.perf_counter()

    try:
        final_state = asyncio.run(graph.ainvoke(initial_state, config=run_config))

        # confirm_node 用 LangGraph interrupt 暂停；这里读取中断问题，等待用户二次回复。
        while isinstance(final_state, dict) and "__interrupt__" in final_state:
            interrupts = final_state.get("__interrupt__") or []
            first = interrupts[0] if interrupts else {}
            payload = getattr(first, "value", first)
            if isinstance(payload, dict):
                question = payload.get("question") or "请确认是否继续。"
            else:
                question = str(payload) or "请确认是否继续。"

            print("\n" + "=" * 72)
            print("【需要二次确认】")
            print(question)
            print("=" * 72)
            try:
                answer = input("确认继续？输入 y 继续，n 取消：").strip()
            except EOFError:
                answer = "n"
            print(f"[确认] 用户输入：{answer or 'n'}", flush=True)
            final_state = asyncio.run(
                graph.ainvoke(Command(resume=answer), config=run_config)
            )
    except KeyboardInterrupt:
        print("\n[中断] 已取消本次运行。")
        return 130
    except NodeError as exc:
        print(f"\n[失败] {exc}")
        print("提示：节点已经打印了它的输入/输出；可加 --debug 查看原始模型返回。")
        try:
            partial = graph.get_state(run_config).values
            trace = partial.get("trace") or []
            if trace:
                print("[断点状态] 已完成：")
                for item in trace:
                    print(f"  - {item}")
        except Exception:
            pass
        if args.debug:
            traceback.print_exc()
        return 1
    except Exception as exc:
        print(f"\n[异常] {type(exc).__name__}: {exc}")
        if args.debug:
            traceback.print_exc()
        else:
            print("提示：加 --debug 可打印完整堆栈。")
        return 1

    elapsed = time.perf_counter() - started

    # 意图无法识别 / 历史记录缺失等由 error_node 写入 error_message；
    # 这类情况不生成最终文件，直接给出可操作的提示。
    if final_state.get("error_message"):
        print("\n" + "=" * 72)
        print("[未执行] 流程未通过（意图识别 / 历史检索 / 分支校验），未生成最终文件。")
        print("详细信息见上方【错误提示】。")
        trace = final_state.get("trace") or []
        if trace:
            print("[执行记录]")
            for item in trace:
                print(f"  - {item}")
        print("=" * 72, flush=True)
        return 2

    validation = final_state.get("validation")
    has_validation = isinstance(validation, dict) and bool(validation)
    passed = bool(validation.get("passed")) if has_validation else None
    branch_validation = final_state.get("branch_validation")
    has_branch_validation = isinstance(branch_validation, dict) and bool(branch_validation)
    branch_passed = bool(branch_validation.get("passed")) if has_branch_validation else None

    final_markdown = render_final(
        final_state,
        run_id=run_id,
        model_name=config.MODEL_NAME,
    )

    print("\n" + "=" * 72)
    print("【最终合并结果】")
    print("=" * 72)
    print(final_markdown, flush=True)

    path, latest = save_result(
        final_markdown,
        final_state,
        run_id=run_id,
    )

    print("\n" + "=" * 72)
    print(f"[完成] 总耗时 {elapsed:.1f}s")
    print(
        f"[意图] {final_state.get('intent_type')}｜"
        f"story_id={final_state.get('story_id')}｜"
        f"第{final_state.get('episode_no', 0)}集"
    )
    if has_validation:
        print(
            f"[结果] 校验：{'通过' if passed else '未通过'}｜"
            f"重写次数：{final_state.get('revision_count', 0)}"
        )
    elif has_branch_validation:
        print(
            f"[结果] 剧情忠实度校验：{'通过' if branch_passed else '未通过'}｜"
            f"分支重写次数：{final_state.get('branch_revision_count', 0)}"
        )
    else:
        print("[结果] 校验：未执行")
    print(f"[保存] {path}")
    print(f"[最新] {latest}")
    if has_validation and not passed:
        print("[提醒] 本次未通过校验，最后一版已保存，请查看校验报告中的问题清单。")
    print("=" * 72, flush=True)

    return 0


def main() -> int:
    configure_console_basic()
    parser = build_parser()
    args = parser.parse_args()

    if not check_dependencies():
        return 1

    if args.menu:
        from menu import run_menu

        return run_menu()

    try:
        return_code = run_demo(args)
    except Exception as exc:
        # 兜底：配置/Chroma 初始化等发生在 run_demo 早期的问题。
        print(f"\n[启动失败] {type(exc).__name__}: {exc}")
        if args.debug:
            traceback.print_exc()
        return_code = 1

    pause_if_needed(args)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())


# FastAPI 入口：uvicorn main:app
# 保留上面的 CLI 入口，避免破坏现有 `python main.py --story-id ...` 用法。
try:
    from api import app  # noqa: E402,F401
except Exception:  # pragma: no cover
    app = None
