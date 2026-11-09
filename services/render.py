"""
Markdown 渲染层。

为什么单独放一个文件？
- LLM 只负责“内容”（谁是说话人、画面是什么），
- 代码负责“格式”（标签、表格列、序号、转义）。

这样即使模型偶尔不听话，最终输出的剧本标签和分镜表列名也不会漂移。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from state import ScriptDraft, StoryboardDraft


# ===========================================================================
# 一、通用工具
# ===========================================================================
def escape_table_cell(text: Any) -> str:
    """
    把任意文本变成安全的 Markdown 表格单元格。

    - 竖线 | 必须转义，否则会把一列切成两列；
    - 单元格内不能有真实换行，用 <br> 代替；
    - 空内容统一显示为“—”。
    """
    value = "" if text is None else str(text)
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = value.replace("|", r"\|")
    value = value.replace("\n", "<br>")
    value = value.strip()
    return value or "—"


def normalize_duration(seconds: float) -> tuple[str, float, bool]:
    """
    统一时长格式。

    返回：(显示文本, 采用的秒数, 是否被上下限修正)
    演示中把镜头限制在 1~12 秒，避免模型偶尔输出 0 秒或 100 秒。
    """
    raw = float(seconds)
    normalized = max(1.0, min(12.0, raw))
    normalized = round(normalized, 1)
    clamped = abs(normalized - raw) > 1e-6

    if float(normalized).is_integer():
        text = f"{int(normalized)}秒"
    else:
        text = f"{normalized:g}秒"
    return text, normalized, clamped


def _find_canon(character: dict[str, Any], canon_profiles: list[dict[str, Any]]) -> dict[str, Any] | None:
    """按姓名/别名找对应的设定卡。"""
    names = [character.get("name", "")] + list(character.get("aliases") or [])
    normalized_names = {str(name).replace(" ", "").strip() for name in names if str(name).strip()}
    for card in canon_profiles:
        card_names = [card.get("name", "")] + list(card.get("aliases") or [])
        for name in card_names:
            if str(name).replace(" ", "").strip() in normalized_names:
                return card
    return None


# ===========================================================================
# 二、节点1：剧情摘要
# ===========================================================================
def render_plot_summary(
    plot_data: dict[str, Any],
    canon_profiles: list[dict[str, Any]] | None = None,
    canon_missed: list[str] | None = None,
    truncated: bool = False,
) -> str:
    """把节点1的结构化结果渲染成 Markdown 剧情摘要。"""
    canon_profiles = canon_profiles or []
    canon_missed = canon_missed or []

    title = plot_data.get("title", "未命名章节")
    lines: list[str] = [f"# 剧情摘要｜《{title}》", ""]

    if truncated:
        lines.append("> ⚠️ 原文超过长度上限，本次只解析了截断后的内容。")
        lines.append("")

    lines.extend(
        [
            "## 一句话梗概",
            str(plot_data.get("one_line_summary", "原文未说明")),
            "",
            "## 主线剧情",
            str(plot_data.get("main_plot", "原文未说明")),
            "",
            "## 出场人物",
        ]
    )

    characters = plot_data.get("characters") or []
    if not characters:
        lines.append("- 原文未识别出明确人物。")
    for character in characters:
        lines.append(
            f"- **{character.get('name', '未命名')}**"
            f"（{character.get('role', '身份未说明')}）"
        )
        lines.append(f"  - 性格：{character.get('personality', '原文未说明')}")
        lines.append(f"  - 外貌：{character.get('appearance', '原文未说明')}")
        lines.append(f"  - 关系：{character.get('relationships', '原文未说明')}")
        actions = character.get("key_actions") or []
        if actions:
            lines.append(f"  - 关键动作：{'；'.join(str(item) for item in actions)}")

        card = _find_canon(character, canon_profiles)
        if card:
            lines.append(
                "  - 设定库校准：已使用 Chroma 人物卡覆盖/补全性格与说话风格"
                f"（匹配方式：{card.get('match_type', 'unknown')}）"
            )
            if card.get("personality"):
                lines.append(f"    - 设定性格：{card.get('personality')}")
            if card.get("speech_style"):
                lines.append(f"    - 设定说话风格：{card.get('speech_style')}")
            taboos = card.get("taboos") or []
            if taboos:
                lines.append(f"    - 设定禁忌：{'；'.join(str(item) for item in taboos)}")
        else:
            lines.append("  - 设定库：未命中，按原文推断")
    lines.append("")

    lines.append("## 场景")
    scenes = plot_data.get("scenes") or []
    if not scenes:
        lines.append("- 原文未识别出明确场景。")
    for index, scene in enumerate(scenes, start=1):
        lines.append(
            f"{index}. **{scene.get('location', '地点未说明')}**"
            f"（{scene.get('time', '时间未说明')}）：{scene.get('summary', '')}"
        )
        if scene.get("atmosphere"):
            lines.append(f"   - 氛围：{scene.get('atmosphere')}")
    lines.append("")

    lines.extend(
        [
            "## 情绪基调",
            str(plot_data.get("emotional_tone", "原文未说明")),
            "",
            "## 关键事件（按顺序）",
        ]
    )
    key_events = plot_data.get("key_events") or []
    if not key_events:
        lines.append("- 原文未提取到明确的关键事件。")
    for index, event in enumerate(key_events, start=1):
        lines.append(f"{index}. {event}")
    lines.append("")

    lines.append("## 不可改动的事实锚点")
    anchors = plot_data.get("fidelity_anchors") or []
    if not anchors:
        lines.append("- 原文未提取到明确锚点。")
    for anchor in anchors:
        lines.append(f"- {anchor}")
    lines.append("")

    lines.append("## 未解决的悬念")
    threads = plot_data.get("unresolved_threads") or []
    if not threads:
        lines.append("- 无。")
    for thread in threads:
        lines.append(f"- {thread}")
    lines.append("")

    lines.append("## 人物设定库命中情况")
    if canon_profiles:
        for card in canon_profiles:
            lines.append(
                f"- 命中：**{card.get('name', '未命名')}**"
                f"（{card.get('match_type', 'unknown')}）"
            )
    else:
        lines.append("- 没有命中任何已有设定。")
    if canon_missed:
        lines.append(f"- 未命中：{'、'.join(canon_missed)}（按原文推断）")
    lines.append("")

    return "\n".join(lines)


# ===========================================================================
# 三、节点2：漫剧剧本
# ===========================================================================
def render_script(draft: ScriptDraft) -> str:
    """把结构化剧本渲染成带【旁白】【台词】【动作描述】标签的 Markdown。"""
    lines: list[str] = [f"# 漫剧剧本｜《{draft.title}》", ""]
    if draft.logline:
        lines.extend([f"> {draft.logline}", ""])

    for index, scene in enumerate(draft.scenes, start=1):
        lines.append(
            f"## 第{index}场｜{scene.location}｜{scene.time}"
        )
        if scene.characters:
            lines.append(f"**出场人物**：{'、'.join(scene.characters)}")
        lines.append("")

        for beat in scene.beats:
            content = beat.content.strip()
            if beat.kind == "台词":
                speaker = (beat.speaker or "未注明").strip()
                lines.append(f"【台词】{speaker}：{content}")
            elif beat.kind == "旁白":
                lines.append(f"【旁白】{content}")
            else:
                lines.append(f"【动作描述】{content}")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


# ===========================================================================
# 四、节点3：分镜表
# ===========================================================================
def render_storyboard(
    draft: StoryboardDraft,
    title: str = "",
    warnings: list[str] | None = None,
) -> str:
    """
    把结构化镜头数组渲染成固定 6 列的 Markdown 表格。

    表格列固定为：
    镜头序号 | 时长 | 画面描述 | 台词/旁白 | 镜头运镜 | 氛围备注
    """
    heading = f"# 分镜表｜《{title}》" if title else "# 分镜表"
    rows: list[str] = []
    total_seconds = 0.0

    for index, shot in enumerate(draft.shots, start=1):
        duration_text, seconds, clamped = normalize_duration(shot.duration_seconds)
        total_seconds += seconds
        if clamped and warnings is not None:
            warnings.append(
                f"镜头{index}时长 {shot.duration_seconds:g} 秒超出 1~12 秒，"
                f"已按 {duration_text} 处理"
            )

        rows.append(
            "| "
            + " | ".join(
                [
                    str(index),
                    duration_text,
                    escape_table_cell(shot.visual),
                    escape_table_cell(shot.dialogue),
                    escape_table_cell(shot.camera),
                    escape_table_cell(shot.mood),
                ]
            )
            + " |"
        )

    total_text = f"{int(round(total_seconds))}秒" if total_seconds else "0秒"
    lines = [
        heading,
        "",
        f"**镜头数**：{len(draft.shots)}｜**预估总时长**：约 {total_text}",
        "",
        "| 镜头序号 | 时长 | 画面描述 | 台词/旁白 | 镜头运镜 | 氛围备注 |",
        "| --- | --- | --- | --- | --- | --- |",
        *rows,
        "",
    ]
    return "\n".join(lines)


# ===========================================================================
# 五、节点4：校验报告
# ===========================================================================
def render_validation(
    validation: dict[str, Any],
    revision_count: int = 0,
    max_revisions: int = 0,
    history: list[dict[str, Any]] | None = None,
) -> str:
    """渲染单轮/多轮校验报告。"""
    passed = bool(validation.get("passed"))
    lines: list[str] = ["## 校验报告", ""]

    if passed:
        lines.append("**结论**：✅ 通过")
    else:
        lines.append("**结论**：❌ 未通过")

    lines.append(
        f"**剧情忠实度**：{'通过' if validation.get('plot_faithful') else '未通过'}"
        f"｜**人物一致性**：{'通过' if validation.get('character_consistent') else '未通过'}"
    )
    lines.append(f"**重写次数**：{revision_count} / {max_revisions}")
    if validation.get("summary"):
        lines.append(f"**总结**：{validation.get('summary')}")
    lines.append("")

    issues = validation.get("issues") or []
    if not issues:
        lines.append("### 问题清单")
        lines.append("- 未发现实质问题。")
    else:
        lines.append("### 问题清单")
        for index, issue in enumerate(issues, start=1):
            lines.append(
                f"{index}. **[{issue.get('category', '未知')}/"
                f"{issue.get('severity', '未知')}]** {issue.get('location', '位置未说明')}"
            )
            lines.append(f"   - 问题：{issue.get('description', '')}")
            lines.append(f"   - 建议：{issue.get('suggestion', '')}")
    lines.append("")

    # 最终报告里展示历次校验，方便判断为什么重写。
    if history and len(history) > 1:
        lines.append("### 历次校验记录")
        for index, item in enumerate(history, start=1):
            status = "通过" if item.get("passed") else "未通过"
            count = len(item.get("issues") or [])
            lines.append(f"- 第{index}轮：{status}（问题 {count} 条）")
        lines.append("")

    return "\n".join(lines)


# ===========================================================================
# 六、续写/修改分支的分集摘要
# ===========================================================================
def render_episode_summary(
    title: str,
    episode_no: int,
    main_plot: str,
    key_events: list[str] | None = None,
    change_summary: str = "",
) -> str:
    """渲染续写/修改分支的新一集摘要（会写入 story_memory）。"""
    lines: list[str] = [
        f"# 剧情摘要｜《{title}》第{episode_no}集",
        "",
        "## 本集主线",
        str(main_plot or "原文未说明"),
        "",
        "## 关键事件",
    ]

    events = key_events or []
    if not events:
        lines.append("- 未提取到明确的关键事件。")
    else:
        for index, event in enumerate(events, start=1):
            lines.append(f"{index}. {event}")
    lines.append("")

    if change_summary:
        lines.extend(["## 修改说明", change_summary, ""])

    return "\n".join(lines)


# ===========================================================================
# 六、续写/修改分支的剧情忠实度报告
# ===========================================================================
def render_branch_validation(
    validation: dict[str, Any],
    revision_count: int = 0,
    max_revisions: int = 0,
) -> str:
    """渲染续写/修改分支的剧情忠实度校验报告。"""
    passed = bool(validation.get("passed"))
    lines: list[str] = [
        "## 剧情忠实度校验（续写/修改分支）",
        "",
        f"**结论**：{'✅ 通过' if passed else '❌ 未通过'}",
        f"**分支重写次数**：{revision_count} / {max_revisions}",
    ]
    if validation.get("summary"):
        lines.append(f"**总结**：{validation.get('summary')}")
    lines.append("")

    issues = validation.get("issues") or []
    lines.append("### 问题清单")
    if not issues:
        lines.append("- 未发现实质问题。")
    else:
        for index, issue in enumerate(issues, start=1):
            lines.append(
                f"{index}. **[{issue.get('category', '未知')}/"
                f"{issue.get('severity', '未知')}]** {issue.get('location', '位置未说明')}"
            )
            lines.append(f"   - 问题：{issue.get('description', '')}")
            lines.append(f"   - 建议：{issue.get('suggestion', '')}")
    lines.append("")
    return "\n".join(lines)


# ===========================================================================
# 七、最终合并文档
# ===========================================================================
def render_final(
    state: dict[str, Any],
    run_id: str = "",
    model_name: str = "",
) -> str:
    """把各分支的结果合并成最终可阅读的 Markdown。"""
    validation = state.get("validation")
    has_validation = isinstance(validation, dict) and bool(validation)
    passed = bool(validation.get("passed")) if has_validation else None
    branch_validation = state.get("branch_validation")
    has_branch_validation = isinstance(branch_validation, dict) and bool(branch_validation)
    branch_passed = bool(branch_validation.get("passed")) if has_branch_validation else None

    title = state.get("title") or "未命名章节"
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    intent_type = str(state.get("intent_type") or "unknown")
    intent_label = {
        "new_script": "新剧本（原有流程）",
        "continue_story": "续写",
        "modify_script": "局部修改",
        "unknown": "未识别",
    }.get(intent_type, intent_type)
    episode_no = state.get("episode_no") or "-"
    story_id = state.get("story_id") or "-"

    if has_validation:
        validation_text = "✅ 通过" if passed else "❌ 未通过"
    elif has_branch_validation:
        validation_text = "分支校验：" + ("✅ 通过" if branch_passed else "❌ 未通过")
    else:
        validation_text = "未执行"

    lines: list[str] = [
        "# AI 漫剧剧本 & 分镜表",
        "",
        f"- **章节/集数**：{title}（第{episode_no}集）",
        f"- **故事 ID**：{story_id}",
        f"- **意图**：{intent_label}",
        f"- **生成时间**：{created_at}",
        f"- **模型**：{model_name or config_model_name()}",
        f"- **运行 ID**：{run_id or '未提供'}",
        f"- **原文长度**：{state.get('input_chars', 0)} 字"
        + ("（已截断）" if state.get("input_truncated") else ""),
        f"- **剧本重写次数**：{state.get('revision_count', 0)} / {state.get('max_revisions', 0)}",
        f"- **前情压缩召回**：{state.get('compressed_source_count', 0)} 集",
        f"- **长期故事圣经**：{state.get('story_bible_chars', 0)} 字",
        f"- **文生图画风**：{state.get('style_desc') or '未设置'}",
        f"- **文生图状态**：{state.get('image_status') or '未执行'}",
        f"- **文生视频状态**：{state.get('video_status') or '预留'}",
        f"- **校验结论**：{validation_text}",
        f"- **故事记忆**：{state.get('persist_status') or '未保存'}",
        "",
    ]

    if has_validation and not passed:
        lines.extend(
            [
                "> ⚠️ 本次结果未通过校验（可能已达到重写上限），以下为最后一版，请人工复核问题清单。",
                "",
            ]
        )
    elif has_branch_validation and not branch_passed:
        lines.extend(
            [
                "> ⚠️ 本次续写/修改未通过剧情忠实度校验，以下为最后一版，请人工复核。",
                "",
            ]
        )

    lines.extend(
        [
            "---",
            "",
            state.get("plot_summary", "# 剧情摘要\n\n（没有输出）"),
            "",
            "---",
            "",
            state.get("script", "# 漫剧剧本\n\n（没有输出）"),
            "",
            "---",
            "",
            state.get("storyboard", "# 分镜表\n\n（没有输出）"),
            "",
            "---",
            "",
            "## 文生图",
            "",
            f"- **画风**：{state.get('style_desc') or '未设置'}",
            f"- **状态**：{state.get('image_status') or '未执行'}",
            f"- **耗时**：{state.get('image_gen_ms', 0)} ms",
            f"- **本地图片**：{state.get('image_path') or '（未生成）'}",
            "",
            "### 正向 Prompt",
            "",
            state.get("image_prompt_positive") or "（未生成）",
            "",
            "### 反向 Prompt",
            "",
            state.get("image_prompt_negative") or "（未生成）",
            "",
            "---",
            "",
            "## 文生视频（预留）",
            "",
            f"- **状态**：{state.get('video_status') or 'reserved'}",
            f"- **视频 prompt**：{state.get('video_prompt') or '（预留）'}",
            f"- **本地视频**：{state.get('video_path') or '（未生成）'}",
            "",
            "---",
            "",
        ]
    )

    if has_validation:
        lines.extend(
            [
                render_validation(
                    validation,
                    revision_count=int(state.get("revision_count", 0)),
                    max_revisions=int(state.get("max_revisions", 0)),
                    history=state.get("validation_history") or [],
                ),
                "",
            ]
        )
    elif has_branch_validation:
        lines.extend(
            [
                render_branch_validation(
                    branch_validation,
                    revision_count=int(state.get("branch_revision_count", 0)),
                    max_revisions=int(state.get("branch_max_revisions", 0)),
                ),
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## 校验报告",
                "",
                "**结论**：未执行。",
                "",
            ]
        )

    lines.extend(
        [
            "---",
            "",
            "## 执行记录",
        ]
    )

    trace = state.get("trace") or []
    if trace:
        lines.extend(f"- {item}" for item in trace)
    else:
        lines.append("- 无")
    lines.append("")

    return "\n".join(lines)


def config_model_name() -> str:
    """延迟读取 config，避免 render 在测试环境导入不必要的依赖。"""
    try:
        from config import MODEL_NAME

        return MODEL_NAME
    except Exception:
        return "unknown"
