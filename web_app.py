"""Streamlit 演示前端：剧本、分镜/图片、视频、Trace。

启动：
    streamlit run web_app.py
后端：
    uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Any

import requests
import streamlit as st

API_BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
TIMEOUT = 900

st.set_page_config(
    page_title="AI 漫剧 Agent",
    page_icon="🎬",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem;}
    .hero {
        background: linear-gradient(120deg, #0f172a, #1e3a5f);
        color: #f8fafc; padding: 1.2rem 1.5rem; border-radius: 14px;
        margin-bottom: 1rem;
    }
    .hero h1 {margin: 0; font-size: 1.8rem;}
    .hero p {margin: .35rem 0 0; color: #cbd5e1;}
    .status-card {
        padding: .8rem 1rem; border-radius: 10px; margin: .5rem 0;
        border-left: 5px solid #64748b; background: #f8fafc;
    }
    .success-card {border-color: #16a34a; background: #f0fdf4;}
    .warn-card {border-color: #ca8a04; background: #fefce8;}
    .error-card {border-color: #dc2626; background: #fef2f2;}
    .small-muted {color: #64748b; font-size: .85rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>🎬 AI 漫剧 Agent</h1>
      <p>LangGraph · Qwen 剧本 · DeepSeek 校验 · Z-Image 分镜 · Seedance 视频（熔断/降级）</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if "logs" not in st.session_state:
    st.session_state.logs = []
if "last_state" not in st.session_state:
    st.session_state.last_state = {}
if "episode" not in st.session_state:
    st.session_state.episode = {}
if "pending_thread" not in st.session_state:
    st.session_state.pending_thread = ""
if "last_result" not in st.session_state:
    st.session_state.last_result = {}


def _log(message: str) -> None:
    st.session_state.logs.append(
        f"{datetime.now():%H:%M:%S}  {message}"
    )


def _api(
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    quiet: bool = False,
) -> dict[str, Any]:
    url = f"{API_BASE}{path}"
    try:
        if method == "GET":
            response = requests.get(url, timeout=TIMEOUT)
        else:
            response = requests.post(url, json=payload or {}, timeout=TIMEOUT)
        response.raise_for_status()
        data = response.json()
        _log(f"{method} {path} → {data.get('status', 'ok')}")
        return data
    except requests.exceptions.RequestException as exc:
        detail = ""
        response = getattr(exc, "response", None)
        if response is not None:
            try:
                detail = str(response.json().get("detail") or "")
            except Exception:
                detail = response.text[:300] if response.text else ""
        _log(f"{method} {path} ✗ {exc}")
        if not quiet:
            st.error(f"接口调用失败：{detail or exc}")
        return {"error": str(exc), "detail": detail}
    except ValueError as exc:
        if not quiet:
            st.error(f"接口返回非 JSON：{exc}")
        return {"error": str(exc)}


def _apply_run(data: dict[str, Any]) -> None:
    st.session_state.last_result = data
    if data.get("status") == "needs_confirmation":
        st.session_state.pending_thread = data.get("thread_id", "")
        st.session_state.last_state = data.get("state") or {}
        return
    st.session_state.pending_thread = ""
    st.session_state.last_state = data.get("state") or {}


def _confirm(answer: str) -> None:
    thread_id = st.session_state.pending_thread
    if not thread_id:
        return
    data = _api("POST", "/api/script/confirm", {"thread_id": thread_id, "answer": answer})
    if not data.get("error"):
        _apply_run(data)
        st.rerun()


def _show_image(url: str, caption: str = "") -> None:
    if not url:
        return
    full = url if url.startswith("http") else f"{API_BASE}{url}"
    st.image(full, caption=caption, use_container_width=True)


def _show_run_state(state: dict[str, Any]) -> None:
    if not state:
        st.info("还没有运行结果。先在上方生成或续写剧本。")
        return
    cols = st.columns(4)
    cols[0].metric("故事 ID", state.get("story_id") or "-")
    cols[1].metric("集数", state.get("episode_no") or "-")
    cols[2].metric("意图", state.get("intent_type") or "-")
    cols[3].metric("入库", state.get("persist_status") or "未保存")

    if state.get("error_message"):
        st.error(state["error_message"])
    if state.get("image_url"):
        st.subheader("本集文生图")
        _show_image(state["image_url"], state.get("image_prompt_positive", "")[:120])
        with st.expander("查看绘图 Prompt"):
            st.code(state.get("image_prompt_positive") or "", language="text")
            st.code(state.get("image_prompt_negative") or "", language="text")

    with st.expander("查看完整剧本", expanded=False):
        st.markdown(state.get("script") or "（暂无）")
    with st.expander("查看完整分镜", expanded=False):
        st.markdown(state.get("storyboard") or "（暂无）")


def _parse_storyboard_rows(text: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in str(text or "").splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if not cells or "镜头序号" in cells[0]:
            continue
        if all(set(cell) <= {"-", ":", " "} for cell in cells):
            continue
        rows.append(cells)
    return rows


# ===========================================================================
# 侧边栏
# ===========================================================================
if st.session_state.get("pending_story_id"):
    st.session_state.story_id_input = st.session_state.pop("pending_story_id")

story_list = _api("GET", "/api/stories", quiet=True).get("stories") or []
story_options = [
    str(item.get("story_id"))
    for item in story_list
    if item.get("story_id")
]
current_story = st.session_state.get("story_id_input", "demo_story")
if current_story not in story_options:
    story_options.insert(0, current_story)
if not story_options:
    story_options = ["demo_story"]

with st.sidebar:
    st.header("连接设置")
    st.caption(f"后端：{API_BASE}")
    if st.button("检查后端", use_container_width=True):
        health = _api("GET", "/api/health")
        if not health.get("error"):
            st.success("后端在线")
            st.json(health)
    story_id = st.selectbox(
        "story_id",
        story_options,
        index=story_options.index(current_story) if current_story in story_options else 0,
        key="story_id_input",
    )
    style = st.selectbox(
        "画风",
        ["日系漫画", "厚涂", "写实", "赛璐璐", "水墨", "像素风"],
        index=2,
    )
    st.caption("视频熔断状态可在视频页查看/重置。")
    if st.button("刷新熔断状态", use_container_width=True):
        circuit = _api("GET", "/api/video/circuit")
        if not circuit.get("error"):
            st.session_state.circuit = circuit
    if st.session_state.get("circuit"):
        st.json(st.session_state["circuit"])


# ===========================================================================
# Tabs
# ===========================================================================
tab_script, tab_story, tab_image, tab_video, tab_trace = st.tabs(
    [
        "📝 剧本管理",
        "🎨 角色 / 分镜",
        "🖼 图片生成",
        "🎬 视频生成",
        "🧾 Trace / 日志",
    ]
)

with tab_script:
    st.subheader("新建剧本")
    st.caption(
        "新建/续写只生成剧本、分镜和视频提示词，不自动生成图片；"
        "需要图片请到【图片生成】页单独生成，避免消耗额度。"
    )
    if "new_story_id" not in st.session_state:
        st.session_state.new_story_id = ""
    if st.session_state.get("pending_new_story_id"):
        st.session_state.new_story_id = st.session_state.pop("pending_new_story_id")
    id_col, id_btn = st.columns([3, 1])
    new_story_id = id_col.text_input(
        "新剧本 story_id（必须是全新 ID）",
        key="new_story_id",
        placeholder="例如 story_006",
    )
    if id_btn.button("自动生成 story_id", use_container_width=True):
        generated = _api("GET", "/api/stories/next-id")
        if not generated.get("error"):
            st.session_state.pending_new_story_id = generated.get("story_id", "")
            st.rerun()

    new_text = st.text_area(
        "小说 / 剧本正文",
        height=260,
        placeholder="粘贴一章小说或剧本……",
    )
    if st.button("生成新剧本", type="primary", use_container_width=True):
        if not new_text.strip():
            st.warning("请先输入正文。")
        elif not new_story_id.strip():
            st.warning("请先填写新剧本 story_id，或点【自动生成 story_id】。")
        else:
            with st.spinner("LangGraph 正在生成剧本、分镜和关键帧……"):
                data = _api(
                    "POST",
                    "/api/script/new",
                    {
                        "text": new_text,
                        "story_id": new_story_id.strip(),
                        "style_desc": style,
                    },
                )
            if not data.get("error"):
                _apply_run(data)
                st.session_state.pending_story_id = new_story_id.strip()
                st.rerun()

    st.divider()
    st.subheader("续写故事")
    instruction = st.text_area(
        "续写指令",
        height=100,
        placeholder="例如：续写第五章，林昭去牢北井……",
    )
    if st.button("续写下一集", use_container_width=True):
        if not instruction.strip():
            st.warning("请先输入续写指令。")
        else:
            with st.spinner("正在检索记忆、续写、校验、出图……"):
                data = _api(
                    "POST",
                    "/api/script/continue",
                    {
                        "instruction": instruction,
                        "story_id": story_id,
                        "style_desc": style,
                    },
                )
            if not data.get("error"):
                _apply_run(data)
                st.rerun()

    if st.session_state.pending_thread:
        st.warning(
            (st.session_state.last_result or {}).get(
                "question", "意图置信度不足，需要确认。"
            )
        )
        c1, c2 = st.columns(2)
        if c1.button("确认继续 y", type="primary"):
            _confirm("y")
        if c2.button("取消 n"):
            _confirm("n")

    st.divider()
    _show_run_state(st.session_state.last_state)


with tab_story:
    st.subheader("角色库 / 历史剧情")
    if st.button("加载故事记忆", use_container_width=True):
        detail = _api("GET", f"/api/story/{story_id}")
        if not detail.get("error"):
            st.session_state.story_detail = detail

    detail = st.session_state.get("story_detail") or {}
    if detail:
        st.caption(
            f"最新第{detail.get('latest_episode')}集｜角色 {len(detail.get('roles') or [])} 个"
        )
        roles = detail.get("roles") or []
        if roles:
            cols = st.columns(min(3, len(roles)))
            for index, role in enumerate(roles):
                with cols[index % len(cols)]:
                    st.markdown(f"**{role.get('name', '未命名')}**")
                    st.caption(role.get("role") or "")
                    st.write(role.get("appearance") or "外貌未说明")
                    st.caption(role.get("relationships") or "")
        with st.expander("历史剧情摘要", expanded=False):
            for summary in detail.get("summaries") or []:
                st.markdown(
                    f"### 第{summary.get('episode_no')}集｜{summary.get('title') or ''}"
                )
                st.markdown(summary.get("content") or "")

    st.divider()
    st.subheader("分镜卡片 / 图片")
    col_ep, col_btn = st.columns([2, 1])
    episode_no = col_ep.number_input("第几集", min_value=1, value=1, step=1)
    if col_btn.button("加载该集分镜", use_container_width=True):
        episode = _api("GET", f"/api/story/{story_id}/episodes/{int(episode_no)}")
        if not episode.get("error"):
            st.session_state.episode = episode

    episode = st.session_state.episode or {}
    if episode:
        st.markdown(f"### 第{episode.get('episode_no')}集｜{episode.get('title') or ''}")
        assets = episode.get("assets") or {}
        for image in assets.get("images") or []:
            _show_image(image.get("url", ""), image.get("name", ""))
        rows = _parse_storyboard_rows(episode.get("storyboard") or "")
        for index, row in enumerate(rows, start=1):
            padded = row + [""] * (6 - len(row))
            with st.container(border=True):
                st.markdown(f"**镜头 #{padded[0]}**  ·  {padded[1]}")
                st.write(padded[2])
                if padded[3] and padded[3] != "—":
                    st.caption(f"台词/旁白：{padded[3]}")
                st.caption(f"运镜：{padded[4]}｜氛围：{padded[5]}")

        st.divider()
        st.subheader("绘图提示词")
        shot_no = st.number_input("镜头序号", min_value=1, value=1, step=1)
        if st.button("生成该镜头提示词", use_container_width=True):
            result = _api(
                "POST",
                "/api/prompt/shot",
                {
                    "story_id": story_id,
                    "episode_no": int(episode_no),
                    "shot_no": int(shot_no),
                },
            )
            if not result.get("error"):
                st.session_state.shot_result = result
                st.rerun()
        shot_result = st.session_state.get("shot_result") or {}
        if shot_result.get("text"):
            st.code(shot_result["text"], language="text")
            st.download_button(
                "下载提示词",
                data=shot_result["text"],
                file_name=f"第{shot_result.get('episode_no')}集_镜头{shot_result.get('shot_no')}.txt",
                mime="text/plain",
            )


with tab_image:
    st.subheader("角色基准图（全身正面）")
    st.caption(
        "基准图只保留角色原生基础形象：干净纯色背景，不带血迹、湿透、"
        "受伤、特殊剧情动作。满意后再确认入库。"
    )
    character_data = _api("GET", f"/api/characters/{story_id}")
    character_names = [
        item.get("name")
        for item in (character_data.get("characters") or [])
        if item.get("name")
    ]
    if character_names:
        ref_name = st.selectbox("选择角色", character_names, key="reference_character")
        ref_style = st.selectbox(
            "基准图画风",
            ["写实", "日系漫画", "厚涂", "赛璐璐", "水墨"],
            index=0,
            key="reference_style",
        )
        if st.session_state.get("reference_character_active") != ref_name:
            st.session_state.reference_character_active = ref_name
            st.session_state.reference_result = {}
        current_card = next(
            (
                item
                for item in (character_data.get("characters") or [])
                if item.get("name") == ref_name
            ),
            {},
        )
        st.caption(
            "角色库当前设定："
            f"{current_card.get('appearance') or '未说明'}"
        )
        custom_appearance = st.text_area(
            "角色外貌/服装设定（留空=读取角色库；可手写覆盖）",
            height=100,
            key=f"custom_appearance_{ref_name}",
            placeholder="例如：青年男性，左眉旧疤，深色立领长衫，随身无鞘短剑",
        )
        ref_cols = st.columns(2)
        make_ref = ref_cols[0].button(
            "生成角色基准图", type="primary", use_container_width=True
        )
        remade_ref = ref_cols[1].button(
            "重新生成基准图", use_container_width=True
        )
        if make_ref or remade_ref:
            st.session_state.reference_result = {}
            st.session_state.reference_confirmed = {}
            with st.spinner("正在生成 1080P 全身正面基准图……"):
                reference = _api(
                    "POST",
                    "/api/characters/reference/generate",
                    {
                        "story_id": story_id,
                        "character_name": ref_name,
                        "style_desc": ref_style,
                        "appearance_override": custom_appearance,
                        "nonce": str(time.time_ns()),
                        "variation": bool(remade_ref),
                    },
                )
            if not reference.get("error"):
                st.session_state.reference_result = reference
                st.rerun()

        reference = st.session_state.get("reference_result") or {}
        images = reference.get("images") or []
        if images:
            cols = st.columns(len(images))
            for col, image in zip(cols, images):
                with col:
                    _show_image(image.get("url", ""), image.get("role", ""))
            with st.expander("查看基准图提示词", expanded=False):
                for role, prompt in (reference.get("prompts") or {}).items():
                    st.markdown(f"**{role}**")
                    st.code(prompt, language="text")
            if st.button("确认满意并写入 Chroma", type="primary"):
                confirmed = _api(
                    "POST",
                    "/api/characters/reference/confirm",
                    {
                        "story_id": story_id,
                        "character_name": ref_name,
                        "image_paths": [item["path"] for item in images],
                        "style_desc": ref_style,
                        "appearance_override": custom_appearance,
                    },
                )
                if not confirmed.get("error"):
                    st.session_state.reference_confirmed = confirmed
                    st.success("角色基准图与结构化档案已写入 Chroma")
                    st.json(confirmed)

        confirmed = st.session_state.get("reference_confirmed") or {}
        if confirmed:
            st.info("已入库角色档案")
            st.json(confirmed.get("profile") or {})
    else:
        st.warning("当前剧本还没有全局角色档案，请先新建剧本完成冷启动。")

    st.divider()
    st.subheader("分镜图生成（独立）")
    image_episode = st.number_input(
        "图片集号",
        min_value=1,
        value=int(st.session_state.get("episode", {}).get("episode_no") or 1),
        step=1,
    )
    image_negative = st.text_input("额外负面词（可留空）")
    image_cols = st.columns(2)
    generate_clicked = image_cols[0].button(
        "生成分镜图", type="primary", use_container_width=True
    )
    regenerate_clicked = image_cols[1].button(
        "重新生成", use_container_width=True
    )
    if generate_clicked or regenerate_clicked:
        st.session_state.image_result = {}
        with st.spinner("Z-Image 正在生成……"):
            image = _api(
                "POST",
                "/api/storyboard/image",
                {
                    "story_id": story_id,
                    "episode_no": int(image_episode),
                    "style_desc": style,
                    "negative_extra": image_negative,
                },
            )
        if not image.get("error"):
            st.session_state.image_result = image
            st.rerun()

    image_result = st.session_state.get("image_result") or {}
    if image_result.get("image_url"):
        _show_image(image_result["image_url"], "Z-Image 分镜图")
        st.caption(f"状态：{image_result.get('image_status')}")
        with st.expander("查看/复制绘图提示词", expanded=True):
            st.code(image_result.get("image_prompt_positive") or "", language="text")
            st.code(image_result.get("image_prompt_negative") or "", language="text")
            st.download_button(
                "下载绘图提示词",
                data=(
                    "正向：\n"
                    + str(image_result.get("image_prompt_positive") or "")
                    + "\n\n负面：\n"
                    + str(image_result.get("image_prompt_negative") or "")
                ),
                file_name=f"{story_id}_第{image_episode}集_绘图提示词.txt",
                mime="text/plain",
            )


with tab_video:
    st.subheader("Seedance 文生视频")
    st.info(
        "每次生成视频会消耗额度，建议每次选取 1~3 个镜头区间，"
        "避免长文本生成失败浪费额度。关闭 enable_video 时只生成提示词，"
        "不调用接口、不扣额度。"
    )

    stories_data = _api("GET", "/api/stories")
    raw_options = stories_data.get("stories") or []
    option_ids = [item.get("story_id") for item in raw_options if item.get("story_id")]
    if story_id not in option_ids:
        option_ids.insert(0, story_id)
    if not option_ids:
        option_ids = [story_id]

    col_story, col_episode, col_start, col_end, col_load = st.columns(
        [2.2, 0.8, 0.9, 0.9, 1.2]
    )
    video_story = col_story.selectbox(
        "选择剧本",
        option_ids,
        index=option_ids.index(story_id) if story_id in option_ids else 0,
    )
    video_episode = col_episode.number_input(
        "集号",
        min_value=1,
        value=int(episode_no),
        step=1,
    )
    start_shot = col_start.number_input(
        "起始镜头",
        min_value=1,
        value=1,
        step=1,
        key="video_start_shot",
    )
    end_shot = col_end.number_input(
        "结束镜头",
        min_value=1,
        value=3,
        step=1,
        key="video_end_shot",
    )
    if col_load.button("加载本集提示词", use_container_width=True):
        with st.spinner("读取本集指定镜头区间……"):
            prompt_data = _api(
                "POST",
                "/api/video/prompt",
                {
                    "story_id": video_story,
                    "episode_no": int(video_episode),
                    "style_desc": style,
                    "start_shot": int(start_shot),
                    "end_shot": int(end_shot),
                },
                quiet=True,
            )
        if prompt_data.get("error"):
            detail = str(prompt_data.get("detail") or "")
            if "该集暂无分镜数据" in detail:
                st.warning("该集暂无分镜数据，请先生成分镜。")
            elif detail:
                st.warning(detail)
            else:
                st.warning("加载失败，请确认该集已有分镜后重试。")
        else:
            st.session_state.video_prompt_text = prompt_data.get("video_prompt", "")
            st.session_state.video_result = {}
            st.success(
                f"已加载 {video_story} 第{video_episode}集"
                f"镜头 {start_shot}-{end_shot}"
                f"（本集共 {prompt_data.get('total_shots', '?')} 个镜头）"
            )
            st.rerun()

    enable_video = st.checkbox(
        "enable_video：勾选才真正调用 Seedance（关闭时只降级保存提示词）",
        value=True,
    )
    if "video_prompt_text" not in st.session_state:
        st.session_state.video_prompt_text = ""
    video_prompt = st.text_area(
        "视频提示词（可修改；先点“加载本集提示词”自动回填）",
        key="video_prompt_text",
        height=180,
    )
    if st.button("生成视频", type="primary", use_container_width=True):
        with st.spinner("Seedance 任务处理中……"):
            result = _api(
                "POST",
                "/api/video/generate",
                {
                    "story_id": video_story,
                    "episode_no": int(video_episode),
                    "prompt": video_prompt,
                    "style_desc": style,
                    "enable_video": enable_video,
                },
            )
        if not result.get("error"):
            st.session_state.video_result = result
            st.rerun()

    result = st.session_state.get("video_result") or {}
    status = result.get("status")
    if status == "success":
        st.success("视频生成成功")
        if result.get("video_url"):
            st.video(f"{API_BASE}{result['video_url']}")
        st.caption(result.get("msg") or "")
    elif status == "skipped_degrade":
        st.warning("已手动关闭视频生成，降级输出视频提示词")
        st.caption(result.get("msg") or "")
    elif status == "fail_degrade":
        st.error("视频 API 调用失败，已降级保存提示词")
        st.caption(result.get("msg") or "")
    elif status == "circuit_open":
        st.error("熔断器已打开，暂时禁止调用视频接口，请稍后重试")
        st.caption(result.get("msg") or "")
    elif status:
        st.info(f"状态：{status}｜{result.get('msg') or ''}")

    if result.get("video_prompt"):
        st.text_area("本次视频提示词", value=result["video_prompt"], height=160)
        st.download_button(
            "下载视频提示词",
            data=result["video_prompt"],
            file_name=f"{video_story}_第{int(video_episode)}集_video_prompt.txt",
            mime="text/plain",
        )

    st.divider()
    c1, c2 = st.columns(2)
    if c1.button("刷新熔断器状态"):
        circuit = _api("GET", "/api/video/circuit")
        if not circuit.get("error"):
            st.session_state.circuit = circuit
    if c2.button("重置视频熔断器", type="secondary"):
        reset = _api("POST", "/api/video/circuit/reset")
        if not reset.get("error"):
            st.session_state.circuit = reset.get("circuit") or {}
            st.success("熔断器已重置")
    if st.session_state.get("circuit"):
        st.json(st.session_state["circuit"])


with tab_trace:
    st.subheader("LangSmith Trace")
    st.caption(
        "trace 来自 LangGraph state；LangSmith 项目由 .env 的 LANGSMITH_PROJECT 控制。"
    )
    st.link_button("打开 LangSmith", "https://smith.langchain.com")
    trace = (st.session_state.last_state or {}).get("trace") or []
    if trace:
        for item in trace:
            st.write(f"- {item}")
    else:
        st.info("运行一次后这里会显示 trace 摘要。")

    st.divider()
    st.subheader("执行日志")
    if st.session_state.logs:
        st.code("\n".join(st.session_state.logs[-80:]), language="text")
    else:
        st.info("暂无日志。")
