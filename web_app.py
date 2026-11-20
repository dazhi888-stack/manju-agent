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


def _refresh_character_views(story_id: str) -> None:
    """重新拉取故事记忆 + 角色基准图状态，删除/生成后保持界面同步。"""
    detail = _api("GET", f"/api/story/{story_id}", quiet=True)
    if not detail.get("error"):
        st.session_state.story_detail = detail
    overview = _api("GET", f"/api/characters/{story_id}", quiet=True)
    if not overview.get("error"):
        st.session_state.character_overview = overview


def _character_delete_controls(story_id: str, name: str, has_refs: bool) -> None:
    """
    角色卡下的删除交互：
    - 🗑 删基准图：只清 Chroma 向量记录 + 本地基准图文件，角色卡保留，
      之后可重新生成基准图（confirm 入库时会覆盖旧记录）；
    - ❌ 删除角色：二次确认后连角色卡文本一起删
      （全局档案 JSON + story_memory 角色向量），角色从角色库彻底消失。
    """
    col_ref, col_char = st.columns(2)
    if col_ref.button(
        "🗑 删基准图",
        key=f"del_ref_{name}",
        use_container_width=True,
        disabled=not has_refs,
        help="删除该角色已入库的基准图与向量记录，保留角色卡文本。",
    ):
        resp = _api(
            "POST",
            "/api/characters/delete",
            {
                "story_id": story_id,
                "character_name": name,
                "include_profile": False,
            },
            quiet=True,
        )
        if resp.get("error"):
            st.warning(f"删除失败：{resp.get('detail')}")
        else:
            st.success(
                f"已删除 {name} 的基准图 {len(resp.get('removed_files') or [])} 张，"
                "可重新生成。"
            )
            _refresh_character_views(story_id)
            st.rerun()
    if col_char.button(
        "❌ 删除角色",
        key=f"del_char_{name}",
        use_container_width=True,
        help="彻底删除该角色：角色卡 + 基准图 + 向量库记录，不可恢复。",
    ):
        st.session_state.confirm_delete_char = name
    if st.session_state.get("confirm_delete_char") == name:
        st.warning(f"将彻底删除「{name}」：角色卡、基准图、向量库记录全部清除，不可恢复！")
        confirm_col, cancel_col = st.columns(2)
        if confirm_col.button(
            "确认删除",
            key=f"del_confirm_{name}",
            type="primary",
            use_container_width=True,
        ):
            resp = _api(
                "POST",
                "/api/characters/delete",
                {
                    "story_id": story_id,
                    "character_name": name,
                    "include_profile": True,
                },
                quiet=True,
            )
            st.session_state.confirm_delete_char = ""
            if resp.get("error"):
                st.warning(f"删除失败：{resp.get('detail')}")
            else:
                st.success(f"「{name}」已彻底删除。")
                _refresh_character_views(story_id)
                st.rerun()
        if cancel_col.button("取消", key=f"del_cancel_{name}", use_container_width=True):
            st.session_state.confirm_delete_char = ""
            st.rerun()


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
        # 同步拉角色基准图入库状态，让角色库能看到已确认入 Chroma 的角色。
        overview = _api("GET", f"/api/characters/{story_id}")
        if not overview.get("error"):
            st.session_state.character_overview = overview

    detail = st.session_state.get("story_detail") or {}
    overview = st.session_state.get("character_overview") or {}
    refs_by_name = {
        str(item.get("name") or ""): (item.get("reference_images") or [])
        for item in (overview.get("characters") or [])
    }
    if detail:
        st.caption(
            f"最新第{detail.get('latest_episode')}集｜角色 {len(detail.get('roles') or [])} 个"
        )
        roles = detail.get("roles") or []
        if roles:
            cols = st.columns(min(3, len(roles)))
            for index, role in enumerate(roles):
                with cols[index % len(cols)]:
                    name = str(role.get("name", "未命名"))
                    refs = refs_by_name.get(name) or []
                    badge = "✅ 已入库" if refs else "⚪ 未生成基准图"
                    st.markdown(f"**{name}**（{badge}）")
                    st.caption(role.get("role") or "")
                    st.write(role.get("appearance") or "外貌未说明")
                    st.caption(role.get("relationships") or "")
                    for ref_image in refs:
                        _show_image(
                            ref_image.get("url", ""),
                            f"基准图·{ref_image.get('image_role') or 'full'}",
                        )
                    _character_delete_controls(story_id, name, bool(refs))
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
        if episode.get("error"):
            st.session_state.episode = {}
            st.warning(
                f"第{int(episode_no)}集加载失败：{episode.get('detail') or '该集数据不存在，请先在这一集上跑一遍生成。'}"
            )
        else:
            st.session_state.episode = episode

    episode = st.session_state.episode or {}
    if episode:
        st.markdown(f"### 第{episode.get('episode_no')}集｜{episode.get('title') or ''}")
        assets = episode.get("assets") or []
        image_list = (assets.get("images") if isinstance(assets, dict) else None) or []
        if image_list:
            st.caption(
                f"本集已收录 {len(image_list)} 张图（一张有效图绑定一个镜头；"
                "不满意的可在图下方删除）"
            )
            for image in image_list:
                shot_no_tag = image.get("shot_no") or 0
                caption = image.get("name", "")
                if shot_no_tag:
                    caption = f"{caption}（镜头#{shot_no_tag}{'·已绑定' if image.get('bound') else ''}）"
                with st.container(border=True):
                    _show_image(image.get("url", ""), caption)
                    if st.button(
                        "🗑️ 删除此图",
                        key=f"del_img_{image.get('name')}",
                        use_container_width=True,
                    ):
                        deleted = _api(
                            "POST",
                            "/api/storyboard/image/delete",
                            {
                                "story_id": story_id,
                                "file_name": image.get("name", ""),
                            },
                        )
                        if deleted.get("error"):
                            st.warning(f"删除失败：{deleted.get('detail')}")
                        else:
                            st.toast("图片已删除")
                            st.rerun()
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
            "重新生成基准图（消耗额度）", use_container_width=True
        )
        st.caption(
            "已确认入库的角色点「生成」会直接复用入库图，不消耗额度；"
            "只有「重新生成」才会真正重画。"
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
                        "force_new": bool(remade_ref),
                    },
                )
            if not reference.get("error"):
                st.session_state.reference_result = reference
                st.rerun()

        reference = st.session_state.get("reference_result") or {}
        if reference.get("reused"):
            st.info(
                "✅ 该角色已有确认入库的基准图，本次直接复用（未调用出图接口）。"
                "如需换图请点「重新生成基准图（消耗额度）」。"
            )
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
                    # 同步刷新入库状态，切到角色库立即可见，无需再点加载。
                    refreshed = _api("GET", f"/api/characters/{story_id}")
                    if not refreshed.get("error"):
                        st.session_state.character_overview = refreshed
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
    # 集数变化时让旧的镜头列表失效，避免拿错集的镜头号。
    if st.session_state.get("shot_options_episode") != int(image_episode):
        st.session_state.shot_options = []
        st.session_state.shot_options_episode = int(image_episode)

    if st.button("读取该集镜头列表", use_container_width=True):
        episode_data = _api(
            "GET", f"/api/story/{story_id}/episodes/{int(image_episode)}"
        )
        if episode_data.get("error"):
            st.session_state.shot_options = []
            st.warning(
                f"第{int(image_episode)}集还没有分镜数据，请先在「剧本」页生成或续写。"
            )
        else:
            st.session_state.shot_options = _parse_storyboard_rows(
                episode_data.get("storyboard") or ""
            )
            st.session_state.shot_options_episode = int(image_episode)

    shot_options = st.session_state.get("shot_options") or []
    # 显示完整画面描述，不截断——选镜头靠的就是读这段文字。
    shot_labels = [
        f"#{row[0]}｜{row[2] if len(row) > 2 else ''}"
        for row in shot_options
    ]
    shot_choice = st.selectbox(
        "选择镜头（不选 = 整集自动挑关键画面）",
        ["整集（自动选关键镜头）"] + shot_labels,
        key="storyboard_shot_choice",
    )
    selected_shot_no = 0
    if shot_choice != "整集（自动选关键镜头）":
        selected_shot_no = int(shot_choice.split("｜")[0].lstrip("#"))

    image_negative = st.text_input("额外负面词（可留空）")
    image_cols = st.columns(3)
    generate_clicked = image_cols[0].button(
        "生成分镜图", type="primary", use_container_width=True
    )
    regenerate_clicked = image_cols[1].button(
        "♻️ 重新生成（自动审核+最多重绘2次）", use_container_width=True
    )
    batch_clicked = image_cols[2].button(
        "⚡ 整集批量出图（并发3路）", use_container_width=True
    )
    if generate_clicked or regenerate_clicked:
        st.session_state.image_result = {}
        st.session_state.manual_prompt = ""
        with st.spinner("出图中：Z-Image 生成 → 审核节点校验（人物数/角色一致性/OOC）……"):
            image = _api(
                "POST",
                "/api/storyboard/image",
                {
                    "story_id": story_id,
                    "episode_no": int(image_episode),
                    "shot_no": selected_shot_no or None,
                    "style_desc": style,
                    "negative_extra": image_negative,
                    "prompt_override": st.session_state.get("manual_prompt") or "",
                },
            )
        if not image.get("error"):
            st.session_state.image_result = image
        else:
            st.warning(f"生成失败：{image.get('detail') or image.get('error')}")
    if batch_clicked:
        st.session_state.batch_image_result = {}
        with st.spinner("整集批量出图：多镜头并发生成 → 每张独立过审核 → 未过的自动重绘……"):
            batch = _api(
                "POST",
                "/api/storyboard/image/batch",
                {
                    "story_id": story_id,
                    "episode_no": int(image_episode),
                    "style_desc": style,
                    "negative_extra": image_negative,
                },
            )
        if not batch.get("error"):
            st.session_state.batch_image_result = batch
        else:
            st.warning(f"批量出图失败：{batch.get('detail') or batch.get('error')}")

    batch_result = st.session_state.get("batch_image_result") or {}
    if batch_result.get("results"):
        summary = batch_result.get("summary") or {}
        st.caption(
            f"批量完成：共 {summary.get('total', 0)} 镜头｜✅ 通过 {summary.get('done', 0)}"
            f"｜⚠️ 待人工 {summary.get('needs_manual', 0)}"
            f"｜❌ 失败 {summary.get('failed', 0)}"
            f"｜总耗时 {batch_result.get('total_seconds', 0)}s"
        )
        for item in batch_result.get("results") or []:
            shot_no = item.get("shot_no")
            status = str(item.get("status") or "")
            mark = {"done": "✅", "needs_manual_prompt": "⚠️"}.get(status, "❌")
            err = item.get("image_error") or ""
            line = f"{mark} 镜头#{shot_no}｜{status}｜耗时 {item.get('elapsed_seconds', '-')}s"
            if err:
                line += f"｜{err}"
            st.caption(line)

    image_result = st.session_state.get("image_result") or {}
    if image_result.get("image_url"):
        shot_tag = image_result.get("shot_no")
        label = f"Z-Image 分镜图（镜头 #{shot_tag}）" if shot_tag else "Z-Image 分镜图（整集）"
        _show_image(image_result["image_url"], label)

        # ---- 审核结果面板 ----
        image_status = str(image_result.get("image_status") or "")
        review = image_result.get("image_review") or {}
        issues = review.get("issues") or []
        if image_status == "done":
            st.success("✅ 审核通过：人物数量、角色一致性、OOC 检查均无问题（尚未入库，等你确认）")
        elif image_status == "needs_manual_prompt":
            st.error(
                "⚠️ 自动重绘 2 次仍未通过审核，本镜头已标记为「待人工修改 prompt」，"
                "不再自动重绘。请在下方修改提示词后手动重绘。\n\n"
                + "\n".join(f"- {issue}" for issue in issues)
            )
        elif issues:
            st.warning(
                "审核发现的问题（已自动重绘）：" + "；".join(str(i) for i in issues)
            )
        st.caption(
            f"状态：{image_status}｜自动重绘 {image_result.get('image_retry_count') or 0} 次"
            f"｜未采用的图不会进入分镜卡片"
        )

        # 待人工修改 prompt：给编辑框 + 手动重绘
        if image_result.get("needs_manual_prompt"):
            suggested = str(review.get("suggested_prompt_fix") or "")
            default_prompt = (
                st.session_state.get("manual_prompt")
                or suggested
                or str(image_result.get("image_prompt_positive") or "")
            )
            manual = st.text_area(
                "人工修改绘图提示词（改完点下方按钮，直接以此重绘）",
                value=default_prompt,
                height=150,
                key="manual_prompt_box",
            )
            st.session_state.manual_prompt = manual
            if st.button("📝 用修改后的提示词重绘", type="primary", use_container_width=True):
                with st.spinner("按人工提示词重绘中……"):
                    image = _api(
                        "POST",
                        "/api/storyboard/image",
                        {
                            "story_id": story_id,
                            "episode_no": int(image_episode),
                            "shot_no": selected_shot_no or None,
                            "style_desc": style,
                            "negative_extra": image_negative,
                            "prompt_override": manual,
                        },
                    )
                if not image.get("error"):
                    st.session_state.image_result = image
                else:
                    st.warning(f"重绘失败：{image.get('detail') or image.get('error')}")

        # 确认制：满意才"采用"进分镜卡片（绑定 shot_id），不满意"丢弃"
        if image_result.get("image_name"):
            keep_cols = st.columns(2)
            if keep_cols[0].button(
                "✅ 采用此图（存入分镜卡片）", type="primary", use_container_width=True
            ):
                confirmed = _api(
                    "POST",
                    "/api/storyboard/image/confirm",
                    {
                        "story_id": story_id,
                        "file_name": image_result["image_name"],
                        "episode_no": int(image_episode),
                        "shot_no": int(shot_tag or 0),
                    },
                )
                if confirmed.get("error"):
                    st.warning(f"采用失败：{confirmed.get('detail')}")
                else:
                    binding = confirmed.get("binding") or {}
                    st.success(
                        f"已存入分镜卡片并绑定镜头（shot_id={binding.get('shot_id')}）。"
                        "同镜头重新采用会自动替换旧图。"
                    )
                    st.session_state.image_result = {}
            if keep_cols[1].button("🗑️ 丢弃此图", use_container_width=True):
                discarded = _api(
                    "POST",
                    "/api/storyboard/image/discard",
                    {
                        "story_id": story_id,
                        "file_name": image_result["image_name"],
                    },
                )
                if discarded.get("error"):
                    st.warning(f"丢弃失败：{discarded.get('detail')}")
                else:
                    st.session_state.image_result = {}

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
                file_name=(
                    f"{story_id}_第{image_episode}集"
                    + (f"_镜头{shot_tag}" if shot_tag else "")
                    + "_绘图提示词.txt"
                ),
                mime="text/plain",
            )


with tab_video:
    st.subheader("🎬 成片合成（配音 + 字幕 + 拼接）")
    st.caption(
        "用已入库的分镜图 + 台词 TTS（微软免费语音）+ ffmpeg 本地合成完整 mp4，"
        "不消耗图片/视频额度；没有绑定图的镜头会跳过并在报告中说明。"
    )
    compose_cols = st.columns([1, 2])
    compose_episode_no = compose_cols[0].number_input(
        "合成集号",
        min_value=1,
        value=int(st.session_state.get("episode", {}).get("episode_no") or 1),
        step=1,
        key="compose_episode_no",
    )
    compose_clicked = compose_cols[1].button(
        "🎬 合成本集成片", type="primary", use_container_width=True
    )
    burn_subtitles = st.checkbox(
        "画面烧录字幕（旧分镜图若自带台词文字，可关掉，仅保留外挂 srt）",
        value=True,
        key="burn_subtitles",
    )
    if compose_clicked:
        st.session_state.compose_result = {}
        with st.spinner(
            "合成中：TTS 配音 → 字幕烧录 → ffmpeg 拼接（一集约 1~3 分钟）……"
        ):
            report = _api(
                "POST",
                "/api/compose",
                {
                    "story_id": story_id,
                    "episode_no": int(compose_episode_no),
                    "burn_subtitles": bool(burn_subtitles),
                },
            )
        if not report.get("error"):
            st.session_state.compose_result = report
        else:
            st.warning(f"合成失败：{report.get('detail') or report.get('error')}")

    compose_result = st.session_state.get("compose_result") or {}
    if compose_result.get("video_url"):
        composed = compose_result.get("shots") or []
        skipped = compose_result.get("skipped") or []
        st.success(
            f"合成完成：{len(composed)} 个镜头｜成片时长 {compose_result.get('video_seconds')}s"
            f"｜配音音色 {compose_result.get('tts_voice')}"
            f"｜合成耗时 {compose_result.get('compose_seconds')}s"
        )
        st.video(compose_result["video_url"])
        link_cols = st.columns(2)
        link_cols[0].markdown(f"[⬇️ 下载成片 mp4]({compose_result['video_url']})")
        if compose_result.get("srt_url"):
            link_cols[1].markdown(f"[⬇️ 下载外挂字幕 srt]({compose_result['srt_url']})")
        if skipped:
            st.caption(
                "跳过的镜头："
                + "；".join(
                    f"#{item.get('shot_no')}（{item.get('reason')}）"
                    for item in skipped
                )
            )

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
