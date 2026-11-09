"""FastAPI 包装层：把现有 LangGraph 暴露成网页可调用的异步接口。

注意：
- 不修改 graph / video_gen 的熔断、降级、文件保存逻辑；
- 视频接口直接调用 video_graph_node；
- 返回原样透传 video 节点的 status / msg / video_prompt / video_path。
"""

from __future__ import annotations

import asyncio
import re
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.error import HTTPError

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from langgraph.types import Command
from pydantic import BaseModel, Field

import config
from character_reference import (
    character_prompts,
    get_character_references,
    profile_from_card,
    reference_block as character_reference_block,
    save_character_reference,
    save_reference_image,
)
from graph import graph
from global_profile import global_profile_cards, load_global_profile, profile_path
from image_gen import ImageApiError, external_image_api, image_gen_node
from node_video_prompt import build_video_prompt_node
from output_writer import save_state_markdown
from shot_prompt import (
    generate_shot_prompt,
    load_episode_context,
    parse_storyboard_shots,
    select_key_shot,
)
from shot_store import load_episode_shots, render_storyboard_markdown
from state import ComicState
from video_prompt import storyboard_shots
from video_gen import video_circuit_breaker, video_graph_node
from vector_store import get_story_memory


app = FastAPI(
    title="AI 漫剧 Agent API",
    version="1.0.0",
    description="LangGraph 漫剧生成 / 文生图 / Seedance 视频（含熔断降级）",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def root() -> str:
    """后端根路径不再 404；给出 Swagger 和 Streamlit 地址。"""
    return """
    <!doctype html>
    <html lang="zh-CN">
    <head><meta charset="utf-8"><title>AI 漫剧 Agent API</title></head>
    <body style="font-family:system-ui;max-width:760px;margin:60px auto;line-height:1.8">
      <h1>🎬 AI 漫剧 Agent API</h1>
      <p>后端已启动。</p>
      <ul>
        <li><a href="/docs">FastAPI Swagger 文档</a></li>
        <li><a href="http://127.0.0.1:8501" target="_blank">打开 Streamlit 演示页面（8501）</a></li>
        <li><a href="/api/health">健康检查</a></li>
      </ul>
      <p style="color:#64748b">如果 Streamlit 页面打不开，请确认第二个终端已运行<br>
      <code>python -m streamlit run web_app.py --server.port 8501</code></p>
    </body></html>
    """


@app.get("/app")
async def app_redirect() -> RedirectResponse:
    """方便从后端直接跳到 Streamlit。"""
    return RedirectResponse(url="http://127.0.0.1:8501")

config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
app.mount(
    "/media",
    StaticFiles(directory=str(config.OUTPUT_DIR)),
    name="media",
)


# ===========================================================================
# 请求模型
# ===========================================================================
class NewScriptRequest(BaseModel):
    text: str = Field(min_length=1, description="小说/剧本正文")
    story_id: str = Field(default="demo_story")
    style_desc: str = Field(default="")
    max_revisions: int | None = None


class ContinueRequest(BaseModel):
    instruction: str = Field(min_length=1, description="续写指令")
    story_id: str = Field(default="demo_story")
    style_desc: str = Field(default="")


class ConfirmRequest(BaseModel):
    thread_id: str
    answer: str = "n"


class ImageRequest(BaseModel):
    story_id: str = "demo_story"
    episode_no: int | None = None
    style_desc: str = ""
    negative_extra: str = ""


class PromptRequest(BaseModel):
    story_id: str = "demo_story"
    episode_no: int | None = None
    shot_no: int | None = None
    style_desc: str = ""
    extra_negative: str = ""


class VideoRequest(BaseModel):
    story_id: str = "demo_story"
    episode_no: int | None = None
    prompt: str = ""
    style_desc: str = ""
    enable_video: bool = True


class VideoPromptRequest(BaseModel):
    story_id: str = "demo_story"
    episode_no: int | None = None
    style_desc: str = ""
    start_shot: int = 1
    end_shot: int = 3


class CharacterGenerateRequest(BaseModel):
    story_id: str = "demo_story"
    character_name: str
    style_desc: str = ""
    appearance_override: str = ""
    nonce: str = ""
    variation: bool = False


class CharacterConfirmRequest(BaseModel):
    story_id: str = "demo_story"
    character_name: str
    image_paths: list[str] = Field(min_length=1)
    style_desc: str = ""
    appearance_override: str = ""


# ===========================================================================
# 工具
# ===========================================================================
def _run_config(thread_id: str) -> dict[str, Any]:
    return {
        "configurable": {"thread_id": thread_id},
        "run_name": "manju-web-api",
        "tags": ["manju-web", "langgraph"],
        "metadata": {"source": "fastapi"},
        "recursion_limit": 80,
    }


def _public_state(state: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "story_id", "episode_no", "title", "intent_type", "intent_confidence",
        "script", "storyboard", "plot_summary", "validation",
        "branch_validation", "persist_status", "error_message",
        "style_desc", "image_prompt_positive", "image_prompt_negative",
        "image_path", "image_status", "image_error", "image_gen_ms",
        "image_records", "video_prompt", "video_path", "video_status",
        "video_gen_ms", "trace", "shots", "episode_tone", "shot_file",
        "shots_cached", "global_profile", "global_profile_path",
        "global_profile_cached", "start_shot", "end_shot",
        "video_prompt_start", "video_prompt_end",
    ]
    result = {key: state.get(key) for key in keys if key in state}
    result["history_roles"] = state.get("history_roles") or []
    result["history_summaries"] = state.get("history_summaries") or []
    result["image_url"] = _media_url(str(state.get("image_path") or ""))
    result["video_url"] = _media_url(str(state.get("video_path") or ""))
    return result


def _media_url(path: str) -> str:
    if not path:
        return ""
    try:
        relative = Path(path).resolve().relative_to(config.OUTPUT_DIR.resolve())
        return f"/media/{relative.as_posix()}"
    except Exception:
        return ""


def _episode_assets(story_id: str, episode_no: int) -> dict[str, Any]:
    story_dir = config.OUTPUT_DIR / story_id
    images = []
    videos = []
    prompts = []
    patterns = {
        "images": (story_dir / "images", (f"第{episode_no}章*",), (".png", ".jpg", ".jpeg", ".webp"), images),
        "videos": (
            story_dir / "videos",
            (f"第{episode_no}集*", f"第{episode_no}章*"),
            (".mp4", ".webm"),
            videos,
        ),
        "prompts": (story_dir / "prompts", (f"第{episode_no}章*",), (".txt",), prompts),
    }
    for _, (directory, globs, suffixes, bucket) in patterns.items():
        if not directory.exists():
            continue
        for pattern in globs:
            for path in directory.glob(pattern):
                if path.suffix.lower() in suffixes:
                    bucket.append(
                        {
                            "name": path.name,
                            "path": str(path),
                            "url": _media_url(str(path)),
                        }
                    )
    return {"images": images, "videos": videos, "prompts": prompts}


async def _invoke_graph(initial: dict[str, Any], thread_id: str) -> dict[str, Any]:
    result = await graph.ainvoke(initial, config=_run_config(thread_id))
    if isinstance(result, dict) and "__interrupt__" in result:
        interrupts = result.get("__interrupt__") or []
        first = interrupts[0] if interrupts else {}
        payload = getattr(first, "value", first)
        question = (
            payload.get("question")
            if isinstance(payload, dict)
            else "需要确认是否继续。"
        )
        return {
            "status": "needs_confirmation",
            "thread_id": thread_id,
            "question": question,
            "state": _public_state(result),
        }
    state = result if isinstance(result, dict) else {}
    status = "error" if state.get("error_message") else "success"
    output_path = ""
    if status == "success":
        try:
            saved = save_state_markdown(state)
            output_path = str(saved) if saved else ""
        except Exception as exc:
            print(f"[Web 输出保存警告] {exc}", flush=True)
    return {
        "status": status,
        "thread_id": thread_id,
        "state": _public_state(state),
        "trace": state.get("trace") or [],
        "output_path": output_path,
    }


def _base_initial(text: str, story_id: str, style_desc: str) -> dict[str, Any]:
    return {
        "raw_text": text,
        "user_query": text,
        "story_id": story_id,
        "style_desc": style_desc or config.DEFAULT_STYLE_DESC,
        "enable_video": False,
        "max_revisions": config.MAX_REVISIONS,
        "branch_max_revisions": config.MAX_BRANCH_REVISIONS,
        "trace": [],
    }


def _story_exists(story_id: str) -> bool:
    """新建剧本前强校验：Chroma、全局档案、输出目录任一存在即视为旧剧本。"""
    if not story_id:
        return False
    if story_id in get_story_memory().list_story_ids():
        return True
    if profile_path(story_id).exists():
        return True
    return (config.OUTPUT_DIR / story_id).exists()


def _reference_profile(
    card: dict[str, Any],
    story_id: str,
    appearance_override: str = "",
) -> Any:
    """默认读角色库；用户填写了外貌设定时覆盖提示词里的外貌字段。"""
    base = profile_from_card(card, story_id)
    override = str(appearance_override or "").strip()
    if override:
        base.facial_features = override
        base.signature_features = override
    return base


@app.get("/api/stories/next-id")
async def next_story_id() -> dict[str, Any]:
    """生成未占用的 story_001 / story_002 ……"""
    memory = get_story_memory()
    existing = set(memory.list_story_ids())
    if config.OUTPUT_DIR.exists():
        existing |= {path.name for path in config.OUTPUT_DIR.iterdir() if path.is_dir()}
    max_no = 0
    for item in existing:
        match = re.fullmatch(r"story_(\d+)", item)
        if match:
            max_no = max(max_no, int(match.group(1)))
    return {"story_id": f"story_{max_no + 1:03d}"}


# ===========================================================================
# 剧本接口
# ===========================================================================
@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {
        "ok": True,
        "image_model": config.IMAGE_API_MODEL,
        "video_model": config.SEEDANCE_MODEL,
        "circuit": video_circuit_breaker.snapshot(),
    }


@app.post("/api/script/new")
async def new_script(req: NewScriptRequest) -> dict[str, Any]:
    story_id = str(req.story_id or "").strip()
    if not story_id:
        raise HTTPException(status_code=422, detail="story_id 不能为空")
    if _story_exists(story_id):
        raise HTTPException(
            status_code=409,
            detail="该story_id已存在，请更换新ID，或者确认是否要清空旧剧本记忆",
        )
    thread_id = uuid.uuid4().hex
    initial = _base_initial(req.text, story_id, req.style_desc)
    if req.max_revisions is not None:
        initial["max_revisions"] = max(0, req.max_revisions)
    try:
        return await _invoke_graph(initial, thread_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")


@app.post("/api/script/continue")
async def continue_story(req: ContinueRequest) -> dict[str, Any]:
    story_id = str(req.story_id or "").strip()
    if not _story_exists(story_id):
        raise HTTPException(
            status_code=404,
            detail=f"story_id={story_id} 不存在；续写必须沿用原剧本的 story_id",
        )
    thread_id = uuid.uuid4().hex
    initial = _base_initial(req.instruction, story_id, req.style_desc)
    initial["user_instruction"] = req.instruction
    try:
        return await _invoke_graph(initial, thread_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")


@app.post("/api/script/confirm")
async def confirm(req: ConfirmRequest) -> dict[str, Any]:
    try:
        result = await graph.ainvoke(
            Command(resume=req.answer),
            config=_run_config(req.thread_id),
        )
        if isinstance(result, dict) and "__interrupt__" in result:
            return {
                "status": "needs_confirmation",
                "thread_id": req.thread_id,
                "state": _public_state(result),
            }
        state = result if isinstance(result, dict) else {}
        status = "error" if state.get("error_message") else "success"
        output_path = ""
        if status == "success":
            try:
                saved = save_state_markdown(state)
                output_path = str(saved) if saved else ""
            except Exception as exc:
                print(f"[Web 输出保存警告] {exc}", flush=True)
        return {
            "status": status,
            "thread_id": req.thread_id,
            "state": _public_state(state),
            "output_path": output_path,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")


# ===========================================================================
# 故事 / 分镜 / 图像接口
# ===========================================================================
@app.get("/api/story/{story_id}")
async def story_detail(story_id: str) -> dict[str, Any]:
    memory = get_story_memory()
    if story_id not in memory.list_story_ids():
        raise HTTPException(status_code=404, detail=f"story_id={story_id} 不存在")
    roles = global_profile_cards(load_global_profile(story_id)) or memory.get_roles(story_id)
    summaries = memory.get_summaries(story_id)
    return {
        "story_id": story_id,
        "latest_episode": memory.get_latest_episode_no(story_id),
        "roles": roles,
        "summaries": [
            {
                "episode_no": item["metadata"].get("episode_no"),
                "title": item["metadata"].get("title"),
                "content": item.get("document", ""),
            }
            for item in summaries
        ],
        "story_bible": memory.get_story_bible(story_id),
    }


@app.get("/api/story/{story_id}/episodes/{episode_no}")
async def episode_detail(story_id: str, episode_no: int) -> dict[str, Any]:
    try:
        context = load_episode_context(story_id, episode_no)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {
        "story_id": story_id,
        "episode_no": context["episode_no"],
        "title": context["title"],
        "summary": context["summary"],
        "script": context["script"],
        "storyboard": context["storyboard"],
        "characters": context["characters"],
        "assets": _episode_assets(story_id, context["episode_no"]),
    }


@app.get("/api/characters/{story_id}")
async def character_references(story_id: str) -> dict[str, Any]:
    profile = load_global_profile(story_id) or {}
    return {
        "story_id": story_id,
        "characters": profile.get("characters") or [],
        "references": get_character_references(story_id),
    }


@app.post("/api/characters/reference/generate")
async def generate_character_reference(req: CharacterGenerateRequest) -> dict[str, Any]:
    """生成半身正面 + 全身正面基准图；此步骤不写 Chroma，等用户确认。"""
    profile = load_global_profile(req.story_id) or {}
    card = next(
        (
            item for item in profile.get("characters") or []
            if str(item.get("name") or "") == req.character_name
        ),
        None,
    )
    if not card:
        raise HTTPException(status_code=404, detail=f"角色不存在：{req.character_name}")
    base = _reference_profile(card, req.story_id, req.appearance_override)
    prompts = character_prompts(base, req.style_desc)
    safe_prompts = character_prompts(base, req.style_desc, safe=True)
    variation_text = (
        "重新生成：改变站姿和构图，保持同一个人物的五官、发型和服装；"
        if req.variation
        else ""
    )
    nonce = req.nonce or str(int(time.time() * 1000))
    images = []
    try:
        # 只生成一张全身正面基准图。
        role = "full"
        content: bytes | None = None
        last_error: Exception | None = None
        for attempt in range(3):
            use_prompt = prompts[role] if attempt == 0 else safe_prompts[role]
            use_prompt = variation_text + use_prompt
            try:
                print(
                    f"[角色基准图] {role} attempt={attempt + 1} "
                    f"prompt_len={len(use_prompt)}",
                    flush=True,
                )
                content = await external_image_api(
                    use_prompt,
                    base.negative_prompt
                    + "，裸露，半裸，赤裸，无衣物，双人，多人，拼图，多视图，"
                    "三视图，角色设定图，转身图，背面图，重复人物",
                    req.style_desc or config.DEFAULT_STYLE_DESC,
                    image_size="1080x1920",
                )
                break
            except HTTPError as exc:
                last_error = exc
                print(
                    f"[角色基准图] {role} HTTP {exc.code} attempt={attempt + 1}",
                    flush=True,
                )
                if exc.code in {429, 451, 500, 502, 503} and attempt < 2:
                    await asyncio.sleep(3)
                    continue
                raise
        if content is None:
            raise last_error or RuntimeError("角色基准图接口没有返回图片")
        path = save_reference_image(req.story_id, base.name, role, content)
        images.append(
            {
                "role": "full",
                "path": str(path),
                "url": f"{_media_url(str(path))}?v={nonce}",
            }
        )
    except ImageApiError as exc:
        raise HTTPException(status_code=502, detail=f"角色基准图生成失败：{exc}")
    except HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                f"角色基准图生成失败：HTTP {exc.code}。"
                "服务商临时风控/地区限制时可稍后重试，或换一个角色/画风。"
            ),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"角色基准图生成失败：{type(exc).__name__}: {exc}"
        )
    return {
        "story_id": req.story_id,
        "character_name": base.name,
        "profile": base.model_dump(),
        "prompts": {"full": prompts["full"]},
        "images": images,
    }


@app.post("/api/characters/reference/confirm")
async def confirm_character_reference(req: CharacterConfirmRequest) -> dict[str, Any]:
    """用户确认满意后，结构化档案 + CLIP 图像向量写入 Chroma。"""
    profile = load_global_profile(req.story_id) or {}
    card = next(
        (
            item for item in profile.get("characters") or []
            if str(item.get("name") or "") == req.character_name
        ),
        None,
    )
    if not card:
        raise HTTPException(status_code=404, detail=f"角色不存在：{req.character_name}")
    base = _reference_profile(card, req.story_id, req.appearance_override)
    base.reference_images = list(req.image_paths)
    try:
        stored = save_character_reference(req.story_id, base, req.image_paths)
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"角色档案入库失败：{type(exc).__name__}: {exc}"
        )
    return {
        "story_id": req.story_id,
        "profile": base.model_dump(),
        "stored": stored,
    }


@app.post("/api/storyboard/image")
async def storyboard_image(req: ImageRequest) -> dict[str, Any]:
    try:
        context = load_episode_context(req.story_id, req.episode_no)
        state: ComicState = {
            "story_id": req.story_id,
            "episode_no": context["episode_no"],
            "storyboard": context["storyboard"],
            "script": context["script"],
            "style_desc": req.style_desc or config.DEFAULT_STYLE_DESC,
            "image_negative_extra": req.negative_extra,
            "canon_profiles": context["characters"],
        }
        result = await image_gen_node(state)
        result["image_url"] = _media_url(result.get("image_path", ""))
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")


@app.post("/api/prompt/shot")
async def shot_prompt(req: PromptRequest) -> dict[str, Any]:
    try:
        context = load_episode_context(req.story_id, req.episode_no)
        shot_no = req.shot_no
        if not shot_no:
            shots = parse_storyboard_shots(context["storyboard"])
            shot_no = int(select_key_shot(shots, context["characters"])["shot_no"])
        return await generate_shot_prompt(
            story_id=req.story_id,
            shot_no=int(shot_no),
            episode_no=context["episode_no"],
            style_desc=req.style_desc,
            extra_negative=req.extra_negative,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")


@app.get("/api/stories")
async def list_stories() -> dict[str, Any]:
    memory = get_story_memory()
    return {
        "stories": [
            {
                "story_id": story_id,
                "latest_episode": memory.get_latest_episode_no(story_id),
            }
            for story_id in memory.list_story_ids()
        ]
    }


@app.post("/api/video/prompt")
async def video_prompt_for_episode(req: VideoPromptRequest) -> dict[str, Any]:
    """根据 story_id + 集号 + 镜头区间读取本集分镜，自动拼接视频提示词。"""
    try:
        context = load_episode_context(req.story_id, req.episode_no)
    except Exception as exc:
        raise HTTPException(
            status_code=404,
            detail=f"该集暂无分镜数据，请先生成分镜。({exc})",
        )
    # 优先使用 shot_generate_node 落盘的 ep_XX.json；旧数据回退 Markdown。
    stored = load_episode_shots(req.story_id, context["episode_no"])
    shots = (stored or {}).get("shots") or storyboard_shots(context.get("storyboard", ""))
    storyboard = (
        render_storyboard_markdown(shots, context.get("title", ""))
        if shots
        else context.get("storyboard", "")
    )
    total = len(shots)
    start = int(req.start_shot)
    end = int(req.end_shot)
    if start < 1 or end < start:
        raise HTTPException(
            status_code=400,
            detail="起始镜头不能大于结束镜头，请重新选择区间",
        )
    if start > total or end > total:
        raise HTTPException(
            status_code=400,
            detail=f"本集一共{total}个镜头，请重新选择区间",
        )
    state: ComicState = {
        "story_id": req.story_id,
        "episode_no": context["episode_no"],
        "title": context.get("title", ""),
        "style_desc": req.style_desc or config.DEFAULT_STYLE_DESC,
        "shots": shots,
        "storyboard": storyboard,
        "global_profile": load_global_profile(req.story_id)
        or {"characters": context.get("characters") or []},
        "start_shot": start,
        "end_shot": end,
    }
    # 按钮触发的就是纯 Python build_video_prompt_node。
    result = build_video_prompt_node(state)
    return {
        "story_id": req.story_id,
        "episode_no": context["episode_no"],
        "title": context.get("title", ""),
        "style_desc": state["style_desc"],
        "start_shot": result.get("video_prompt_start", start),
        "end_shot": result.get("video_prompt_end", end),
        "total_shots": total,
        "video_prompt": result.get("video_prompt", ""),
    }


# ===========================================================================
# 视频接口：原样复用 video_graph_node
# ===========================================================================
@app.post("/api/video/generate")
async def generate_video(req: VideoRequest) -> dict[str, Any]:
    prompt = req.prompt.strip()
    context: dict[str, Any] | None = None
    if not prompt:
        try:
            context = load_episode_context(req.story_id, req.episode_no)
            prompt = str(context.get("summary") or context.get("storyboard") or "")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"无法读取视频上下文：{exc}")
    ref_block = character_reference_block(req.story_id)
    if ref_block and "@Image" not in prompt:
        prompt = ref_block + "\n" + prompt
    state: ComicState = {
        "story_id": req.story_id,
        "episode_no": (
            context["episode_no"] if context else (req.episode_no or 1)
        ),
        "style_desc": req.style_desc or config.DEFAULT_STYLE_DESC,
        "video_prompt": prompt,
        "image_prompt_positive": prompt,
        "enable_video": bool(req.enable_video),
    }
    try:
        result = await video_graph_node(state)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")

    # video_graph_node 在图内只回写 state；这里原样透传固定字段。
    status = result.get("status") or result.get("video_status") or "fail_degrade"
    msg = result.get("msg") or ""
    video_path = result.get("video_path") or ""
    return {
        "status": status,
        "msg": msg,
        "video_prompt": result.get("video_prompt") or prompt,
        "video_path": video_path,
        "video_url": _media_url(video_path),
        "circuit": video_circuit_breaker.snapshot(),
    }


@app.post("/api/video/circuit/reset")
async def reset_video_circuit() -> dict[str, Any]:
    video_circuit_breaker.reset()
    return {
        "ok": True,
        "msg": "视频熔断器已重置为 CLOSED",
        "circuit": video_circuit_breaker.snapshot(),
    }


@app.get("/api/video/circuit")
async def video_circuit() -> dict[str, Any]:
    return video_circuit_breaker.snapshot()
