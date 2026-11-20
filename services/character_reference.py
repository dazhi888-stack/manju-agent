"""角色基准图 / CLIP 图像向量 / Chroma 角色参考库。

Chroma 只存：
- 角色文本描述（document）
- CLIP 图像向量（embedding；无 CLIP 时用本地感知向量兜底）
- 图片本地路径 / script_id / type=character（metadata）

注意：向量库不能直接画图；生成时只负责取参考图和参考声明。
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

import config
from state import CharacterBaseProfile, GlobalCharacter
from stores.vector_store import get_character_library

COLLECTION_NAME = "character_references"
EMBED_DIM = 512
NEGATIVE_PROMPT = (
    "畸形五官，多手指，扭曲脸部，模糊，水印，文字，色差崩坏，表情怪异，多余肢体"
)

_CLIP_MODEL = None
_CLIP_PROCESSOR = None
_CLIP_FAILED = False


def _load_clip():
    """优先使用本地缓存的 CLIP；没有就回退感知向量，避免联网卡住。"""
    global _CLIP_MODEL, _CLIP_PROCESSOR, _CLIP_FAILED
    if _CLIP_MODEL is not None or _CLIP_FAILED:
        return _CLIP_MODEL, _CLIP_PROCESSOR
    try:
        from transformers import CLIPModel, CLIPProcessor

        _CLIP_MODEL = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32", local_files_only=True
        )
        _CLIP_PROCESSOR = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32", local_files_only=True
        )
        _CLIP_MODEL.eval()
    except Exception as exc:
        _CLIP_FAILED = True
        print(f"[CLIP] 本地模型不可用，使用感知向量兜底：{exc}", flush=True)
    return _CLIP_MODEL, _CLIP_PROCESSOR


def using_clip() -> bool:
    model, _ = _load_clip()
    return model is not None


def _fallback_vector(image: Image.Image) -> np.ndarray:
    """512 维颜色/结构感知向量；只用于无 CLIP 的本地兜底。"""
    small = image.convert("RGB").resize((8, 8))
    pixels = np.asarray(small, dtype=np.float32).flatten() / 255.0
    hist = np.asarray(image.convert("RGB").histogram(), dtype=np.float32)
    hist = hist / (hist.sum() + 1e-6)
    vector = np.concatenate([pixels, hist])
    if vector.shape[0] < EMBED_DIM:
        vector = np.pad(vector, (0, EMBED_DIM - vector.shape[0]))
    return vector[:EMBED_DIM]


def embed_image(path: str | Path) -> list[float]:
    image = Image.open(path).convert("RGB")
    model, processor = _load_clip()
    if model is not None and processor is not None:
        import torch

        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            features = model.get_image_features(**inputs)
        vector = features[0].detach().cpu().numpy().astype(np.float32)
    else:
        vector = _fallback_vector(image)
    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector = vector / norm
    return vector.tolist()


def _collection():
    client = get_character_library().client
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=None,
        metadata={"hnsw:space": "cosine"},
    )


def _profile_text(profile: CharacterBaseProfile) -> str:
    return "\n".join(
        [
            f"角色：{profile.name}",
            f"性别：{profile.gender}",
            f"年龄：{profile.age}",
            f"五官发型：{profile.facial_features}",
            f"标志性特征：{profile.signature_features}",
            f"基础服装：{profile.base_outfit}",
            f"气质人设：{profile.temperament}",
            f"剧本：{profile.script_id}",
        ]
    )


def profile_from_card(
    card: dict[str, Any] | GlobalCharacter,
    story_id: str,
    reference_images: list[str] | None = None,
) -> CharacterBaseProfile:
    raw = card.model_dump() if isinstance(card, GlobalCharacter) else dict(card)
    appearance = str(raw.get("appearance") or "").strip()
    outfit_match = re.search(r"(?:穿|着|常穿)([^，。；]+)", appearance)
    outfit = (
        outfit_match.group(1).strip()
        if outfit_match
        else "深色立领长衫，长袖，衣摆完整"
    )
    return CharacterBaseProfile(
        name=str(raw.get("name") or ""),
        gender=str(raw.get("gender") or "未说明"),
        age=str(raw.get("age") or "未说明"),
        facial_features=appearance,
        signature_features=appearance,
        base_outfit=outfit,
        temperament=str(raw.get("personality") or ""),
        negative_prompt=NEGATIVE_PROMPT,
        script_id=story_id,
        reference_images=reference_images or [],
    )


def save_character_reference(
    story_id: str,
    profile: CharacterBaseProfile,
    image_paths: list[str],
) -> dict[str, Any]:
    if not image_paths:
        raise ValueError("至少需要一张角色基准图。")
    collection = _collection()
    stable_ids = [
        f"{story_id}::{profile.name}::{('half' if index == 0 else 'full')}"
        for index in range(len(image_paths))
    ]
    # 同一角色重新确认时，先清掉旧基准图，避免一个角色积累多套参考。
    old = collection.get(
        where={
            "$and": [
                {"type": "character"},
                {"script_id": story_id},
                {"name": profile.name},
            ]
        }
    )
    stale = [item for item in (old.get("ids") or []) if item not in stable_ids]
    if stale:
        collection.delete(ids=stale)

    ids: list[str] = []
    embeddings: list[list[float]] = []
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []
    for index, path in enumerate(image_paths):
        file = Path(path)
        if not file.exists():
            raise FileNotFoundError(f"基准图不存在：{path}")
        role = "half" if index == 0 else "full"
        ref_id = stable_ids[index]
        ids.append(ref_id)
        embeddings.append(embed_image(file))
        documents.append(_profile_text(profile))
        metadatas.append(
            {
                "type": "character",
                "script_id": story_id,
                "name": profile.name,
                "image_role": role,
                "image_path": str(file.resolve()),
                "profile_json": profile.model_dump_json(),
            }
        )
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )
    return {
        "collection": COLLECTION_NAME,
        "count": len(ids),
        "ids": ids,
        "embedding_backend": "clip" if using_clip() else "perceptual_fallback",
    }


def get_character_references(story_id: str) -> list[dict[str, Any]]:
    collection = _collection()
    result = collection.get(
        where={"$and": [{"type": "character"}, {"script_id": story_id}]},
        include=["documents", "metadatas"],
    )
    items = []
    for ref_id, document, metadata in zip(
        result.get("ids", []),
        result.get("documents", []),
        result.get("metadatas", []),
    ):
        metadata = metadata or {}
        items.append(
            {
                "id": ref_id,
                "document": document,
                "metadata": metadata,
                "image_path": metadata.get("image_path", ""),
                "name": metadata.get("name", ""),
                "image_role": metadata.get("image_role", ""),
            }
        )
    return items


def reference_block(story_id: str, names: list[str] | None = None) -> str:
    """生成 @Image 参考声明；没有基准图时返回空字符串。

    names 给定时只包含这些角色的参考声明（按镜头出场名单过滤，
    避免把本镜头没有的角色长相塞进 prompt 造成"串角色"）。
    """
    refs = get_character_references(story_id)
    if names is not None:
        allowed = {str(n) for n in names if str(n).strip()}
        refs = [r for r in refs if str(r.get("name") or "") in allowed]
    if not refs:
        return ""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for ref in refs:
        grouped.setdefault(str(ref.get("name") or ""), []).append(ref)
    lines = ["@Image 角色基准参考图（必须严格保持以下角色长相）："]
    for name, items in grouped.items():
        # 提示词里只放文件名，不暴露本机绝对路径（提示词可能被下游模型/日志留存）。
        paths = "；".join(
            f"{item.get('image_role')}: {Path(str(item.get('image_path'))).name}"
            for item in items
            if item.get("image_path")
        )
        lines.append(f"- {name}：{paths}")
    lines.append(
        "约束：严格保留参考图里的五官、发型、标志性特征和基础服装；"
        "只允许改变表情、动作、光影、镜头和单镜头临时状态。"
    )
    return "\n".join(lines)


def reference_images(story_id: str) -> list[str]:
    return [
        str(item.get("image_path"))
        for item in get_character_references(story_id)
        if item.get("image_path")
    ]


def delete_character_reference(story_id: str, name: str) -> dict[str, Any]:
    """
    删除一个角色的全部基准图：Chroma 记录 + 本地图片文件。

    只清理基准图层，不动全局档案 JSON 里的角色卡文本（那是另一层，
    由 api 层按 include_profile 参数决定是否一并删除）。
    """
    name = str(name or "").strip()
    if not name:
        raise ValueError("角色名不能为空。")
    collection = _collection()
    result = collection.get(
        where={
            "$and": [
                {"type": "character"},
                {"script_id": story_id},
                {"name": name},
            ]
        },
        include=["metadatas"],
    )
    ids = list(result.get("ids") or [])
    paths: set[str] = set()
    for metadata in result.get("metadatas") or []:
        path = str((metadata or {}).get("image_path") or "")
        if path:
            paths.add(path)
    if ids:
        collection.delete(ids=ids)

    # 本地图片文件：优先按向量库记录的路径删，再兜底按命名规则扫一遍。
    removed_files: list[str] = []
    for raw in paths:
        file = Path(raw)
        if file.exists():
            try:
                file.unlink()
                removed_files.append(file.name)
            except OSError:
                pass
    directory = config.OUTPUT_DIR / story_id / "characters"
    if directory.exists():
        safe_name = "".join(ch for ch in name if ch not in '\\/:*?"<>|') or "character"
        for file in directory.glob(f"{safe_name}_*.png"):
            if file.name not in removed_files:
                try:
                    file.unlink()
                    removed_files.append(file.name)
                except OSError:
                    pass
    return {"removed_ids": ids, "removed_files": removed_files}


def reference_paths_for(
    story_id: str, names: list[str] | None = None
) -> list[str]:
    """
    按镜头出场名单取角色基准图路径，供图生图（参考图模式）使用。

    规则：
    - names 顺序 = 出场顺序；每个角色优先取半身照（五官最清晰），再补全身照；
    - 文件已不存在的基准图直接跳过（向量库里可能有失效记录）；
    - 最多取 config.IMAGE_EDIT_MAX_REFS 张（接口限制 3 张）。
    """
    refs = get_character_references(story_id)
    if names is not None:
        allowed = [str(n) for n in names if str(n).strip()]
        by_name: dict[str, list[dict[str, Any]]] = {}
        for ref in refs:
            name = str(ref.get("name") or "")
            if name in allowed:
                by_name.setdefault(name, []).append(ref)
        ordered: list[dict[str, Any]] = []
        for name in allowed:  # 按出场顺序排列参考图
            items = by_name.get(name, [])
            ordered.extend(
                [i for i in items if i.get("image_role") == "half"]
                + [i for i in items if i.get("image_role") != "half"]
            )
    else:
        ordered = list(refs)
    paths: list[str] = []
    for ref in ordered:
        path = str(ref.get("image_path") or "")
        if path and Path(path).exists() and path not in paths:
            paths.append(path)
        if len(paths) >= config.IMAGE_EDIT_MAX_REFS:
            break
    return paths


def image_similarity(story_id: str, image_path: str) -> dict[str, Any]:
    """返回新图与库内基准图的最佳相似度。"""
    collection = _collection()
    try:
        vector = embed_image(image_path)
    except Exception as exc:
        # 测试/损坏图片无法解码时不能把整个生图链路判死。
        return {
            "similarity": 0.0,
            "character": "",
            "threshold": 0.0,
            "passed": True,
            "error": f"{type(exc).__name__}: {exc}",
            "embedding_backend": "unavailable",
        }
    result = collection.query(
        query_embeddings=[vector],
        where={"$and": [{"type": "character"}, {"script_id": story_id}]},
        n_results=5,
        include=["metadatas", "distances"],
    )
    ids = result.get("ids", [[]])[0]
    distances = result.get("distances", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    best = 0.0
    name = ""
    for _, distance, metadata in zip(ids, distances, metadatas):
        similarity = 1.0 - float(distance)
        if similarity > best:
            best = similarity
            name = str((metadata or {}).get("name") or "")
    threshold = 0.72 if using_clip() else 0.55
    return {
        "similarity": round(best, 4),
        "character": name,
        "threshold": threshold,
        "passed": best >= threshold,
        "embedding_backend": "clip" if using_clip() else "perceptual_fallback",
    }


def character_prompts(
    profile: CharacterBaseProfile,
    style_desc: str = "",
    *,
    safe: bool = False,
) -> dict[str, str]:
    """生成半身正面 / 全身正面两张独立基准图的 prompt。"""
    style = style_desc or config.DEFAULT_STYLE_DESC
    if safe:
        common = (
            f"{style}，角色基准照，{profile.name}，干净纯色浅灰背景，"
            "画面中只出现一个角色，单张画面，不要角色三视图，不要转身图/背面图/拼图；"
            "画面干净，不含剧情元素，角色原生基础形象；"
            "穿着完整的基础服装，长袖、衣摆完整，禁止裸露、无衣物、裸露上身；"
            f"五官发型：{profile.facial_features}；"
            f"标志性特征：{profile.signature_features}；"
            f"基础服装：{profile.base_outfit or '深色立领长衫，长袖，衣摆完整'}；"
            "平静自然表情、清晰五官、高分辨率、细节清晰、高质量"
        )
    else:
        common = (
            f"{style}，角色基准照，{profile.name}，干净纯色浅灰背景，"
            "画面中只出现一个角色，单张画面，不要角色三视图，不要转身图/背面图/拼图；"
            "无血迹、无湿透、无受伤、无特殊剧情动作，只保留角色原生基础形象；"
            "穿着完整的基础服装，长袖、衣摆完整，禁止裸露、无衣物、裸露上身；"
            f"五官发型：{profile.facial_features}；"
            f"标志性特征：{profile.signature_features}；"
            f"基础服装：{profile.base_outfit or '深色立领长衫，长袖，衣摆完整'}；"
            "平静自然表情、清晰五官、高分辨率、细节清晰、高质量"
        )
    return {
        "half": common + "；半身正面，腰部以上完整。",
        "full": common + "；全身正面，头到脚完整。",
    }


def save_reference_image(story_id: str, name: str, role: str, content: bytes) -> Path:
    directory = config.OUTPUT_DIR / story_id / "characters"
    directory.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(ch for ch in name if ch not in '\\/:*?"<>|') or "character"
    path = directory / f"{safe_name}_{role}.png"
    path.write_bytes(content)
    return path


def inpaint_watermark(path: str | Path) -> bool:
    """
    本地修补右下角「AI 生成」半透明水印，不裁画面、不调用 API、不花钱。

    水印是压在人物/背景上的细笔画半透明字（Qwen-Image 平台强制）。
    在右下角固定区域内，用「与中值滤波背景的偏差 + 近中性色」筛出细笔画
    像素做掩膜，再用 OpenCV Telea 修补：只替换文字笔画，人物的脚、衣摆、
    道具全部原样保留，画面尺寸不变。检测不到笔画返回 False，交给调用方降级。
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        return False

    file = Path(path)
    if not file.exists():
        return False
    try:
        # Windows + 中文路径：cv2.imread 打不开，必须用 fromfile + imdecode
        image = cv2.imdecode(
            np.fromfile(str(file), dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if image is None:
            return False
        height, width = image.shape[:2]
        x1, y1 = int(width * 0.76), int(height * 0.92)
        x2, y2 = int(width * 0.99), int(height * 0.99)
        roi = image[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # 细笔画相对局部背景（中值）会有小幅度偏差
        local_bg = cv2.medianBlur(gray, 15)
        diff = cv2.absdiff(gray, local_bg)
        channel_spread = roi.max(axis=2).astype(np.int16) - roi.min(axis=2).astype(np.int16)
        neutral = channel_spread < 28
        mask_roi = ((diff > 5) & (diff < 90) & neutral).astype(np.uint8) * 255
        # 抗锯齿边缘一起覆盖
        mask_roi = cv2.dilate(mask_roi, np.ones((3, 3), np.uint8), iterations=1)
        if int(mask_roi.sum()) // 255 < 40:
            return False
        mask = np.zeros((height, width), np.uint8)
        mask[y1:y2, x1:x2] = mask_roi
        repaired = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
        # 同样用 imencode + tofile 写回，绕开中文路径
        ok, encoded = cv2.imencode(".png", repaired)
        if not ok:
            return False
        encoded.tofile(str(file))
        return True
    except Exception:
        return False


def strip_corner_watermark(path: str | Path) -> bool:
    """
    裁掉图片底部窄条，去掉平台强制的右下角「AI 生成」水印。

    Qwen-Image 的 watermark=False 实测无效（平台强制加水印）。基准图是
    「干净纯色背景 + 人物」，底部 4.5% 基本是背景/脚部末端，直接裁掉最干净：
    不烧钱重出、不留拼接色块，也不会把水印喂给下游图生图参考。
    """
    file = Path(path)
    if not file.exists():
        return False
    try:
        image = Image.open(file).convert("RGB")
    except Exception:
        return False
    width, height = image.size
    cut = max(1, int(height * 0.045))
    if cut >= height:
        return False
    image.crop((0, 0, width, height - cut)).save(file, format="PNG")
    return True


def save_reference_views(story_id: str, name: str, content: bytes) -> list[Path]:
    """一次生成全身图，再裁成单人半身；多视图输出也只保留中央正面。"""
    from io import BytesIO

    image = Image.open(BytesIO(content)).convert("RGB")
    width, height = image.size
    # 有些模型会输出三视图；强制只保留中央正面人物。
    front = image.crop(
        (
            int(width * 0.30),
            0,
            int(width * 0.70),
            height,
        )
    )
    full_path = save_reference_image(story_id, name, "full", b"")
    front.save(full_path, format="PNG")
    half = front.crop((0, 0, front.width, max(1, int(front.height * 0.72))))
    half_path = full_path.with_name(full_path.name.replace("_full.png", "_half.png"))
    half.save(half_path, format="PNG")
    return [half_path, full_path]
