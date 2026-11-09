"""
Chroma 人物设定库。

这个文件承担三件事：
1. 提供一个离线可用的本地 embedding（避免首次运行下载模型失败）；
2. 提供人物设定入库、查询、匹配的稳定接口，供节点1/2/4使用；
3. 提供一个小型命令行，方便单独管理人物库。

重要说明：
- 本地哈希 embedding 只适合 Demo 和“姓名/别名精确匹配”场景，语义理解很弱。
- 生产环境应换成 BGE / OpenAI-compatible embedding，并重建 collection。
  换 embedding 后，旧向量不能复用，必须重新 upsert 全部人物。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

from config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    DEMO_CHARACTERS_PATH,
    EMBEDDING_DIM,
    MAX_CANON_CHARS_PER_CHARACTER,
    MAX_HISTORY_CHARS,
    MAX_RECALL_EPISODES,
    SEMANTIC_MATCH_THRESHOLD,
    STORY_COLLECTION_NAME,
    configure_console,
)


# ===========================================================================
# 一、离线 embedding
# ===========================================================================
class LocalHashEmbedding:
    """
    确定性字符 n-gram 哈希 embedding。

    为什么不用 Chroma 默认的 ONNX 模型？
    - 默认模型首次使用会从网上下载，网络不好时 Demo 直接跑不起来；
    - 本 Demo 只有极少量人物，精确姓名/别名匹配已经够用；
    - 哈希实现完全离线、结果稳定，方便测试。

    实现方式：
    - 中文按“单字 + 相邻双字”切分；
    - 每个 token 用 blake2b 哈希到固定维度，带正负号；
    - 最后做 L2 归一化，使余弦距离稳定。

    注意：Python 内置 hash() 每次进程启动会随机化，不能用于持久化向量；
    这里用 hashlib.blake2b，保证重启后向量一致。
    """

    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        self.dim = int(dim)

    # Chroma 1.5.x 要求 embedding function 提供这些方法，否则持久化/查询会报错。
    @staticmethod
    def name() -> str:
        return "local-hash-ngram-v1"

    def get_config(self) -> dict[str, Any]:
        return {"dim": self.dim}

    def is_legacy(self) -> bool:
        # Chroma 1.5.x 在保存 embedding function 配置时会检查这个属性。
        return False

    def default_space(self) -> str:
        return "cosine"

    def supported_spaces(self) -> list[str]:
        return ["cosine"]

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> "LocalHashEmbedding":
        return LocalHashEmbedding(int(config.get("dim", EMBEDDING_DIM)))

    def __call__(self, input: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in input]

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in input]

    def embed_query(self, input: list[str]) -> list[list[float]]:
        # Chroma 查询时会显式调用 embed_query；Protocol 不会自动继承 __call__。
        return [self._embed(text) for text in input]

    def _embed(self, text: str) -> list[float]:
        text = unicodedata.normalize("NFKC", text or "").strip().lower()
        vector = [0.0] * self.dim

        if not text:
            return vector

        # 单字 + 相邻双字。对中文足够轻量，也兼顾英文单词。
        tokens: list[str] = list(text)
        tokens.extend(text[i : i + 2] for i in range(len(text) - 1))
        tokens.extend(word for word in text.split() if len(word) > 1)

        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]


# ===========================================================================
# 二、人物设定库
# ===========================================================================
class CharacterLibrary:
    """
    Chroma 人物设定库的薄封装。

    对外接口（后续接自己的设定库时，保持这几个方法签名即可）：
        upsert_character(card)      单条入库/更新
        upsert_characters(cards)    批量入库
        list_characters()           列出全部人物
        match_characters(names)     按姓名/别名匹配
        semantic_search(text)       向量语义检索
        reset()                     清空并重建集合
    """

    def __init__(
        self,
        persist_dir: str | Path | None = None,
        collection_name: str = COLLECTION_NAME,
        in_memory: bool = False,
    ) -> None:
        self.persist_dir = Path(persist_dir) if persist_dir else CHROMA_DIR
        self.collection_name = collection_name
        self.embedding = LocalHashEmbedding()

        if in_memory:
            # 测试用：不落盘，避免 Windows 下 Chroma 文件句柄导致清理失败。
            self.client = chromadb.EphemeralClient(
                settings=Settings(anonymized_telemetry=False, allow_reset=True),
            )
        else:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            # PersistentClient 会把数据写到 chroma_db/，下次运行仍然存在。
            self.client = chromadb.PersistentClient(
                path=str(self.persist_dir),
                settings=Settings(anonymized_telemetry=False, allow_reset=True),
            )
        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self):
        # hnsw:space=cosine 让 query 返回余弦距离，similarity = 1 - distance。
        return self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # 入库
    # ------------------------------------------------------------------
    def upsert_character(self, character: dict[str, Any]) -> str:
        """
        写入/更新一个人物设定。

        character 至少需要 name。推荐字段见 data/demo_characters.json。
        重复调用同一个 id 是幂等的，不会产生重复记录。
        """
        name = str(character.get("name", "")).strip()
        if not name:
            raise ValueError("人物设定缺少 name 字段，无法入库。")

        character_id = str(character.get("id") or "").strip()
        if not character_id:
            # 没有 id 时按姓名生成稳定 id，避免重复插入。
            character_id = "char_" + hashlib.sha1(name.encode("utf-8")).hexdigest()[:10]

        card = dict(character)
        card["id"] = character_id
        card["name"] = name
        aliases = card.get("aliases") or []
        if isinstance(aliases, str):
            aliases = [item.strip() for item in aliases.split(",") if item.strip()]
        card["aliases"] = aliases

        document = self._build_document(card)
        metadata = {
            "name": name,
            "aliases": ",".join(aliases),
            "source": str(card.get("source", "manual")),
            "version": str(card.get("version", "1.0")),
            # Chroma 的 metadata 只适合放标量；把完整卡片的 JSON 也存进去，
            # 这样 list_characters() 能还原出结构，而不必解析文档文本。
            "card_json": json.dumps(card, ensure_ascii=False),
        }

        self.collection.upsert(
            ids=[character_id],
            documents=[document],
            metadatas=[metadata],
        )
        return character_id

    def upsert_characters(self, characters: list[dict[str, Any]]) -> int:
        """批量入库，返回成功写入的条数。"""
        count = 0
        for character in characters:
            self.upsert_character(character)
            count += 1
        return count

    def _build_document(self, card: dict[str, Any]) -> str:
        """把人物卡拼成一段适合检索/展示的文本。"""
        aliases = "、".join(card.get("aliases") or []) or "无"
        taboos = "；".join(card.get("taboos") or []) or "无"
        examples = "；".join(card.get("examples") or []) or "无"

        return "\n".join(
            [
                f"姓名：{card.get('name', '')}",
                f"别名：{aliases}",
                f"性格：{card.get('personality', '原文未说明')}",
                f"外貌：{card.get('appearance', '原文未说明')}",
                f"说话风格：{card.get('speech_style', '原文未说明')}",
                f"背景：{card.get('background', '原文未说明')}",
                f"人物关系：{card.get('relationships', '原文未说明')}",
                f"禁忌行为：{taboos}",
                f"台词示例：{examples}",
            ]
        )

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    def list_characters(self) -> list[dict[str, Any]]:
        """列出全部人物设定，按姓名排序。"""
        result = self.collection.get(include=["documents", "metadatas"])
        items: list[dict[str, Any]] = []

        for char_id, document, metadata in zip(
            result.get("ids", []),
            result.get("documents", []),
            result.get("metadatas", []),
        ):
            metadata = metadata or {}
            card: dict[str, Any] = {}
            raw_json = metadata.get("card_json")
            if raw_json:
                try:
                    card = json.loads(raw_json)
                except (TypeError, json.JSONDecodeError):
                    card = {}

            # 兼容旧数据：没有 card_json 时用 metadata/document 拼一个最小卡片。
            if not card:
                card = {
                    "id": char_id,
                    "name": metadata.get("name", char_id),
                    "aliases": [
                        item
                        for item in str(metadata.get("aliases", "")).split(",")
                        if item
                    ],
                    "personality": document or "",
                    "source": metadata.get("source", "unknown"),
                    "version": metadata.get("version", "1.0"),
                }
            else:
                card.setdefault("id", char_id)
                card["document"] = document or ""

            items.append(card)

        items.sort(key=lambda item: str(item.get("name", "")))
        return items

    def get_character_count(self) -> int:
        return int(self.collection.count())

    def match_characters(self, names: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
        """
        根据节点1抽取出的人物名，匹配设定库。

        返回：
            matched: 命中的设定卡列表，每张卡额外带 match_type / similarity
            missed:  没有命中的姓名列表

        匹配策略：
        1. 姓名或别名精确匹配（主路径，最可靠）；
        2. 精确匹配失败时，用向量检索兜底，但必须超过保守阈值才采用。
        """
        matched: list[dict[str, Any]] = []
        missed: list[str] = []
        seen_ids: set[str] = set()

        cards = self.list_characters()
        # 建立 姓名/别名 -> 卡片 的索引。
        index: dict[str, dict[str, Any]] = {}
        for card in cards:
            for key in [card.get("name", "")] + list(card.get("aliases") or []):
                normalized = self._normalize_name(str(key))
                if normalized:
                    index[normalized] = card

        for raw_name in names:
            name = str(raw_name).strip()
            if not name:
                continue

            card = index.get(self._normalize_name(name))
            if card is not None:
                card_id = str(card.get("id", card.get("name", "")))
                if card_id not in seen_ids:
                    enriched = dict(card)
                    enriched["match_type"] = "name_or_alias"
                    enriched["similarity"] = 1.0
                    matched.append(enriched)
                    seen_ids.add(card_id)
                continue

            # 精确匹配失败：尝试语义检索。哈希 embedding 下通常不会误命中，
            # 换成真实 embedding 后这里可以识别“阿昭”之类没有登记在别名里的称呼。
            hits = self.semantic_search(name, top_k=1)
            if hits and hits[0]["similarity"] >= SEMANTIC_MATCH_THRESHOLD:
                hit = hits[0]
                card_id = str(hit.get("id", hit.get("name", "")))
                if card_id not in seen_ids:
                    enriched = dict(hit)
                    enriched["match_type"] = "semantic"
                    matched.append(enriched)
                    seen_ids.add(card_id)
                continue

            missed.append(name)

        return matched, missed

    def semantic_search(self, text: str, top_k: int = 3) -> list[dict[str, Any]]:
        """
        向量检索接口（为后续真实 embedding 预留）。

        返回按距离升序的候选卡片，附加 distance / similarity。
        similarity = 1 - cosine_distance。
        """
        if not text.strip() or self.collection.count() == 0:
            return []

        result = self.collection.query(
            query_texts=[text],
            n_results=min(max(1, top_k), self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        hits: list[dict[str, Any]] = []
        for char_id, document, metadata, distance in zip(
            ids, documents, metadatas, distances
        ):
            metadata = metadata or {}
            card: dict[str, Any] = {}
            raw_json = metadata.get("card_json")
            if raw_json:
                try:
                    card = json.loads(raw_json)
                except (TypeError, json.JSONDecodeError):
                    card = {}
            if not card:
                card = {
                    "id": char_id,
                    "name": metadata.get("name", char_id),
                    "personality": document or "",
                }
            card.setdefault("id", char_id)
            card["document"] = document or ""
            card["distance"] = float(distance)
            card["similarity"] = max(0.0, 1.0 - float(distance))
            hits.append(card)

        hits.sort(key=lambda item: item.get("distance", 1e9))
        return hits

    @staticmethod
    def _normalize_name(name: str) -> str:
        """统一全角/半角、大小写和空白，避免“林 昭”匹配不到“林昭”。"""
        normalized = unicodedata.normalize("NFKC", name).strip().casefold()
        return "".join(normalized.split())

    # ------------------------------------------------------------------
    # 初始化 / 重置
    # ------------------------------------------------------------------
    def seed_from_file(self, path: str | Path = DEMO_CHARACTERS_PATH) -> int:
        """从 JSON 文件读取人物卡并入库。"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"找不到人物设定文件：{path}")
        with path.open("r", encoding="utf-8-sig") as file:
            characters = json.load(file)
        if not isinstance(characters, list):
            raise ValueError("人物设定 JSON 必须是数组，例如 [{...}, {...}]")
        return self.upsert_characters(characters)

    def ensure_seeded(self, path: str | Path = DEMO_CHARACTERS_PATH) -> int:
        """
        首次运行时自动注入示例人物。

        只在集合为空时执行，避免每次运行都覆盖用户自己的人物库。
        想强制重新注入，使用 --seed-only 或 --reset-db。
        """
        if self.get_character_count() > 0:
            return 0
        return self.seed_from_file(path)

    def reset(self) -> None:
        """清空集合（仅人物设定，不影响其他 collection）。"""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            # 集合不存在时会抛 NotFoundError；这里不关心，继续重建即可。
            pass
        self.collection = self._get_or_create_collection()


# ===========================================================================
# 三、进程内共享的单例
# ===========================================================================
_DEFAULT_LIBRARY: CharacterLibrary | None = None


def get_character_library() -> CharacterLibrary:
    """
    获取进程内共享的人物库。

    为什么用单例？
    - 节点1、main.py 的 --seed/--reset 应该操作同一个 Chroma 客户端；
    - 避免同一个进程里反复打开 PersistentClient 造成 Windows 文件锁问题。
    """
    global _DEFAULT_LIBRARY
    if _DEFAULT_LIBRARY is None:
        _DEFAULT_LIBRARY = CharacterLibrary()
    return _DEFAULT_LIBRARY


# ===========================================================================
# 四、故事级记忆库（role / summary / script / storyboard）
# ===========================================================================
STORY_DOC_TYPES = ("role", "summary", "script", "storyboard", "story_bible")


class StoryMemoryStore:
    """
    按 story_id 管理“故事级”记忆，和全局角色库 character_settings 分开。

    为什么单独一个 collection？
    - character_settings 是跨故事的全局人设库，负责 OOC 匹配；
    - story_memory 是某个故事自己的快照：本故事的角色、每集摘要、剧本、分镜；
    - 用 metadata 的 story_id + doc_type 过滤，不会污染现有角色匹配逻辑。

    doc_type：
        role      角色人设（本故事内）
        summary   每集剧情摘要
        script    每集完整剧本（修改分支必须用到）
        storyboard 每集分镜表
    """

    def __init__(
        self,
        client: Any | None = None,
        collection_name: str = STORY_COLLECTION_NAME,
        in_memory: bool = False,
    ) -> None:
        if in_memory:
            self.client = chromadb.EphemeralClient(
                settings=Settings(anonymized_telemetry=False, allow_reset=True),
            )
        elif client is not None:
            # 复用 CharacterLibrary 的 client，避免同一进程打开多个 PersistentClient。
            self.client = client
        else:
            self.client = get_character_library().client

        self.embedding = LocalHashEmbedding()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # 过滤条件 / 入库
    # ------------------------------------------------------------------
    @staticmethod
    def _where(story_id: str, doc_type: str | None = None) -> dict[str, Any]:
        """
        Chroma 1.5.9 对多个 metadata 条件要求显式 $and；
        直接写 {"story_id": ..., "doc_type": ...} 会报错。
        """
        clauses: list[dict[str, Any]] = [
            {"story_id": {"$eq": str(story_id)}},
        ]
        if doc_type:
            clauses.append({"doc_type": {"$eq": str(doc_type)}})
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    def upsert_doc(
        self,
        story_id: str,
        doc_type: str,
        content: str,
        episode_no: int = 0,
        title: str = "",
        doc_id: str = "",
        extra_metadata: dict[str, Any] | None = None,
    ) -> str:
        """通用写入接口；doc_type 必须是四种类型之一。"""
        story_id = str(story_id or "").strip()
        doc_type = str(doc_type or "").strip()
        content = str(content or "").strip()

        if not story_id:
            raise ValueError("story_id 不能为空，无法写入故事记忆。")
        if doc_type not in STORY_DOC_TYPES:
            raise ValueError(f"doc_type 必须是 {STORY_DOC_TYPES} 之一，收到：{doc_type!r}")
        if not content:
            raise ValueError(f"{doc_type} 内容为空，拒绝写入。")

        metadata: dict[str, Any] = {
            "story_id": story_id,
            "doc_type": doc_type,
            "episode_no": int(episode_no),
            "title": str(title or ""),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        for key, value in (extra_metadata or {}).items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                metadata[key] = value
            else:
                metadata[key] = json.dumps(value, ensure_ascii=False)

        if not doc_id:
            if doc_type == "role" and metadata.get("name"):
                name_hash = hashlib.sha1(
                    str(metadata["name"]).encode("utf-8")
                ).hexdigest()[:10]
                doc_id = f"{story_id}:role:{name_hash}"
            else:
                doc_id = f"{story_id}:{doc_type}:{int(episode_no)}"

        self.collection.upsert(
            ids=[doc_id],
            documents=[content],
            metadatas=[metadata],
        )
        return doc_id

    def upsert_role(
        self,
        story_id: str,
        character: dict[str, Any],
        episode_no: int = 0,
    ) -> str:
        """写入/更新本故事的一个角色卡；同名角色会合并，不会重复。"""
        name = str(character.get("name", "")).strip()
        if not name:
            raise ValueError("角色卡缺少 name，无法写入 story_memory。")

        existing = self.get_role(story_id, name) or {}
        card = dict(existing)
        card.pop("match_type", None)
        card.update(character)
        card["name"] = name

        aliases = card.get("aliases") or []
        if isinstance(aliases, str):
            aliases = [item.strip() for item in aliases.split(",") if item.strip()]
        card["aliases"] = aliases

        document = self._build_role_document(card)
        extra_metadata = {
            "name": name,
            "aliases": ",".join(str(item) for item in aliases),
            "card_json": json.dumps(card, ensure_ascii=False),
        }
        return self.upsert_doc(
            story_id=story_id,
            doc_type="role",
            content=document,
            episode_no=episode_no,
            title=name,
            extra_metadata=extra_metadata,
        )

    def upsert_episode(
        self,
        story_id: str,
        episode_no: int,
        title: str,
        plot_summary: str,
        script: str,
        storyboard: str,
    ) -> None:
        """把一集的摘要、剧本、分镜一起写入/覆盖。"""
        if int(episode_no) <= 0:
            raise ValueError("episode_no 必须大于 0。")

        for doc_type, content in (
            ("summary", plot_summary),
            ("script", script),
            ("storyboard", storyboard),
        ):
            self.upsert_doc(
                story_id=story_id,
                doc_type=doc_type,
                content=content,
                episode_no=int(episode_no),
                title=title,
            )

    def delete_episode(self, story_id: str, episode_no: int) -> None:
        """删除指定故事、指定集的 summary/script/storyboard。"""
        episode_no = int(episode_no)
        if episode_no <= 0:
            return
        ids = [
            f"{story_id}:{doc_type}:{episode_no}"
            for doc_type in ("summary", "script", "storyboard")
        ]
        self.collection.delete(ids=ids)

    def replace_episode(
        self,
        story_id: str,
        episode_no: int,
        title: str,
        plot_summary: str,
        script: str,
        storyboard: str,
    ) -> None:
        """
        覆盖同一集：先删旧的 summary/script/storyboard，再写入新版本。

        稳定 ID 本身已经可以 upsert 覆盖；这里显式删除，确保旧版本不会残留，
        也满足“同一章多次生成时不要混淆摘要”的要求。
        """
        episode_no = int(episode_no)
        if episode_no <= 0:
            raise ValueError("episode_no 必须大于 0。")
        old_ids = [
            f"{story_id}:{doc_type}:{episode_no}"
            for doc_type in ("summary", "script", "storyboard")
        ]
        try:
            self.collection.delete(ids=old_ids)
        except Exception:
            # 旧记录不存在时删除可能报错，不影响后续 upsert。
            pass
        self.upsert_episode(
            story_id=story_id,
            episode_no=episode_no,
            title=title,
            plot_summary=plot_summary,
            script=script,
            storyboard=storyboard,
        )

    def find_episode_by_title(self, story_id: str, title: str) -> int | None:
        """
        按集标题找已有集数，用于“同一章重新生成”时覆盖旧记录，而不是新开一集。

        只读 summary metadata/正文，不进入 LLM 上下文。
        """
        normalized = str(title or "").strip()
        if not normalized:
            return None
        for doc in self.get_summaries(story_id):
            existing_title = str(doc["metadata"].get("title", "")).strip()
            if existing_title and (
                existing_title == normalized
                or existing_title in normalized
                or normalized in existing_title
            ):
                return int(doc["metadata"].get("episode_no", 0)) or None
        return None

    @staticmethod
    def _build_role_document(card: dict[str, Any]) -> str:
        """角色卡拼成可检索、可展示的文本。"""
        aliases = "、".join(str(item) for item in (card.get("aliases") or [])) or "无"
        taboos = "；".join(str(item) for item in (card.get("taboos") or [])) or "无"
        examples = "；".join(str(item) for item in (card.get("examples") or [])) or "无"
        return "\n".join(
            [
                f"姓名：{card.get('name', '')}",
                f"别名：{aliases}",
                f"身份：{card.get('role', card.get('identity', '原文未说明'))}",
                f"性格：{card.get('personality', '原文未说明')}",
                f"外貌：{card.get('appearance', '原文未说明')}",
                f"说话风格：{card.get('speech_style', '原文未说明')}",
                f"背景：{card.get('background', '原文未说明')}",
                f"人物关系：{card.get('relationships', '原文未说明')}",
                f"关键动作：{'；'.join(str(item) for item in (card.get('key_actions') or [])) or '无'}",
                f"禁忌行为：{taboos}",
                f"台词示例：{examples}",
            ]
        )

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    def get_docs(self, story_id: str, doc_type: str | None = None) -> list[dict[str, Any]]:
        """按 story_id + 可选 doc_type 取原始文档。"""
        result = self.collection.get(
            where=self._where(story_id, doc_type),
            include=["documents", "metadatas"],
        )
        ids = result.get("ids") or []
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []

        docs: list[dict[str, Any]] = []
        for doc_id, document, metadata in zip(ids, documents, metadatas):
            if document is None:
                continue
            docs.append(
                {
                    "id": doc_id,
                    "document": document,
                    "metadata": metadata or {},
                }
            )
        return docs

    def get_roles(self, story_id: str) -> list[dict[str, Any]]:
        """取本故事全部角色卡，按姓名排序。"""
        cards: dict[str, dict[str, Any]] = {}
        for doc in self.get_docs(story_id, "role"):
            metadata = doc["metadata"]
            card: dict[str, Any] = {}
            raw_json = metadata.get("card_json")
            if raw_json:
                try:
                    card = json.loads(raw_json)
                except (TypeError, json.JSONDecodeError):
                    card = {}
            if not card:
                card = {
                    "name": metadata.get("name", "未命名"),
                    "personality": doc["document"],
                }
            card.setdefault("name", metadata.get("name", "未命名"))
            card["match_type"] = "story_memory"
            card.setdefault("episode_no", metadata.get("episode_no", 0))
            cards[str(card["name"])] = card

        return sorted(cards.values(), key=lambda item: str(item.get("name", "")))

    def get_role(self, story_id: str, name: str) -> dict[str, Any] | None:
        """按姓名/别名找单个角色。"""
        normalized = self._normalize_name(name)
        for card in self.get_roles(story_id):
            keys = [card.get("name", "")] + list(card.get("aliases") or [])
            if normalized in {self._normalize_name(str(key)) for key in keys}:
                return card
        return None

    def get_story_bible(self, story_id: str) -> str:
        """读取该故事的长期记忆故事圣经；没有就返回空串。"""
        docs = self.get_docs(story_id, "story_bible")
        if not docs:
            return ""
        docs.sort(key=lambda item: item["metadata"].get("updated_episode", 0))
        return str(docs[-1].get("document") or "")

    def upsert_story_bible(
        self,
        story_id: str,
        content: str,
        updated_episode: int,
    ) -> str:
        """写入/覆盖长期记忆故事圣经（不参与最近 3 集摘要召回）。"""
        return self.upsert_doc(
            story_id=story_id,
            doc_type="story_bible",
            content=content,
            episode_no=0,
            title="长期记忆",
            extra_metadata={"updated_episode": int(updated_episode)},
        )

    def get_summaries(self, story_id: str) -> list[dict[str, Any]]:
        """取全部集摘要，按 episode_no 升序（仅供导入/管理，不用于续写上下文）。"""
        docs = self.get_docs(story_id, "summary")
        docs.sort(key=lambda item: int(item["metadata"].get("episode_no", 0)))
        return docs

    def get_recent_summaries(
        self,
        story_id: str,
        limit: int = MAX_RECALL_EPISODES,
    ) -> list[dict[str, Any]]:
        """
        固定只召回最近 N 集摘要（默认 3），不把全量摘要读进上下文。

        先只读 metadata 算最近集数，再按 episode_no $in 精确取最多 N 条文档。
        """
        limit = max(0, int(limit))
        if limit <= 0:
            return []

        meta_result = self.collection.get(
            where=self._where(story_id, "summary"),
            include=["metadatas"],
        )
        episodes = sorted(
            {
                int(metadata.get("episode_no", 0))
                for metadata in (meta_result.get("metadatas") or [])
                if metadata
            },
            reverse=True,
        )[:limit]
        if not episodes:
            return []

        where = {
            "$and": [
                {"story_id": {"$eq": str(story_id)}},
                {"doc_type": {"$eq": "summary"}},
                {"episode_no": {"$in": episodes}},
            ]
        }
        result = self.collection.get(where=where, include=["documents", "metadatas"])
        docs: list[dict[str, Any]] = []
        for doc_id, document, metadata in zip(
            result.get("ids") or [],
            result.get("documents") or [],
            result.get("metadatas") or [],
        ):
            if document is None:
                continue
            docs.append(
                {
                    "id": doc_id,
                    "document": document,
                    "metadata": metadata or {},
                }
            )
        docs.sort(key=lambda item: int(item["metadata"].get("episode_no", 0)))
        return docs

    def get_latest_episode_no(self, story_id: str) -> int:
        """取该故事已有的最大集数；没有历史返回 0。"""
        latest = 0
        for doc in self.get_docs(story_id):
            if doc["metadata"].get("doc_type") == "role":
                continue
            latest = max(latest, int(doc["metadata"].get("episode_no", 0)))
        return latest

    def get_episode_docs(self, story_id: str, episode_no: int) -> dict[str, dict[str, Any]]:
        """取指定集的 summary/script/storyboard 文档，按 doc_type 索引。"""
        result: dict[str, dict[str, Any]] = {}
        for doc in self.get_docs(story_id):
            if int(doc["metadata"].get("episode_no", 0)) != int(episode_no):
                continue
            doc_type = str(doc["metadata"].get("doc_type", ""))
            if doc_type in {"summary", "script", "storyboard"}:
                result[doc_type] = doc
        return result

    def list_story_ids(self) -> list[str]:
        """列出 story_memory 中出现过的全部 story_id。"""
        result = self.collection.get(include=["metadatas"])
        story_ids = {
            str(metadata.get("story_id"))
            for metadata in (result.get("metadatas") or [])
            if metadata and metadata.get("story_id")
        }
        return sorted(story_ids)

    def build_history_context(
        self,
        story_id: str,
        max_summaries: int = MAX_RECALL_EPISODES,
        max_chars: int = MAX_HISTORY_CHARS,
    ) -> dict[str, Any]:
        """
        组装角色 + 最近 N 集历史摘要（默认 3）。

        只召回最近 N 集，不把全部历史摘要塞进上下文。
        """
        roles = self.get_roles(story_id)
        summaries = self.get_recent_summaries(story_id, limit=max_summaries)
        latest_episode_no = self.get_latest_episode_no(story_id)

        lines: list[str] = ["【角色人设】"]
        if roles:
            for card in roles:
                lines.append(self._format_role_brief(card))
                lines.append("")
        else:
            lines.append("（暂无角色设定，请严格依据历史摘要，不要自行发明设定。）")
            lines.append("")

        lines.append("【最近历史剧情摘要】")
        if summaries:
            for doc in summaries:
                metadata = doc["metadata"]
                episode_no = int(metadata.get("episode_no", 0))
                title = str(metadata.get("title", "") or "")
                lines.append(f"### 第{episode_no}集｜{title}")
                lines.append(str(doc["document"]).strip())
                lines.append("")
        else:
            lines.append("（暂无历史摘要。）")
            lines.append("")

        text = "\n".join(lines).strip()
        if len(text) > max_chars:
            text = text[:max_chars] + "\n……（历史上下文过长，已截断）"

        return {
            "roles": roles,
            "summaries": summaries,
            "latest_episode_no": latest_episode_no,
            "text": text,
        }

    @staticmethod
    def _format_role_brief(card: dict[str, Any]) -> str:
        aliases = "、".join(str(item) for item in (card.get("aliases") or [])) or "无"
        taboos = "；".join(str(item) for item in (card.get("taboos") or [])) or "无"
        return "\n".join(
            [
                f"### {card.get('name', '未命名')}",
                f"- 别名：{aliases}",
                f"- 性格：{card.get('personality', '原文未说明')}",
                f"- 说话风格：{card.get('speech_style', '原文未说明')}",
                f"- 人物关系：{card.get('relationships', '原文未说明')}",
                f"- 禁忌：{taboos}",
            ]
        )

    @staticmethod
    def _normalize_name(name: str) -> str:
        normalized = unicodedata.normalize("NFKC", str(name or "")).strip().casefold()
        return "".join(normalized.split())


_STORY_MEMORY: StoryMemoryStore | None = None


def get_story_memory() -> StoryMemoryStore:
    """获取进程内共享的故事记忆库（复用人物库的 Chroma client）。"""
    global _STORY_MEMORY
    if _STORY_MEMORY is None:
        _STORY_MEMORY = StoryMemoryStore()
    return _STORY_MEMORY


# ===========================================================================
# 五、给提示词使用的人物卡格式化
# ===========================================================================
def format_character_cards(
    cards: list[dict[str, Any]],
    max_chars_per_card: int = MAX_CANON_CHARS_PER_CHARACTER,
) -> str:
    """
    把匹配到的角色卡整理成提示词里的“人物设定”段落。

    只注入命中的角色，不把整个向量库塞进上下文；
    单张卡过长时截断，避免上下文膨胀。
    """
    if not cards:
        return "（本次没有命中已有的人物设定；请严格依据剧情摘要和原文，不要自行发明设定。）"

    blocks: list[str] = []
    for card in cards:
        aliases = "、".join(card.get("aliases") or []) or "无"
        taboos = "；".join(card.get("taboos") or []) or "无"
        examples = "；".join(card.get("examples") or []) or "无"
        block = "\n".join(
            [
                f"### {card.get('name', '未命名')}（匹配方式：{card.get('match_type', 'unknown')}）",
                f"- 别名：{aliases}",
                f"- 性格：{card.get('personality', '原文未说明')}",
                f"- 外貌：{card.get('appearance', '原文未说明')}",
                f"- 说话风格：{card.get('speech_style', '原文未说明')}",
                f"- 背景：{card.get('background', '原文未说明')}",
                f"- 人物关系：{card.get('relationships', '原文未说明')}",
                f"- 禁忌行为：{taboos}",
                f"- 台词示例：{examples}",
            ]
        )
        if len(block) > max_chars_per_card:
            block = block[:max_chars_per_card] + "\n……（人物卡过长，已截断）"
        blocks.append(block)

    return "\n\n".join(blocks)


# ===========================================================================
# 五、独立命令行：管理人物库
# ===========================================================================
def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="管理 AI 漫剧 Demo 的 Chroma 人物设定库")
    parser.add_argument("--seed", action="store_true", help="从 data/demo_characters.json 注入示例人物")
    parser.add_argument("--reset", action="store_true", help="先清空人物集合，再执行其他操作")
    parser.add_argument("--list", action="store_true", help="列出当前人物设定")
    parser.add_argument("--search", type=str, default="", help="语义检索一个人物/关键词")
    parser.add_argument("--db", type=str, default="", help="自定义 Chroma 持久化目录（默认 chroma_db/）")
    return parser


def main() -> int:
    configure_console()
    args = _build_arg_parser().parse_args()
    library = CharacterLibrary(persist_dir=args.db or None)

    if args.reset:
        library.reset()
        print(f"[Chroma] 已清空集合：{COLLECTION_NAME}")

    if args.seed or args.reset:
        count = library.seed_from_file(DEMO_CHARACTERS_PATH)
        print(f"[Chroma] 已注入示例人物 {count} 条：{DEMO_CHARACTERS_PATH}")

    if args.search:
        print(f"[Chroma] 检索：{args.search}")
        for hit in library.semantic_search(args.search, top_k=5):
            print(
                f"  - {hit.get('name')} | 相似度 {hit.get('similarity', 0):.3f} "
                f"| 距离 {hit.get('distance', 0):.3f}"
            )

    if args.list or not any([args.seed, args.reset, args.search]):
        characters = library.list_characters()
        print(f"[Chroma] 当前共 {len(characters)} 个人物：")
        for card in characters:
            print(f"  - {card.get('name')}（id={card.get('id')}）")
            print(f"    性格：{card.get('personality', '')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
