"""
离线测试：不调用 DeepSeek，只验证确定性部分。

运行：
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import unittest
import uuid
import shutil
import asyncio
from pathlib import Path
from unittest.mock import patch

# 让 tests/ 目录下也能直接 import 项目模块。
PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from config import DEMO_CHARACTERS_PATH
from output_importer import parse_output_markdown
from graph import (
    route_after_branch_validation,
    route_after_confirm,
    route_after_continue_retrieval,
    route_after_intent,
    route_after_storyboard,
    route_after_storyboard_retrieval,
    route_after_validation,
)
from render import (
    escape_table_cell,
    normalize_duration,
    render_episode_summary,
    render_final,
    render_plot_summary,
    render_script,
    render_storyboard,
    render_validation,
)
from state import (
    BranchIssue,
    BranchValidationResult,
    ScriptBeat,
    ScriptDraft,
    ScriptScene,
    Shot,
    StoryboardDraft,
)
from vector_store import CharacterLibrary, LocalHashEmbedding, StoryMemoryStore


class TestEmbedding(unittest.TestCase):
    def test_deterministic_and_normalized(self) -> None:
        embedding = LocalHashEmbedding(dim=128)
        first = embedding._embed("林昭推开破庙的门")
        second = embedding._embed("林昭推开破庙的门")
        self.assertEqual(first, second)
        self.assertEqual(len(first), 128)
        norm = sum(value * value for value in first) ** 0.5
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_different_text_differs(self) -> None:
        embedding = LocalHashEmbedding(dim=128)
        self.assertNotEqual(
            embedding._embed("林昭"),
            embedding._embed("苏晚"),
        )


class TestCharacterLibrary(unittest.TestCase):
    def setUp(self) -> None:
        # 用内存版 Chroma，避免测试在 Windows 下留下文件锁和临时目录。
        self.library = CharacterLibrary(in_memory=True)
        self.library.seed_from_file(DEMO_CHARACTERS_PATH)

    def tearDown(self) -> None:
        # 内存版没有文件需要清理；显式断开引用，帮助资源释放。
        self.library = None

    def test_count_and_list(self) -> None:
        self.assertEqual(self.library.get_character_count(), 1)
        cards = self.library.list_characters()
        self.assertEqual(cards[0]["name"], "林昭")
        self.assertIn("外冷内热", cards[0]["personality"])

    def test_match_name_and_alias(self) -> None:
        matched, missed = self.library.match_characters(["林昭", "阿昭", "苏晚"])
        self.assertEqual([card["name"] for card in matched], ["林昭"])
        self.assertEqual(missed, ["苏晚"])
        self.assertEqual(matched[0]["match_type"], "name_or_alias")

    def test_semantic_search_returns_similarity(self) -> None:
        hits = self.library.semantic_search("外冷内热的剑客", top_k=3)
        self.assertTrue(hits)
        self.assertEqual(hits[0]["name"], "林昭")
        self.assertGreaterEqual(hits[0]["similarity"], 0.0)
        self.assertLessEqual(hits[0]["similarity"], 1.0)

    def test_reset(self) -> None:
        self.library.reset()
        self.assertEqual(self.library.get_character_count(), 0)
        self.library.seed_from_file(DEMO_CHARACTERS_PATH)
        self.assertEqual(self.library.get_character_count(), 1)


class TestRender(unittest.TestCase):
    def _script_draft(self) -> ScriptDraft:
        return ScriptDraft(
            title="雨夜破庙",
            logline="一个旧捕快在雨夜护住故人之妹。",
            scenes=[
                ScriptScene(
                    location="破庙",
                    time="夜",
                    characters=["林昭", "苏晚"],
                    beats=[
                        ScriptBeat(kind="旁白", content="雨点砸在残瓦上。"),
                        ScriptBeat(kind="动作描述", content="林昭按住短剑。"),
                        ScriptBeat(kind="台词", speaker="苏晚", content="有人跟着我。"),
                    ],
                )
            ],
        )

    def test_script_tags_and_speaker(self) -> None:
        markdown = render_script(self._script_draft())
        self.assertIn("【旁白】雨点砸在残瓦上。", markdown)
        self.assertIn("【动作描述】林昭按住短剑。", markdown)
        self.assertIn("【台词】苏晚：有人跟着我。", markdown)
        self.assertIn("## 第1场", markdown)

    def test_storyboard_fixed_columns_and_escape(self) -> None:
        draft = StoryboardDraft(
            shots=[
                Shot(
                    duration_seconds=3,
                    visual="破庙外景|雨水\n沿瓦滴落",
                    dialogue="",
                    camera="远景推近",
                    mood="冷蓝",
                ),
                Shot(
                    duration_seconds=0.2,
                    visual="林昭按住剑柄",
                    dialogue="【台词】苏晚：有人跟着我。",
                    camera="中景",
                    mood="压抑",
                ),
            ]
        )
        warnings: list[str] = []
        markdown = render_storyboard(draft, title="雨夜破庙", warnings=warnings)

        self.assertIn("| 镜头序号 | 时长 | 画面描述 | 台词/旁白 | 镜头运镜 | 氛围备注 |", markdown)
        self.assertIn(r"破庙外景\|雨水<br>沿瓦滴落", markdown)
        self.assertIn("| 1 | 3秒 |", markdown)
        self.assertIn("| 2 | 1秒 |", markdown)
        self.assertIn("—", markdown)
        self.assertTrue(warnings)
        self.assertIn("已按 1秒 处理", warnings[0])

    def test_duration_format(self) -> None:
        self.assertEqual(normalize_duration(3.0)[0], "3秒")
        self.assertEqual(normalize_duration(3.5)[0], "3.5秒")
        self.assertEqual(normalize_duration(0.2)[0], "1秒")
        self.assertEqual(normalize_duration(99)[0], "12秒")

    def test_escape_table_cell(self) -> None:
        self.assertEqual(escape_table_cell("a|b\nc"), r"a\|b<br>c")
        self.assertEqual(escape_table_cell(""), "—")

    def test_plot_summary_marks_canon(self) -> None:
        plot_data = {
            "title": "雨夜破庙",
            "one_line_summary": "林昭护住苏晚。",
            "main_plot": "雨夜，苏晚带半枚铜符来到破庙。",
            "characters": [
                {
                    "name": "林昭",
                    "aliases": ["阿昭"],
                    "role": "主角",
                    "personality": "冷静",
                    "appearance": "玄色劲装",
                    "relationships": "与苏晚是旧识",
                    "key_actions": ["藏起苏晚"],
                },
                {
                    "name": "苏晚",
                    "aliases": [],
                    "role": "配角",
                    "personality": "紧张",
                    "appearance": "青衫",
                    "relationships": "林昭旧识",
                    "key_actions": ["求助"],
                },
            ],
            "scenes": [{"location": "破庙", "time": "夜", "atmosphere": "阴冷", "summary": "两人相遇"}],
            "emotional_tone": "冷峻",
            "key_events": ["苏晚求助", "周砚追来"],
            "fidelity_anchors": ["半枚铜符"],
            "unresolved_threads": ["另一半铜符在哪"],
        }
        canon = [
            {
                "name": "林昭",
                "aliases": ["阿昭"],
                "personality": "外冷内热",
                "speech_style": "话少",
                "taboos": ["不会主动示弱"],
                "match_type": "name_or_alias",
            }
        ]
        markdown = render_plot_summary(plot_data, canon, ["苏晚"], truncated=True)
        self.assertIn("设定库校准", markdown)
        self.assertIn("外冷内热", markdown)
        self.assertIn("未命中", markdown)
        self.assertIn("截断", markdown)

    def test_validation_report(self) -> None:
        passed = {
            "passed": True,
            "plot_faithful": True,
            "character_consistent": True,
            "issues": [],
            "summary": "没有实质问题",
        }
        markdown = render_validation(passed, revision_count=0, max_revisions=2)
        self.assertIn("✅ 通过", markdown)

        failed = {
            "passed": False,
            "plot_faithful": False,
            "character_consistent": True,
            "issues": [
                {
                    "category": "剧情偏离",
                    "severity": "严重",
                    "location": "剧本第2场",
                    "description": "新增了原著没有的追杀",
                    "suggestion": "删除新增追杀",
                }
            ],
            "summary": "存在剧情偏离",
        }
        markdown = render_validation(failed, revision_count=1, max_revisions=2)
        self.assertIn("❌ 未通过", markdown)
        self.assertIn("剧情偏离", markdown)
        self.assertIn("删除新增追杀", markdown)


class TestRoute(unittest.TestCase):
    def test_pass_saves(self) -> None:
        state = {
            "validation": {"passed": True},
            "revision_count": 0,
            "max_revisions": 2,
        }
        self.assertEqual(route_after_validation(state), "image_gen_node")

    def test_fail_with_budget_retries(self) -> None:
        state = {
            "validation": {"passed": False},
            "revision_count": 0,
            "max_revisions": 2,
        }
        self.assertEqual(route_after_validation(state), "script_node")

    def test_fail_without_budget_ends(self) -> None:
        state = {
            "validation": {"passed": False},
            "revision_count": 2,
            "max_revisions": 2,
        }
        self.assertEqual(route_after_validation(state), "end")

    def test_intent_routes(self) -> None:
        self.assertEqual(
            route_after_intent({"intent_type": "new_script", "intent_confidence": 0.95}),
            "parse_node",
        )
        self.assertEqual(
            route_after_intent(
                {"intent_type": "continue_story", "intent_confidence": 0.90}
            ),
            "retrieve_continue_node",
        )
        self.assertEqual(
            route_after_intent(
                {"intent_type": "modify_script", "intent_confidence": 0.90}
            ),
            "retrieve_storyboard_node",
        )
        self.assertEqual(
            route_after_intent({"intent_type": "unknown", "intent_confidence": 0.95}),
            "error_node",
        )

    def test_confidence_routes(self) -> None:
        # 0.60 < c < 0.85 → confirm_node
        self.assertEqual(
            route_after_intent(
                {"intent_type": "continue_story", "intent_confidence": 0.72}
            ),
            "confirm_node",
        )
        # <= 0.60 → error_node
        self.assertEqual(
            route_after_intent(
                {"intent_type": "modify_script", "intent_confidence": 0.55}
            ),
            "error_node",
        )

    def test_confirm_routes(self) -> None:
        self.assertEqual(
            route_after_confirm(
                {"confirmed": True, "intent_type": "continue_story"}
            ),
            "retrieve_continue_node",
        )
        self.assertEqual(
            route_after_confirm({"confirmed": False, "intent_type": "modify_script"}),
            "error_node",
        )

    def test_branch_validation_routes(self) -> None:
        self.assertEqual(
            route_after_branch_validation(
                {"branch_validation": {"passed": True}, "intent_type": "continue_story"}
            ),
            "storyboard_node",
        )
        self.assertEqual(
            route_after_branch_validation(
                {"branch_validation": {"passed": False}, "intent_type": "continue_story"}
            ),
            "continue_script_node",
        )
        self.assertEqual(
            route_after_branch_validation(
                {
                    "branch_validation": {"passed": False},
                    "intent_type": "modify_script",
                    "error_message": "max",
                }
            ),
            "error_node",
        )

    def test_retrieval_routes(self) -> None:
        self.assertEqual(route_after_continue_retrieval({}), "continue_script_node")
        self.assertEqual(
            route_after_continue_retrieval({"error_message": "no history"}),
            "error_node",
        )
        self.assertEqual(route_after_storyboard_retrieval({}), "modify_script_node")
        self.assertEqual(
            route_after_storyboard_retrieval({"error_message": "no episode"}),
            "error_node",
        )

    def test_storyboard_routes_by_intent(self) -> None:
        self.assertEqual(route_after_storyboard({"intent_type": "new_script"}), "validate_node")
        self.assertEqual(
            route_after_storyboard({"intent_type": "continue_story"}),
            "image_gen_node",
        )
        self.assertEqual(
            route_after_storyboard({"intent_type": "modify_script"}),
            "image_gen_node",
        )


class TestStoryMemoryStore(unittest.TestCase):
    def setUp(self) -> None:
        # 每个用例使用独立 collection，避免 Chroma EphemeralClient 共享状态。
        self.memory = StoryMemoryStore(
            in_memory=True,
            collection_name=f"test_story_{uuid.uuid4().hex[:8]}",
        )
        self.memory.upsert_role(
            "s1",
            {
                "name": "林晚",
                "personality": "谨慎",
                "speech_style": "短句",
                "relationships": "认识少年",
            },
        )
        self.memory.upsert_episode("s1", 1, "第一夜", "# 摘要1", "# 剧本1", "# 分镜1")
        self.memory.upsert_episode("s1", 2, "第二夜", "# 摘要2", "# 剧本2", "# 分镜2")

    def test_story_ids_and_roles(self) -> None:
        self.assertEqual(self.memory.list_story_ids(), ["s1"])
        roles = self.memory.get_roles("s1")
        self.assertEqual(roles[0]["name"], "林晚")
        self.assertEqual(roles[0]["match_type"], "story_memory")

    def test_latest_and_episode_filter(self) -> None:
        self.assertEqual(self.memory.get_latest_episode_no("s1"), 2)
        docs = self.memory.get_episode_docs("s1", 1)
        self.assertEqual(docs["summary"]["document"], "# 摘要1")
        self.assertEqual(docs["script"]["document"], "# 剧本1")
        self.assertNotIn("storyboard", self.memory.get_episode_docs("s1", 99))

    def test_summaries_sorted_and_context(self) -> None:
        summaries = self.memory.get_summaries("s1")
        self.assertEqual(
            [item["metadata"]["episode_no"] for item in summaries],
            [1, 2],
        )
        context = self.memory.build_history_context("s1")
        self.assertIn("角色人设", context["text"])
        self.assertIn("第2集", context["text"])
        self.assertEqual(context["latest_episode_no"], 2)

    def test_recent_summaries_capped_at_three(self) -> None:
        for episode in range(3, 6):
            self.memory.upsert_episode(
                "s1",
                episode,
                f"第{episode}夜",
                f"# 摘要{episode}",
                f"# 剧本{episode}",
                f"# 分镜{episode}",
            )
        recent = self.memory.get_recent_summaries("s1", limit=3)
        self.assertEqual(
            [item["metadata"]["episode_no"] for item in recent],
            [3, 4, 5],
        )
        context = self.memory.build_history_context("s1")
        self.assertEqual(len(context["summaries"]), 3)
        self.assertNotIn("第1集", context["text"])

    def test_replace_episode_and_title_lookup(self) -> None:
        self.assertEqual(self.memory.find_episode_by_title("s1", "第一夜"), 1)
        self.memory.replace_episode(
            "s1",
            1,
            "第一夜",
            "# 摘要1-新版",
            "# 剧本1-新版",
            "# 分镜1-新版",
        )
        summaries = self.memory.get_summaries("s1")
        self.assertEqual(
            [item["metadata"]["episode_no"] for item in summaries],
            [1, 2],
        )
        self.assertEqual(summaries[0]["document"], "# 摘要1-新版")

    def test_story_bible_roundtrip(self) -> None:
        self.memory.upsert_story_bible("s1", "# 长期故事圣经\n旧铺垫", 2)
        self.assertIn("旧铺垫", self.memory.get_story_bible("s1"))
        # 故事圣经不能被当成最近 3 集摘要
        recent = self.memory.get_recent_summaries("s1", limit=3)
        self.assertEqual(
            [item["metadata"]["episode_no"] for item in recent],
            [1, 2],
        )


class TestContextCompressNode(unittest.TestCase):
    def test_compresses_recent_three_summaries(self) -> None:
        import nodes
        from state import ContextCompression

        class FakeMemory:
            def get_recent_summaries(self, story_id, limit=3):
                return [
                    {
                        "metadata": {"episode_no": i, "title": f"第{i}夜"},
                        "document": f"摘要{i}",
                    }
                    for i in (1, 2, 3)
                ]

            def list_story_ids(self):
                return ["s1"]

            def get_story_bible(self, story_id):
                return ""

        result = ContextCompression(
            synopsis="林昭在破庙继续调查旧案，苏晚同行，铜符线索仍未解决。",
            key_facts=["铜符仍缺一半"],
            open_threads=["周砚为何不杀林昭"],
        )
        with patch("node_context.get_story_memory", return_value=FakeMemory()), patch(
            "node_context.invoke_structured", return_value=result
        ):
            update = nodes.context_compress_node(
                {"raw_text": "续写第三夜", "story_id": "s1"}
            )

        self.assertEqual(update["compressed_source_count"], 3)
        self.assertIn("前情压缩梗概", update["compressed_context"])
        self.assertIn("铜符仍缺一半", update["compressed_context"])
        self.assertEqual(len(update["history_summaries"]), 3)


class TestIntentFallback(unittest.TestCase):
    class _FakeMemory:
        def list_story_ids(self) -> list[str]:
            return []

    def test_llm_failure_becomes_unknown(self) -> None:
        import nodes

        with patch("node_intent.get_story_memory", return_value=self._FakeMemory()), patch(
            "node_intent.invoke_structured",
            side_effect=nodes.NodeError("意图识别", "fake failure"),
        ):
            result = nodes.intent_node({"raw_text": "随便看看"})

        self.assertEqual(result["intent_type"], "unknown")
        self.assertTrue(result["error_message"])


class TestOutputImporter(unittest.TestCase):
    def test_parse_generated_markdown(self) -> None:
        markdown = """# AI 漫剧剧本 & 分镜表

- **章节/集数**：第二章 半枚之外（第2集）
- **故事 ID**：verify_story
- **校验结论**：分支校验：✅ 通过

---

# 剧情摘要｜《第二章 半枚之外》第2集

## 本集主线
林昭与苏晚调查旧案。

## 出场人物
- **林昭**（主角）
  - 性格：外冷内热
  - 外貌：玄衣
  - 关系：与苏晚是旧识

## 场景
1. **破庙**（夜）：两人交谈。

---

# 漫剧剧本｜《第二章 半枚之外》

## 第1场｜破庙｜夜
**出场人物**：林昭、苏晚

【台词】林昭：走。
【台词】苏晚：去哪？

---

# 分镜表｜《第二章 半枚之外》

| 镜头序号 | 时长 | 画面描述 | 台词/旁白 | 镜头运镜 | 氛围备注 |
| --- | --- | --- | --- | --- | --- |
| 1 | 3秒 | 破庙内 | 【台词】林昭：走。 | 中景 | 冷 |
"""
        data = parse_output_markdown(markdown)
        self.assertIsNotNone(data)
        assert data is not None
        self.assertEqual(data["story_id"], "verify_story")
        self.assertEqual(data["episode_no"], 2)
        self.assertEqual(data["title"], "第二章 半枚之外")
        self.assertIn("# 漫剧剧本", data["script"])
        self.assertIn("# 分镜表", data["storyboard"])
        self.assertEqual({item["name"] for item in data["characters"]}, {"林昭", "苏晚"})


class TestBranchValidationNode(unittest.TestCase):
    def _state(self, revision: int = 0) -> dict:
        return {
            "script": "# 剧本",
            "intent_type": "continue_story",
            "canon_profiles": [{"name": "林晚", "personality": "谨慎"}],
            "history_summaries": [
                {"metadata": {"episode_no": 1, "title": "第一夜"}, "document": "# 摘要"}
            ],
            "user_instruction": "续写第二夜",
            "branch_revision_count": revision,
            "branch_max_revisions": 2,
            "branch_validation_history": [],
        }

    def test_minor_issue_does_not_block(self) -> None:
        import nodes

        result = BranchValidationResult(
            passed=False,
            issues=[
                BranchIssue(
                    category="人物OOC",
                    severity="轻微",
                    location="第1场",
                    description="语气略长",
                    suggestion="缩短台词",
                )
            ],
            summary="只有轻微问题",
        )
        with patch("node_branch.invoke_structured", return_value=result):
            update = nodes.branch_validate_node(self._state())
        self.assertTrue(update["branch_validation"]["passed"])
        self.assertEqual(update["branch_feedback"], "")

    def test_severe_issue_returns_feedback(self) -> None:
        import nodes

        result = BranchValidationResult(
            passed=False,
            issues=[
                BranchIssue(
                    category="吃设定",
                    severity="严重",
                    location="第2场",
                    description="提前合拢铜符",
                    suggestion="保留另一半下落不明",
                )
            ],
            summary="严重吃设定",
        )
        with patch("node_branch.invoke_structured", return_value=result):
            update = nodes.branch_validate_node(self._state(revision=0))
        self.assertFalse(update["branch_validation"]["passed"])
        self.assertIn("提前合拢铜符", update["branch_feedback"])
        self.assertNotIn("error_message", update)

    def test_max_revision_blocks_persistence(self) -> None:
        import nodes

        result = BranchValidationResult(
            passed=False,
            issues=[
                BranchIssue(
                    category="剧情矛盾",
                    severity="严重",
                    location="全局",
                    description="冲突",
                    suggestion="修改",
                )
            ],
            summary="未通过",
        )
        with patch("node_branch.invoke_structured", return_value=result):
            update = nodes.branch_validate_node(self._state(revision=2))
        self.assertFalse(update["branch_validation"]["passed"])
        self.assertIn("error_message", update)
        self.assertIn("不会写入 story_memory", update["error_message"])


class TestIntentHelpers(unittest.TestCase):
    def test_retry_prompt_contains_previous_issues(self) -> None:
        import nodes
        from state import ContinueScriptDraft

        captured: dict = {}
        draft = ContinueScriptDraft(
            title="第二夜",
            logline="测试",
            plot_summary="林昭在破庙里继续调查铜符线索，苏晚在旁边整理旧案卷宗，两人暂时没有离开破庙。",
            key_events=["调查"],
            script=ScriptDraft(
                title="第二夜",
                logline="测试",
                scenes=[
                    ScriptScene(
                        location="破庙",
                        time="夜",
                        characters=["林昭"],
                        beats=[ScriptBeat(kind="旁白", content="雨声渐小。")],
                    )
                ],
            ),
        )

        def fake_invoke(**kwargs):
            captured.update(kwargs)
            return draft

        state = {
            "plot_summary": "历史上下文",
            "user_instruction": "续写第二夜",
            "episode_no": 2,
            "branch_feedback": "第1轮问题：提前合拢铜符，属于吃设定。",
            "branch_revision_count": 0,
            "branch_max_revisions": 2,
            "canon_profiles": [{"name": "林昭", "personality": "寡言"}],
            "script": "上一版剧本",
        }
        with patch("node_continue.invoke_structured", side_effect=fake_invoke):
            update = nodes.continue_script_node(state)

        self.assertIn("提前合拢铜符", captured["user_prompt"])
        self.assertIn("上一版剧本", captured["user_prompt"])
        self.assertFalse(captured["use_script_llm"])  # 修复轮用 DeepSeek
        self.assertEqual(update["branch_revision_count"], 1)

    def test_issue_ledger_marks_repeated_and_resolved(self) -> None:
        import nodes

        previous = [
            {
                "id": "I1-1",
                "status": "open",
                "first_round": 1,
                "last_round": 1,
                "repeat_count": 1,
                "category": "吃设定",
                "severity": "严重",
                "location": "第2场",
                "description": "提前合拢铜符",
                "suggestion": "保留另一半下落不明",
                "location_key": "第2场",
                "description_key": "提前合拢铜符",
            }
        ]
        current = [
            {
                "category": "吃设定",
                "severity": "严重",
                "location": "第2场",
                "description": "铜符提前合拢",
                "suggestion": "改掉",
            }
        ]
        ledger, issues = nodes._reconcile_issue_ledger(
            previous, current, 2, False
        )
        self.assertEqual(issues[0]["issue_id"], "I1-1")
        self.assertEqual(issues[0]["status"], "repeated")
        self.assertEqual(ledger[0]["repeat_count"], 2)

        # 下一轮问题消失：旧台账标 resolved，反馈只列新问题。
        new_issues = [
            {
                "category": "人物OOC",
                "severity": "严重",
                "location": "第3场",
                "description": "林昭突然大笑",
                "suggestion": "改成沉默",
            }
        ]
        ledger2, issues2 = nodes._reconcile_issue_ledger(
            ledger, new_issues, 3, False
        )
        self.assertEqual(
            next(item for item in ledger2 if item["id"] == "I1-1")["status"],
            "resolved",
        )
        feedback = nodes._build_branch_feedback(
            {"issues": issues2}, 3, 1, 2, ledger=ledger2
        )
        self.assertIn("已解决，禁止重新引入", feedback)
        self.assertIn("林昭突然大笑", feedback)
        self.assertIn("I1-1", feedback)

    def test_parse_story_id(self) -> None:
        from nodes import _parse_story_id

        self.assertEqual(_parse_story_id("用 story_id=rain_store 续写"), "rain_store")
        self.assertEqual(_parse_story_id("故事ID是 demo_story"), "demo_story")
        self.assertEqual(_parse_story_id("续写第四夜"), "")

    def test_parse_target_episode(self) -> None:
        from nodes import _parse_target_episode

        self.assertEqual(_parse_target_episode("续写第四夜"), 4)
        self.assertEqual(_parse_target_episode("改第12集"), 12)
        self.assertEqual(_parse_target_episode("调整第3话"), 3)
        self.assertIsNone(_parse_target_episode("续写下一集"))

    def test_normalize_yes_no(self) -> None:
        from nodes import _normalize_yes_no

        self.assertTrue(_normalize_yes_no("y"))
        self.assertTrue(_normalize_yes_no("确认"))
        self.assertFalse(_normalize_yes_no("n"))
        self.assertFalse(_normalize_yes_no("随便"))


class TestScriptCleaner(unittest.TestCase):
    def test_duplicate_speaker_names_are_collapsed(self) -> None:
        from script_cleaner import clean_script_text

        text = (
            "# 漫剧剧本\n"
            "【台词】林昭：林昭：你来了。\n"
            "【台词】林昭 林昭：走。\n"
            "【台词】林昭（林昭）：好。\n"
            "【台词】苏晚：正常台词。\n"
            "【旁白】林昭 林昭。"
        )
        cleaned, changed = clean_script_text(text)
        self.assertEqual(changed, 3)
        self.assertIn("【台词】林昭：你来了。", cleaned)
        self.assertIn("【台词】林昭：走。", cleaned)
        self.assertIn("【台词】林昭：好。", cleaned)
        self.assertIn("【台词】苏晚：正常台词。", cleaned)
        self.assertIn("【旁白】林昭 林昭。", cleaned)

    def test_valid_speaker_is_not_changed(self) -> None:
        from script_cleaner import clean_script_text

        cleaned, changed = clean_script_text("【台词】苏晚：林昭，你来了。")
        self.assertEqual(changed, 0)
        self.assertEqual(cleaned, "【台词】苏晚：林昭，你来了。")


class TestShotPrompt(unittest.TestCase):
    def test_parse_storyboard_shots(self) -> None:
        from shot_prompt import parse_storyboard_shots

        board = (
            "| 镜头序号 | 时长 | 画面描述 | 台词/旁白 | 镜头运镜 | 氛围备注 |\n"
            "| --- | --- | --- | --- | --- | --- |\n"
            "| 1 | 5秒 | 雨夜破庙 | 旁白 | 远景 | 冷蓝 |\n"
            "| 2 | 4秒 | 林昭擦剑 | — | 近景 | 暖橙 |\n"
        )
        shots = parse_storyboard_shots(board)
        self.assertEqual([item["shot_no"] for item in shots], [1, 2])
        self.assertIn("林昭擦剑", shots[1]["visual"])

    def test_render_and_clean_prompt(self) -> None:
        from shot_prompt import (
            _clean_positive_prompt,
            render_shot_prompt,
        )

        positive = _clean_positive_prompt(
            "国漫古风，近景构图，林昭擦剑，镜头缓慢推近，时长4秒，电影感构图"
        )
        text = render_shot_prompt(positive, "畸形五官，多手指")
        self.assertEqual(
            text,
            "正向提示词\n国漫古风，林昭擦剑，电影感构图\n---\n负面提示词\n畸形五官，多手指",
        )


class TestImageVideoNodes(unittest.TestCase):
    def test_image_node_prompt_and_save(self) -> None:
        import config
        import image_gen
        from state import ImagePromptDraft

        async def fake_api(positive, negative, style):
            return b"fake-png"

        draft = ImagePromptDraft(
            positive_prompt=(
                "日系漫画风格，林昭左眉旧疤，破庙雨夜关键帧，"
                "电影感构图，清晰五官，高质量"
            ),
            negative_extra=["多手"],
        )
        tmp = Path("outputs/_test_image_node")
        tmp.mkdir(parents=True, exist_ok=True)
        old_output = config.OUTPUT_DIR
        old_url, old_key = config.IMAGE_API_URL, config.IMAGE_API_KEY
        old_prompt_llm = config.IMAGE_PROMPT_USE_LLM
        config.OUTPUT_DIR = tmp
        config.IMAGE_API_URL = "https://example.test/image"
        config.IMAGE_API_KEY = "test-key"
        config.IMAGE_PROMPT_USE_LLM = True
        try:
            with patch("nodes.invoke_structured", return_value=draft), patch(
                "image_gen.external_image_api", side_effect=fake_api
            ):
                update = asyncio.run(image_gen.image_gen_node(
                    {
                        "storyboard": "第1场｜破庙｜夜",
                        "style_desc": "日系漫画",
                        "episode_no": 3,
                        "story_id": "demo_story",
                        "canon_profiles": [
                            {
                                "name": "林昭",
                                "role": "主角",
                                "appearance": "左眉旧疤，无鞘短剑",
                            }
                        ],
                    }
                ))
            self.assertEqual(update["image_status"], "done")
            self.assertTrue(Path(update["image_path"]).exists())
            self.assertIn("畸形五官", update["image_prompt_negative"])
            self.assertIn("多手", update["image_prompt_negative"])
            self.assertEqual(len(update["image_records"]), 1)
        finally:
            config.OUTPUT_DIR = old_output
            config.IMAGE_API_URL, config.IMAGE_API_KEY = old_url, old_key
            config.IMAGE_PROMPT_USE_LLM = old_prompt_llm
            shutil.rmtree(tmp, ignore_errors=True)

    def test_image_node_prompt_only_without_api(self) -> None:
        import config
        import image_gen
        from state import ImagePromptDraft

        draft = ImagePromptDraft(
            positive_prompt=(
                "水墨风格，林昭左眉旧疤，破庙雨夜关键帧，"
                "留白构图，清晰五官，高质量"
            )
        )
        old_url, old_key = config.IMAGE_API_URL, config.IMAGE_API_KEY
        config.IMAGE_API_URL = ""
        config.IMAGE_API_KEY = ""
        try:
            with patch("nodes.invoke_structured", return_value=draft):
                update = asyncio.run(image_gen.image_gen_node(
                    {
                        "storyboard": "第1场｜破庙｜夜",
                        "style_desc": "水墨",
                        "episode_no": 3,
                        "story_id": "demo_story",
                    }
                ))
            self.assertEqual(update["image_status"], "prompt_only")
            self.assertEqual(update["image_path"], "")
        finally:
            config.IMAGE_API_URL, config.IMAGE_API_KEY = old_url, old_key

    def test_video_node_is_reserved(self) -> None:
        import config
        from pathlib import Path
        from video_gen import video_circuit_breaker, video_gen_node

        tmp = Path("outputs/_test_video_node")
        tmp.mkdir(parents=True, exist_ok=True)
        old_output = config.OUTPUT_DIR
        config.OUTPUT_DIR = tmp
        video_circuit_breaker.reset()
        try:
            update = asyncio.run(
                video_gen_node(
                    {
                        "video_prompt": "测试视频提示词",
                        "style_desc": "日系漫画",
                        "story_id": "demo_story",
                        "episode_no": 1,
                    },
                    enable_video=False,
                )
            )
            self.assertEqual(
                set(update),
                {"video_path", "video_prompt", "status", "msg"},
            )
            self.assertEqual(update["status"], "skipped_degrade")
            self.assertEqual(update["video_path"], "")
            self.assertTrue(
                (tmp / "demo_story" / "第1集_video_prompt.txt").exists()
            )
        finally:
            config.OUTPUT_DIR = old_output
            video_circuit_breaker.reset()
            shutil.rmtree(tmp, ignore_errors=True)


class TestVideoCircuitBreaker(unittest.TestCase):
    def test_three_failures_open_half_open_and_recover(self) -> None:
        from video_gen import CircuitBreaker, CircuitState

        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=40)
        for _ in range(3):
            breaker.record_failure()
        self.assertEqual(breaker.state, CircuitState.OPEN)
        self.assertFalse(breaker.allow_request()[0])

        breaker.opened_at -= 41
        allowed, status = breaker.allow_request()
        self.assertTrue(allowed)
        self.assertEqual(status, "HALF_OPEN")
        self.assertFalse(breaker.allow_request()[0])

        breaker.record_success()
        self.assertEqual(breaker.state, CircuitState.CLOSED)
        self.assertEqual(breaker.failure_count, 0)

    def test_half_open_failure_reopens(self) -> None:
        from video_gen import CircuitBreaker, CircuitState

        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=40)
        for _ in range(3):
            breaker.record_failure()
        breaker.opened_at -= 41
        self.assertTrue(breaker.allow_request()[0])
        breaker.record_failure()
        self.assertEqual(breaker.state, CircuitState.OPEN)

    def test_api_failure_degrades_and_counts(self) -> None:
        import config
        from pathlib import Path
        from video_gen import video_circuit_breaker, video_gen_node

        tmp = Path("outputs/_test_video_fail")
        tmp.mkdir(parents=True, exist_ok=True)
        old_output = config.OUTPUT_DIR
        config.OUTPUT_DIR = tmp
        video_circuit_breaker.reset()
        try:
            with patch(
                "video_gen.call_seedance_async",
                side_effect=RuntimeError("quota exceeded"),
            ):
                update = asyncio.run(
                    video_gen_node(
                        {
                            "video_prompt": "测试视频提示词",
                            "style_desc": "日系漫画",
                            "story_id": "demo_story",
                            "episode_no": 2,
                        }
                    )
                )
            self.assertEqual(update["status"], "fail_degrade")
            self.assertEqual(video_circuit_breaker.failure_count, 1)
            self.assertTrue(
                (tmp / "demo_story" / "第2集_video_prompt.txt").exists()
            )
        finally:
            config.OUTPUT_DIR = old_output
            video_circuit_breaker.reset()
            shutil.rmtree(tmp, ignore_errors=True)

    def test_circuit_open_skips_api(self) -> None:
        import config
        from pathlib import Path
        from video_gen import video_circuit_breaker, video_gen_node

        tmp = Path("outputs/_test_video_open")
        tmp.mkdir(parents=True, exist_ok=True)
        old_output = config.OUTPUT_DIR
        config.OUTPUT_DIR = tmp
        video_circuit_breaker.reset()
        for _ in range(3):
            video_circuit_breaker.record_failure()
        try:
            with patch("video_gen.call_seedance_async") as api:
                update = asyncio.run(
                    video_gen_node(
                        {
                            "video_prompt": "测试视频提示词",
                            "style_desc": "日系漫画",
                            "story_id": "demo_story",
                            "episode_no": 3,
                        }
                    )
                )
            self.assertEqual(update["status"], "circuit_open")
            api.assert_not_called()
            self.assertTrue(
                (tmp / "demo_story" / "第3集_video_prompt.txt").exists()
            )
        finally:
            config.OUTPUT_DIR = old_output
            video_circuit_breaker.reset()
            shutil.rmtree(tmp, ignore_errors=True)


class TestVideoPromptBuilder(unittest.TestCase):
    def test_prompt_uses_episode_summary_and_storyboard_only(self) -> None:
        from video_prompt import build_episode_video_prompt

        summary = (
            "# 剧情摘要｜测试\n\n## 本集主线\n"
            "林昭在破庙遇到苏晚。\n\n## 出场人物\n"
            "- 林昭：左眉旧疤\n"
        )
        storyboard = (
            "| 镜头序号 | 时长 | 画面描述 | 台词/旁白 | 镜头运镜 | 氛围备注 |\n"
            "| --- | --- | --- | --- | --- | --- |\n"
            "| 1 | 4秒 | 林昭擦剑 | — | 近景 | 冷蓝 |\n"
            "| 2 | 3秒 | 苏晚推门 | — | 中景 | 雨夜 |\n"
        )
        prompt = build_episode_video_prompt(summary, storyboard, "水墨", 1, 2)
        self.assertIn("镜头1：林昭擦剑", prompt)
        self.assertIn("镜头2：苏晚推门", prompt)
        self.assertNotIn("## 出场人物", prompt)
        self.assertNotIn("左眉旧疤", prompt)

    def test_range_validation(self) -> None:
        from video_prompt import build_episode_video_prompt

        board = (
            "| 镜头序号 | 时长 | 画面描述 | 台词/旁白 | 镜头运镜 | 氛围备注 |\n"
            "| --- | --- | --- | --- | --- | --- |\n"
            "| 1 | 4秒 | 镜头一 | — | 近景 | 冷蓝 |\n"
            "| 2 | 3秒 | 镜头二 | — | 中景 | 雨夜 |\n"
        )
        prompt = build_episode_video_prompt("", board, "水墨", 1, 1)
        self.assertIn("镜头1：镜头一", prompt)
        self.assertNotIn("镜头2：镜头二", prompt)
        with self.assertRaises(ValueError) as ctx:
            build_episode_video_prompt("", board, "水墨", 1, 3)
        self.assertEqual(str(ctx.exception), "本集一共2个镜头，请重新选择区间")

    def test_empty_episode_raises_friendly_error(self) -> None:
        from video_prompt import build_episode_video_prompt

        with self.assertRaises(ValueError) as ctx:
            build_episode_video_prompt("", "", "水墨")
        self.assertIn("该集暂无分镜数据", str(ctx.exception))


class TestScriptLlmFallback(unittest.TestCase):
    def test_429_falls_back_to_pro_model(self) -> None:
        import nodes
        from pydantic import BaseModel

        class Tiny(BaseModel):
            answer: str

        class FakeStructured:
            def __init__(self, should_429: bool) -> None:
                self.should_429 = should_429

            def invoke(self, messages):
                if self.should_429:
                    raise RuntimeError("429 Too Many Requests")
                return {
                    "parsed": Tiny(answer="ok"),
                    "raw": None,
                    "parsing_error": None,
                }

        class FakeLlm:
            def __init__(self, should_429: bool) -> None:
                self.should_429 = should_429

            def with_structured_output(self, schema, include_raw=True):
                return FakeStructured(self.should_429)

        calls: list[str] = []

        def fake_get_script_llm(temperature, model=None, max_retries=None):
            calls.append(model or "free")
            return FakeLlm(model != __import__("config").SCRIPT_MODEL_PRO)

        with patch("node_shared.config.get_script_llm", side_effect=fake_get_script_llm):
            result = nodes.invoke_structured(
                node_name="测试剧本",
                schema=Tiny,
                system_prompt="固定 system",
                user_prompt="动态 user",
                temperature=0.0,
                use_script_llm=True,
            )

        self.assertEqual(result.answer, "ok")
        self.assertEqual(len(calls), 2)
        self.assertNotEqual(calls[0], calls[1])

    def test_script_system_prefix_is_fixed(self) -> None:
        import prompts

        for system in (
            prompts.SCRIPT_SYSTEM,
            prompts.CONTINUE_SCRIPT_SYSTEM,
            prompts.MODIFY_SCRIPT_SYSTEM,
        ):
            self.assertTrue(system.startswith(prompts.SCRIPT_FIXED_PREFIX))

    def test_recall_summary_goes_to_user_not_system(self) -> None:
        import prompts

        user_prompt = prompts.build_continue_script_prompt(
            history_context="【召回摘要XYZZY】林昭在破庙。",
            canon_text="林昭：前捕快。",
            instruction="续写第四章",
            episode_no=4,
        )
        self.assertIn("召回摘要XYZZY", user_prompt)
        self.assertNotIn("召回摘要XYZZY", prompts.CONTINUE_SCRIPT_SYSTEM)


class TestIssueMemory(unittest.TestCase):
    def test_branch_feedback_accumulates_history(self) -> None:
        import nodes

        history = [
            {
                "issues": [
                    {
                        "category": "人物OOC",
                        "severity": "严重",
                        "location": "第1场",
                        "description": "旧问题A：林昭话太多",
                        "suggestion": "缩短台词",
                    }
                ]
            }
        ]
        current = {
            "issues": [
                {
                    "category": "吃设定",
                    "severity": "严重",
                    "location": "第2场",
                    "description": "新问题B：提前合拢铜符",
                    "suggestion": "保留悬念",
                }
            ]
        }
        feedback = nodes._build_branch_feedback(current, 2, 1, 2, history)
        self.assertIn("旧问题A", feedback)
        self.assertIn("新问题B", feedback)
        self.assertIn("I1-1", feedback)
        self.assertIn("DeepSeek 修复轮", feedback)

    def test_new_script_feedback_accumulates_history(self) -> None:
        import nodes

        history = [
            {
                "issues": [
                    {
                        "category": "剧情偏离",
                        "severity": "严重",
                        "location": "第1场",
                        "description": "旧问题A",
                        "suggestion": "删除",
                    }
                ]
            }
        ]
        current = {
            "issues": [
                {
                    "category": "人物OOC",
                    "severity": "严重",
                    "location": "第2场",
                    "description": "新问题B",
                    "suggestion": "修改",
                }
            ]
        }
        feedback = nodes._build_feedback(current, 2, 1, 2, history)
        self.assertIn("旧问题A", feedback)
        self.assertIn("新问题B", feedback)
        self.assertIn("历史·第1轮", feedback)

    def test_branch_validator_prompt_contains_previous_issues(self) -> None:
        import nodes

        captured: dict = {}
        result = BranchValidationResult(passed=True, issues=[], summary="ok")

        def fake_invoke(**kwargs):
            captured.update(kwargs)
            return result

        state = {
            "script": "# 剧本",
            "intent_type": "continue_story",
            "canon_profiles": [],
            "history_summaries": [],
            "user_instruction": "续写",
            "branch_validation_history": [
                {
                    "issues": [
                        {
                            "category": "人物OOC",
                            "severity": "严重",
                            "location": "第1场",
                            "description": "旧问题A",
                            "suggestion": "修改",
                        }
                    ]
                }
            ],
            "branch_revision_count": 1,
            "branch_max_revisions": 2,
        }
        with patch("node_branch.invoke_structured", side_effect=fake_invoke):
            nodes.branch_validate_node(state)
        self.assertIn("旧问题A", captured["user_prompt"])
        self.assertIn("历史问题台账", captured["user_prompt"])


class TestGlobalAndShotPipeline(unittest.TestCase):
    def test_global_profile_cached(self) -> None:
        import config
        import shutil
        from pathlib import Path
        from global_profile import ensure_global_profile, profile_path
        from state import GlobalCharacter, GlobalProfileDraft

        tmp = Path("data/_test_global_profile")
        old_data = config.DATA_DIR
        config.DATA_DIR = tmp
        draft = GlobalProfileDraft(
            worldview="古代临安城，捕快与旧案交织的悬疑世界；官府与江湖势力并存。",
            characters=[
                GlobalCharacter(
                    name="林昭",
                    role="主角",
                    appearance="左眉旧疤，无鞘短剑",
                    personality="冷静寡言",
                )
            ],
        )
        try:
            with patch("global_profile.invoke_structured", return_value=draft) as llm:
                first, path, cached = ensure_global_profile("s1", "原文", "测试")
                self.assertFalse(cached)
                self.assertTrue(path.exists())
                self.assertIn("林昭", path.read_text(encoding="utf-8"))
                second, _, cached2 = ensure_global_profile("s1", "原文", "测试")
                self.assertTrue(cached2)
                self.assertEqual(first["worldview"], second["worldview"])
                self.assertEqual(llm.call_count, 1)
        finally:
            config.DATA_DIR = old_data
            shutil.rmtree(tmp, ignore_errors=True)

    def test_shot_generate_cache_and_json(self) -> None:
        import config
        import shutil
        from pathlib import Path
        from node_shot_generate import shot_generate_node
        from state import EpisodeShot, EpisodeShotsDraft

        tmp = Path("outputs/_test_shot_generate")
        old_output = config.OUTPUT_DIR
        config.OUTPUT_DIR = tmp
        draft = EpisodeShotsDraft(
            episode_tone="冷峻悬疑",
            shots=[
                EpisodeShot(
                    shot_no=1,
                    duration_seconds=4,
                    visual="林昭在破庙擦剑",
                    camera="近景",
                    mood="冷蓝",
                    characters=["林昭"],
                ),
                EpisodeShot(
                    shot_no=2,
                    duration_seconds=3,
                    visual="苏晚推门",
                    camera="中景",
                    mood="雨夜",
                    characters=["苏晚"],
                ),
            ],
        )
        state = {
            "story_id": "s1",
            "episode_no": 1,
            "title": "第一集",
            "plot_summary": "林昭在破庙遇到苏晚。",
            "global_profile": {
                "characters": [{"name": "林昭"}, {"name": "苏晚"}]
            },
        }
        try:
            with patch("node_shot_generate.invoke_structured", return_value=draft) as llm:
                first = shot_generate_node(state)
                self.assertTrue(first["shot_file"].endswith("ep_01.json"))
                self.assertEqual(len(first["shots"]), 2)
                self.assertEqual(first["episode_tone"], "冷峻悬疑")
                self.assertTrue(Path(first["shot_file"]).exists())
                second = shot_generate_node(state)
                self.assertTrue(second["shots_cached"])
                self.assertEqual(llm.call_count, 1)
        finally:
            config.OUTPUT_DIR = old_output
            shutil.rmtree(tmp, ignore_errors=True)

    def test_global_profile_filters_temporary_state(self) -> None:
        from global_profile import sanitize_profile

        profile = sanitize_profile(
            {
                "characters": [
                    {
                        "name": "林昭",
                        "appearance": "左眉旧疤，随身无鞘短剑，衣角湿透，肩头有血",
                        "personality": "冷静寡言，紧张",
                    }
                ]
            }
        )
        card = profile["characters"][0]
        self.assertIn("左眉旧疤", card["appearance"])
        self.assertIn("无鞘短剑", card["appearance"])
        self.assertNotIn("湿透", card["appearance"])
        self.assertNotIn("血", card["appearance"])
        self.assertNotIn("紧张", card["personality"])

    def test_build_video_prompt_node_slices_shots(self) -> None:
        import config
        import shutil
        from pathlib import Path
        from node_video_prompt import build_video_prompt_node
        from shot_store import render_storyboard_markdown

        tmp = Path("outputs/_test_video_prompt_node")
        old_output = config.OUTPUT_DIR
        config.OUTPUT_DIR = tmp
        shots = [
            {"shot_no": 1, "duration_seconds": 3, "visual": "镜头一", "dialogue": "", "camera": "中景", "mood": ""},
            {"shot_no": 2, "duration_seconds": 3, "visual": "镜头二", "dialogue": "", "camera": "中景", "mood": ""},
            {"shot_no": 3, "duration_seconds": 3, "visual": "镜头三", "dialogue": "", "camera": "中景", "mood": ""},
        ]
        state = {
            "story_id": "s1",
            "episode_no": 1,
            "shots": shots,
            "storyboard": render_storyboard_markdown(shots, "测试"),
            "style_desc": "水墨",
            "start_shot": 2,
            "end_shot": 3,
        }
        try:
            result = build_video_prompt_node(state)
            self.assertEqual(result["video_prompt_start"], 2)
            self.assertEqual(result["video_prompt_end"], 3)
            self.assertIn("镜头2：镜头二", result["video_prompt"])
            self.assertNotIn("镜头1：镜头一", result["video_prompt"])
            self.assertIn("运镜", result["video_prompt"])
            self.assertIn("氛围", result["video_prompt"])
        finally:
            config.OUTPUT_DIR = old_output
            shutil.rmtree(tmp, ignore_errors=True)

    def test_new_script_branch_validation_uses_shot_ooc(self) -> None:
        import nodes
        from graph import route_after_branch_validation
        from state import BranchValidationResult

        result = BranchValidationResult(passed=True, issues=[], summary="ok")
        state = {
            "intent_type": "new_script",
            "story_id": "s1",
            "episode_no": 1,
            "script": "# 剧本",
            "plot_summary": "本集主线",
            "global_profile": {"characters": [{"name": "林昭"}]},
            "shots": [{"shot_no": 1, "visual": "林昭擦剑"}],
        }
        with patch("node_shot_validate.invoke_structured", return_value=result):
            update = nodes.branch_validate_node(state)
        self.assertTrue(update["branch_validation"]["passed"])
        self.assertEqual(
            route_after_branch_validation(
                {"intent_type": "new_script", "shots": state["shots"], **update}
            ),
            "build_video_prompt_node",
        )


class TestBranchRender(unittest.TestCase):
    def test_episode_summary(self) -> None:
        markdown = render_episode_summary(
            "第二夜",
            2,
            "少年再次出现，林晚发现反字糖。",
            ["少年出现", "林晚发现反字糖"],
            change_summary="",
        )
        self.assertIn("第2集", markdown)
        self.assertIn("关键事件", markdown)

    def test_final_without_validation(self) -> None:
        markdown = render_final(
            {
                "intent_type": "continue_story",
                "story_id": "s1",
                "episode_no": 2,
                "title": "第二夜",
                "plot_summary": "# 摘要2",
                "script": "# 剧本2",
                "storyboard": "# 分镜2",
                "persist_status": "已保存第2集",
                "trace": ["续写完成"],
            },
            model_name="test",
        )
        self.assertIn("未执行", markdown)
        self.assertNotIn("❌ 未通过", markdown)
        self.assertIn("已保存第2集", markdown)

    def test_final_with_branch_validation(self) -> None:
        markdown = render_final(
            {
                "intent_type": "modify_script",
                "story_id": "s1",
                "episode_no": 1,
                "title": "第一夜",
                "plot_summary": "# 摘要",
                "script": "# 剧本",
                "storyboard": "# 分镜",
                "branch_validation": {
                    "passed": True,
                    "issues": [],
                    "summary": "没有严重问题",
                },
                "branch_revision_count": 1,
                "branch_max_revisions": 2,
                "persist_status": "已保存第1集",
                "trace": ["修改完成"],
            },
            model_name="test",
        )
        self.assertIn("剧情忠实度校验（续写/修改分支）", markdown)
        self.assertIn("分支校验：✅ 通过", markdown)
        self.assertIn("1 / 2", markdown)


if __name__ == "__main__":
    unittest.main()
