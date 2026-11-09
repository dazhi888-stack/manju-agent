"""
LangGraph 状态结构 + 各节点使用的 Pydantic 结构。

为什么要分两层？
- ComicState 是“图的状态”，只放字符串、dict、list，方便打印和排查。
- Pydantic 模型只在节点内部用来约束 LLM 输出；拿到结果后转成 dict / Markdown 再写回状态。
  这样状态不会被一堆复杂对象塞满，新手用 print(state) 也能看懂。
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field, model_validator


# ===========================================================================
# 一、LangGraph 全局状态
# ===========================================================================
class ComicState(TypedDict, total=False):
    """
    整张图共享的状态。

    total=False 表示字段可以不存在，节点用 state.get(...) 读取，
    这样初始状态只需要传 raw_text / max_revisions 等少数字段。
    """

    # ---- 输入 ----
    raw_text: str                  # 用户输入：新剧本正文，或续写/修改指令
    input_chars: int               # 原文长度（字数）
    input_truncated: bool          # 是否因为超过 MAX_INPUT_CHARS 被截断

    # ---- 意图识别（新增，图的最开始） ----
    story_id: str                  # 故事 ID，用于 story_memory 过滤
    user_query: str                # 用户原始输入，interrupt 暂停时随 state 保存
    intent_type: str               # new_script / continue_story / modify_script / unknown
    intent_confidence: float
    intent_reason: str
    intent_error: str
    user_instruction: str          # 续写/修改分支的原始指令
    target_episode: int | None     # 用户明确提到的第几集/第几夜
    pending_intent: str            # 等待用户确认的 intent
    pending_confidence: float      # 等待用户确认时的 confidence
    needs_confirmation: bool       # 中等置信度，等待用户二次确认
    confirmed: bool                # 用户确认结果
    confirm_answer: str            # 用户原始确认回复
    confirmation_question: str     # confirm_node 输出的话术

    # ---- 故事记忆检索（续写/修改分支） ----
    episode_no: int
    compressed_context: str        # context_compress_node 生成的短梗概
    compressed_story_id: str       # 压缩上下文对应的 story_id
    compressed_source_count: int   # 实际召回的摘要条数（最多 3）
    long_term_context: str         # 跨集长期故事圣经（不占最近 3 集名额）
    story_bible_chars: int
    history_roles: list[dict]
    history_summaries: list[dict]
    history_script: str
    history_storyboard: str       # 兼容旧字段；完整分镜只存文件，不再进入 LLM 上下文

    # ---- 续写/修改分支的剧情忠实度校验 ----
    branch_validation: dict
    branch_validation_history: Annotated[list[dict], operator.add]
    branch_issue_ledger: list[dict]   # 跨轮问题台账：id/status/首次与最后出现轮次
    branch_feedback: str
    branch_revision_count: int
    branch_max_revisions: int

    # ---- 修改/持久化 ----
    change_summary: str
    persist_status: str
    error_message: str

    # ---- 节点1：原文解析 ----
    title: str                     # 章节标题（模型抽取，用于文件名/报告）
    plot_data: dict                # 结构化剧情摘要
    plot_summary: str              # 渲染后的 Markdown 剧情摘要
    canon_profiles: list[dict]     # 本地全局档案角色卡（旧故事回退 Chroma）
    canon_missed: list[str]        # 没有命中设定库的角色名

    # ---- 节点2：剧本改写 ----
    script: str                    # 漫剧剧本 Markdown

    # ---- 节点3：分镜生成 ----
    storyboard: str                # 分镜表 Markdown（兼容现有渲染/校验）
    shots: list[dict]              # shot_generate_node 的结构化镜头列表
    shot_file: str                 # outputs/<story_id>/shots/ep_XX.json
    shots_cached: bool             # 是否直接命中本地镜头 JSON
    episode_tone: str              # 本集情绪基调

    # ---- 全局角色档案（本地 JSON，优先于 Chroma） ----
    global_profile: dict           # 全局角色 + 世界观
    global_profile_path: str
    global_profile_cached: bool

    # ---- 视频提示词切片 ----
    start_shot: int
    end_shot: int
    video_prompt_start: int
    video_prompt_end: int
    shot_feedback: str             # 分镜 OOC 校验失败时给 shot_generate 的意见

    # ---- 文生图 / 文生视频 ----
    style_desc: str                # 用户指定画风，例如 日系漫画/厚涂/水墨
    image_negative_extra: str      # 用户额外负面词
    image_prompt_positive: str
    image_prompt_negative: str
    image_path: str
    image_status: str             # done / prompt_only / skipped / failed
    image_retry_count: int
    image_error: str
    image_gen_ms: float
    image_records: list[dict]     # 每张图的 prompt/路径/耗时/状态
    video_prompt: str
    video_path: str
    video_status: str             # reserved / done / skipped / failed
    video_gen_ms: float
    video_force: bool             # 菜单 10 强制调用 Seedance（可能收费）

    # ---- 节点4：校验 ----
    validation: dict               # 最新一轮校验结果（含 passed）
    validation_history: Annotated[list[dict], operator.add]  # 每轮都追加
    feedback: str                  # 最新一轮给节点2/3的修改意见
    revision_count: int            # 已经完成了几次“剧本重写”
    max_revisions: int             # 最多允许重写几次

    # ---- 审计 ----
    trace: Annotated[list[str], operator.add]  # 每个节点的简短执行记录


# ===========================================================================
# 二、意图识别结构
# ===========================================================================
class IntentResult(BaseModel):
    """意图识别的强制 JSON 结构，只允许三个字段。"""

    intent: Literal["new_script", "continue_story", "modify_script", "unknown"] = Field(
        description=(
            "new_script=粘贴完整小说/剧本；continue_story=续写/下一集；"
            "modify_script=修改已有剧情或镜头；unknown=无法判断"
        )
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="判断置信度，0~1；不确定就返回 unknown",
    )
    reason: str = Field(description="一句话说明判断依据")


# 兼容你现有代码里使用的 AgentState 名称。
AgentState = ComicState


class StoryBible(BaseModel):
    """跨集长期记忆：已发生主线、人物状态、关键物件、未解决线索。"""

    synopsis: str = Field(
        min_length=50,
        max_length=1500,
        description="按时间顺序保留的核心主线，最多 1500 字",
    )
    character_states: list[str] = Field(
        default_factory=list,
        max_length=12,
        description="主要人物当前状态/关系/已知信息，最多 12 条",
    )
    key_objects: list[str] = Field(
        default_factory=list,
        max_length=12,
        description="关键物件、地点、组织、设定，最多 12 条",
    )
    plot_threads: list[str] = Field(
        default_factory=list,
        max_length=12,
        description="尚未解决的伏笔/悬念/目标，最多 12 条",
    )
    timeline: list[str] = Field(
        default_factory=list,
        max_length=12,
        description="关键事件时间线，最多 12 条",
    )


class ContextCompression(BaseModel):
    """context_compress_node 的输出：多条历史摘要压缩成短梗概。"""

    synopsis: str = Field(
        min_length=20,
        max_length=800,
        description="合并后的前情梗概，300~500字，不新增原文没有的事实；最多 800 字",
    )
    key_facts: list[str] = Field(
        default_factory=list,
        max_length=8,
        description="续写时必须保持的关键事实，例如人物关系、已发生结果、关键物件；最多 8 条",
    )
    open_threads: list[str] = Field(
        default_factory=list,
        max_length=8,
        description="尚未解决的悬念，供续写承接；最多 8 条",
    )


# ===========================================================================
# 三、节点1：原文解析结构
# ===========================================================================
class CharacterInPlot(BaseModel):
    """从原文中抽取的一个人物。"""

    name: str = Field(description="人物姓名，必须使用原文中的写法")
    aliases: list[str] = Field(
        default_factory=list,
        description="原文中出现过的别名、称呼，例如“阿昭”“昭哥”；没有就返回空列表",
    )
    role: str = Field(description="在本次章节中的身份/定位，例如主角、配角、反派、路人")
    personality: str = Field(description="根据本章言行推断的性格；不要编造原文没有的特征")
    appearance: str = Field(description="外貌、衣着、体态等可见特征；原文未说明就写“原文未说明”")
    relationships: str = Field(description="与其他出场人物的关系；原文未说明就写“原文未说明”")
    key_actions: list[str] = Field(
        default_factory=list,
        description="该人物在本章做过的关键动作，按发生顺序列出",
    )


class SceneInfo(BaseModel):
    """原文中的一个场景。"""

    location: str = Field(description="地点，例如“城西破庙”")
    time: str = Field(description="时间，例如“雨夜”“清晨”")
    atmosphere: str = Field(description="该场景的氛围，例如“阴冷、压抑、潮湿”")
    summary: str = Field(description="该场景发生了什么，1~3 句话")


class PlotExtraction(BaseModel):
    """节点1要求 LLM 输出的结构化剧情摘要。"""

    title: str = Field(description="章节标题；如果原文没有标题，用一句话概括后自拟一个短标题")
    one_line_summary: str = Field(description="一句话梗概，不超过 60 字")
    main_plot: str = Field(description="本章主线剧情，200~400 字，严格基于原文")
    characters: list[CharacterInPlot] = Field(
        description="本章所有有台词或有行动的人物；不要遗漏主角和关键配角"
    )
    scenes: list[SceneInfo] = Field(description="按发生顺序列出场景，通常 1~6 个")
    emotional_tone: str = Field(description="整体情绪基调，例如“冷峻、悬疑、压抑中带一点温度”")
    key_events: list[str] = Field(
        description="按顺序列出不可省略的关键事件，供后续校验剧情是否偏离",
    )
    fidelity_anchors: list[str] = Field(
        description="不可改动的事实锚点：关键台词、重要物件、人物关系、结局等；3~8 条",
    )
    unresolved_threads: list[str] = Field(
        default_factory=list,
        description="本章留下的悬念/未解决问题；没有就返回空列表",
    )


# ===========================================================================
# 四、节点2：漫剧剧本结构
# ===========================================================================
class ScriptBeat(BaseModel):
    """剧本中的最小单元：一句旁白、一句台词，或一段动作描述。"""

    kind: Literal["旁白", "台词", "动作描述"] = Field(
        description="只能是“旁白”“台词”“动作描述”三者之一",
    )
    speaker: str | None = Field(
        default=None,
        description=(
            "当 kind=台词 时必填，只写一个说话人姓名；"
            "禁止写“林昭：林昭”“林昭 林昭”“林昭（林昭）”等重复姓名；旁白/动作描述填 null"
        ),
    )
    content: str = Field(
        min_length=1,
        description=(
            "具体内容。台词不要带引号，不要把“某某说”写进 content；"
            "台词正文开头不要再重复“角色名：”"
        ),
    )

    @model_validator(mode="after")
    def _check_dialogue_speaker(self) -> "ScriptBeat":
        """台词必须有说话人；这条规则也参与结构化解析重试。"""
        if self.kind == "台词" and not (self.speaker or "").strip():
            raise ValueError("kind='台词' 时 speaker 不能为空")
        return self


class ScriptScene(BaseModel):
    """一场戏。"""

    location: str = Field(description="地点")
    time: str = Field(description="时间，例如“夜”“清晨”“雨中”")
    characters: list[str] = Field(
        default_factory=list,
        description="本场出场人物姓名列表",
    )
    beats: list[ScriptBeat] = Field(
        min_length=1,
        description="按播放顺序排列的旁白/台词/动作描述",
    )


class ScriptDraft(BaseModel):
    """节点2要求 LLM 输出的完整剧本。"""

    title: str = Field(description="漫剧标题，通常沿用章节标题")
    logline: str = Field(description="一句话卖点/梗概，不超过 80 字")
    scenes: list[ScriptScene] = Field(
        min_length=1,
        description="按顺序排列的场次，通常 3~8 场",
    )


class ContinueScriptDraft(BaseModel):
    """续写分支的结构化输出。"""

    title: str = Field(description="新一集标题，例如“第4章 雨夜破庙”，不要只写“第四夜”")
    logline: str = Field(description="一句话卖点/梗概，不超过 80 字")
    plot_summary: str = Field(
        min_length=20,
        description="本集主线摘要，200~400字；会作为下一集的历史记忆",
    )
    key_events: list[str] = Field(
        default_factory=list,
        description="本集按顺序发生的关键事件，供后续衔接使用",
    )
    script: ScriptDraft = Field(description="新一集的完整漫剧剧本")


class ModifyScriptDraft(BaseModel):
    """修改分支的结构化输出。"""

    title: str = Field(description="修改后的本集标题")
    change_summary: str = Field(
        min_length=1,
        description="只改了什么，例如“将第三夜结局改成苏晚离开”；供分镜和报告使用",
    )
    plot_summary: str = Field(
        min_length=20,
        description="修改后本集摘要；会覆盖 story_memory 中同一集的摘要",
    )
    script: ScriptDraft = Field(description="修改后的完整剧本，未涉及内容必须保持原样")


# ===========================================================================
# 五、节点3：分镜结构
# ===========================================================================
class Shot(BaseModel):
    """一个镜头。字段与最终表格列一一对应（序号由代码生成）。"""

    duration_seconds: float = Field(
        description="镜头时长，单位秒，通常 2~8 秒；代码会统一格式并做上下限保护",
    )
    visual: str = Field(
        min_length=1,
        description="画面描述：谁、在哪、做什么、光线/环境如何，必须可拍",
    )
    dialogue: str = Field(
        default="",
        description="该镜头对应的台词或旁白；必须来自剧本，没有就填空字符串",
    )
    camera: str = Field(
        min_length=1,
        description="镜头运镜，例如“中景固定”“远景缓慢推近”“特写跟拍”",
    )
    mood: str = Field(
        min_length=1,
        description="氛围备注：色调、光线、声音提示等；不涉及实际配音/配乐生成",
    )


class StoryboardDraft(BaseModel):
    """节点3要求 LLM 输出的分镜数组。"""

    shots: list[Shot] = Field(
        min_length=1,
        description="按剧本顺序拆出的镜头列表，通常 10~30 个",
    )


class GlobalCharacter(BaseModel):
    """全剧复用角色档案；首次新建剧本时抽取一次并落 JSON。"""

    name: str = Field(description="角色主姓名")
    aliases: list[str] = Field(default_factory=list, description="别名/称呼")
    role: str = Field(default="配角", description="主角/配角/反派/龙套")
    gender: str = Field(default="未说明", description="性别；不确定填未说明")
    age: str = Field(default="未说明", description="年龄/年龄段；不确定填未说明")
    appearance: str = Field(
        default="",
        description=(
            "只写永久固定特征：脸型、五官、发型、身高体型、永久旧疤/胎记、"
            "长期固定配饰；禁止写潮湿、血迹、污渍、临时穿戴、当下表情、临时受伤"
        ),
    )
    personality: str = Field(
        default="",
        description="长期稳定的性格与行为方式；不写某一集临时情绪",
    )
    speech_style: str = Field(default="", description="说话风格")
    background: str = Field(default="", description="身份、经历、与主线关系")
    relationships: str = Field(default="", description="与其他角色的关系")
    taboos: str = Field(default="", description="禁止出现的 OOC 行为")


class GlobalProfileDraft(BaseModel):
    """DeepSeek 冷启动抽取的全局角色 + 世界观。"""

    worldview: str = Field(
        min_length=20,
        description="故事发生的时代、地点、社会规则、核心设定；不写本集剧情",
    )
    characters: list[GlobalCharacter] = Field(
        min_length=1,
        description="全剧稳定角色档案，包含主角和常驻配角",
    )


class EpisodeShot(BaseModel):
    """shot_generate_node 的单个镜头。"""

    shot_no: int = Field(ge=1, description="镜头序号，从 1 开始连续")
    duration_seconds: float = Field(description="镜头时长，通常 2~8 秒")
    visual: str = Field(min_length=1, description="可拍的画面描述")
    dialogue: str = Field(default="", description="台词/旁白；没有就空字符串")
    camera: str = Field(default="中景固定", description="运镜")
    mood: str = Field(default="", description="氛围/光线/声音")
    characters: list[str] = Field(default_factory=list, description="本镜头出场角色")


class EpisodeShotsDraft(BaseModel):
    """按集生成的分镜 JSON 结构。"""

    episode_tone: str = Field(
        min_length=2,
        description="本集情绪基调，例如冷峻悬疑、压抑、克制",
    )
    shots: list[EpisodeShot] = Field(
        min_length=1,
        description="按顺序排列的镜头，shot_no 从 1 连续递增",
    )


class CharacterBaseProfile(BaseModel):
    """角色基准图确认后的结构化档案，写入 Chroma 参考图库。"""

    name: str
    gender: str = "未说明"
    age: str = "未说明"
    facial_features: str = ""
    signature_features: str = ""
    base_outfit: str = ""
    temperament: str = ""
    negative_prompt: str = (
        "畸形五官，多手指，扭曲脸部，模糊，水印，文字，色差崩坏，表情怪异，多余肢体"
    )
    script_id: str = ""
    reference_images: list[str] = Field(default_factory=list)


class ImagePromptDraft(BaseModel):
    """文生图节点：把分镜转换成绘图正向/负向提示词。"""

    positive_prompt: str = Field(
        min_length=30,
        description=(
            "标准化绘图正向 prompt：角色锚点、场景、构图、光线、画风关键词；"
            "必须保证林昭长相一致"
        ),
    )
    negative_extra: list[str] = Field(
        default_factory=list,
        max_length=12,
        description="在基础负面模板之外追加的负面要求；没有就返回空列表",
    )


class ShotPromptDraft(BaseModel):
    """单镜头绘图提示词生成结果。"""

    positive_prompt: str = Field(
        min_length=30,
        description="只包含绘图 prompt，不包含剧本、台词、运镜、时长",
    )
    negative_prompt: str = Field(
        min_length=5,
        description="完整负面提示词，包含基础负面模板和额外负面要求",
    )


# ===========================================================================
# 六、节点4：校验结构
# ===========================================================================
class ValidationIssue(BaseModel):
    """一条校验问题。"""

    category: Literal["剧情偏离", "人物OOC"] = Field(
        description="只能是“剧情偏离”或“人物OOC”",
    )
    severity: Literal["严重", "轻微"] = Field(
        description="严重：必须修改；轻微：建议修改。Demo 中任何问题都会触发重写",
    )
    location: str = Field(
        description="问题位置，例如“剧本第2场”“分镜镜头7”，必须具体",
    )
    description: str = Field(
        description="问题是什么，必须指出与原文/设定冲突的具体内容",
    )
    suggestion: str = Field(
        description="可执行的修改建议，例如“把台词改回……”“删除新增的……”",
    )


class ValidationResult(BaseModel):
    """节点4要求 LLM 输出的校验结果。"""

    plot_faithful: bool = Field(
        description="是否忠于原著剧情。只要有关键剧情被增加/删改/矛盾，就返回 false",
    )
    character_consistent: bool = Field(
        description="人物是否 OOC。只要有性格、说话风格、行为选择冲突，就返回 false",
    )
    issues: list[ValidationIssue] = Field(
        default_factory=list,
        description="发现的问题列表；没有实质问题就返回空列表，不要为了凑数挑文笔毛病",
    )
    summary: str = Field(
        description="一句话总结本轮校验结论",
    )


# ===========================================================================
# 七、续写/修改分支的剧情忠实度校验
# ===========================================================================
class BranchIssue(BaseModel):
    """续写/修改分支的一条问题。"""

    issue_id: str = Field(
        default="",
        description=(
            "历史台账中的问题ID；同一个问题必须复用原ID，"
            "新问题留空由代码分配。不要每轮给同一个问题换新ID。"
        ),
    )
    category: Literal["人物OOC", "吃设定", "剧情矛盾", "修改越界", "重复驳回"] = Field(
        description="问题类型；修改越界=改了用户没要求改的内容；重复驳回=换皮重复上一轮问题",
    )
    severity: Literal["严重", "轻微"] = Field(
        description="严重：必须修改；轻微：建议修改",
    )
    location: str = Field(
        description="具体位置，例如“剧本第2场”“台词：林昭：……”，必须可定位",
    )
    description: str = Field(
        description="与角色人设/历史摘要冲突的具体内容",
    )
    suggestion: str = Field(
        description="可执行的修改建议",
    )


class BranchValidationResult(BaseModel):
    """续写/修改分支在生成剧本后、生成分镜前的校验结果。"""

    passed: bool = Field(
        description="是否通过；有任何实质问题就返回 false",
    )
    issues: list[BranchIssue] = Field(
        default_factory=list,
        description="问题列表；通过时返回空列表",
    )
    summary: str = Field(
        description="一句话总结本轮校验",
    )
    resolved_issue_ids: list[str] = Field(
        default_factory=list,
        description="本轮已经实质性解决的历史问题ID；只是换词不算解决",
    )
    repeated_issue_ids: list[str] = Field(
        default_factory=list,
        description="仍然存在或只是换皮重复的历史问题ID",
    )
