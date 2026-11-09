# AI 漫剧 Agent

> 小说/剧本 → 漫剧剧本 → 6 列分镜 → 分镜 OOC 校验 → 视频提示词 → 持久化  
> 基于 **LangGraph + Chroma + FastAPI + Streamlit** 的多集连续生成 Demo。

---

## 1. 项目定位

这是一个面向“漫剧 / 短剧生产”的 AI Agent 原型。

它解决的不是“让模型写一段剧本”，而是多集连续生成时的工程问题：

- 上下文越来越长，不能每集都塞全文；
- 多集人物容易 OOC，外貌、服装、伤疤前后不一致；
- 剧本、分镜、图片、视频多模型调用，成本和失败率高；
- 不同剧本之间容易串记忆；
- 视频 / 图片失败不应拖垮整条主线。

项目当前是一套**可运行、可演示、可继续扩展**的工程化 Demo，不是完整商业产品。

---

## 2. 当前能力

| 能力 | 状态 | 说明 |
|---|---|---|
| 新建剧本 | ✅ | 粘贴小说，生成剧情摘要、剧本、分镜 |
| 续写故事 | ✅ | 按 `story_id` 检索记忆后续写 |
| 修改剧本 | ✅ | 指定集数做局部修改 |
| 意图识别 | ✅ | 新建 / 续写 / 修改 / unknown，带置信度确认 |
| 6 列分镜 | ✅ | 镜头序号 / 时长 / 画面 / 台词 / 运镜 / 氛围 |
| 分镜 JSON 缓存 | ✅ | `outputs/<story_id>/shots/ep_XX.json` |
| 分镜 OOC 校验 | ✅ | 失败回 `shot_generate_node`，超限不持久化 |
| 视频提示词切片 | ✅ | 纯 Python，按起始 / 结束镜头拼接 |
| Z-Image 分镜图 | ✅ | 手动生成，不自动消耗图片额度 |
| 角色基准图 | ✅ | 手动生成、确认后写 Chroma |
| 角色相似度校验 | ✅ | 有参考图时校验，失败重试 |
| Seedance 视频 | ⚠️ 预留 | `call_seedance_async` 当前为空实现，走降级 |
| 熔断 / 降级 | ✅ | 3 次失败打开、40 秒冷却、半开探测 |
| LangSmith trace | ✅ | 节点 trace / run 记录 |
| 故事 ID 隔离 | ✅ | 新剧本强校验，禁止覆盖合并 |
| 自动生成 story_id | ✅ | `story_001 / story_002 ...` |
| 离线测试 | ✅ | 66 项 |

> 重要：当前视频模型没有真正接通。视频节点会返回 `fail_degrade` 并保存提示词，
> 这是有意设计，方便在未接入 Seedance 时保证主流程不崩。

---

## 3. 技术栈

- Python 3.13
- LangGraph 1.1.2
- LangChain 1.2.x
- Chroma 1.5.x
- FastAPI + Uvicorn
- Streamlit
- Pydantic 2.x
- DeepSeek / 硅基流动 Qwen / 硅基流动 Z-Image
- 火山方舟 Seedance（预留）

---

## 4. 架构

```text
Streamlit (web_app.py)
        │
        ▼
FastAPI (api.py / main:app)
        │
        ▼
LangGraph (graph.py)
        │
        ├─ context_compress_node
        ├─ intent_node
        │    ├─ new_script
        │    ├─ continue_story
        │    ├─ modify_script
        │    └─ confirm / error
        │
        ├─ parse_node
        ├─ script_node
        ├─ shot_generate_node
        ├─ branch_validate_node
        ├─ build_video_prompt_node
        ├─ video_gen_node
        └─ persist_story_node
```

### 主链路

```text
new_script:
parse → script → shot_generate → branch_validate
      → build_video_prompt → video_gen → persist

continue_story:
retrieve_continue → continue_script → shot_generate
                 → branch_validate → build_video_prompt
                 → video_gen → persist

modify_script:
retrieve_storyboard → modify_script → shot_generate
                    → branch_validate → build_video_prompt
                    → video_gen → persist
```

> 新剧本和续写**不会自动生图**；图片由用户在【图片生成】页手动触发。

### 图片 / 角色参考链路（独立）

```text
POST /api/storyboard/image
POST /api/characters/reference/generate
POST /api/characters/reference/confirm
```

---

## 5. 记忆设计（核心）

项目不是把所有东西都塞进向量库，而是分四层：

### 5.1 工作记忆：LangGraph State

当前集的状态：

```text
story_id / episode_no / title
script / storyboard / shots
global_profile / canon_profiles
branch_validation / validation
video_prompt / image_path
trace
```

### 5.2 情节记忆：Chroma `story_memory`

- 每集摘要；
- 每集剧本；
- 故事圣经；
- 角色摘要。

续写时默认只召回最近 3 集摘要，避免长上下文。

完整分镜不放进 LLM 上下文，只保存文件引用。

### 5.3 长期设定：本地 JSON

```text
data/stories/<story_id>/global_profile.json
```

保存：

- 世界观；
- 角色永久外貌；
- 固有配饰；
- 性格、说话风格、禁忌；
- 角色定位。

明确不写入全局档案的内容：

- 潮湿、淋雨、湿发；
- 血迹、污渍、泥水；
- 临时穿戴、当下表情；
- 单集受伤、当前动作。

这些只写进对应 shot 的 `visual/mood`，只在该镜头生效。

### 5.4 结构化产物：本地 shot JSON

```text
outputs/<story_id>/shots/ep_XX.json
```

包含：

```json
{
  "episode_no": 1,
  "episode_tone": "冷峻悬疑",
  "shots": [
    {
      "shot_no": 1,
      "duration_seconds": 5,
      "visual": "...",
      "dialogue": "...",
      "camera": "...",
      "mood": "...",
      "characters": ["林昭"]
    }
  ]
}
```

缓存命中时跳过 LLM，节省 token。

---

## 6. RAG 设计

### 写入

| 数据 | 存储 |
|---|---|
| 角色主档 | 本地 `global_profile.json` |
| 分集摘要 / 剧本 | Chroma `story_memory` |
| 故事圣经 | Chroma `story_memory` |
| 完整分镜 | `ep_XX.json` 本地文件 |
| 角色参考图 | Chroma `character_references` + 本地路径 |

### 查询

```text
story_id 过滤
  → 最近 3 集摘要
  → 长期故事圣经
  → 角色卡（优先本地 JSON）
  → 当前集 shot JSON（按需）
```

### 设计取舍

- 角色主档放本地 JSON：强一致、可覆盖、可读、可 diff；
- Chroma 放摘要与参考图：适合召回，不适合做主数据；
- 分镜不整表入向量库：避免一次召回 60 个镜头；
- 所有查询带 `story_id`，防止串剧本。

> 当前 Demo 使用本地哈希 embedding，语义能力有限；生产环境应换成 BGE / OpenAI-compatible embedding，并重建 collection。

---

## 7. 角色一致性

### 7.1 全局角色档案

首次新建剧本时抽取一次：

- 角色基础外貌；
- 永久旧疤；
- 固有配饰；
- 性格、说话风格、禁忌；
- 世界观。

JSON 保存后，同一 `story_id` 后续复用，不重复调用 LLM。

### 7.2 角色基准图

前端【图片生成】页：

1. 选择角色；
2. 默认读取角色库；
3. 可手动填写外貌 / 服装设定覆盖；
4. 生成一张 1080×1920 全身正面基准图；
5. 不满意可重新生成；
6. 确认满意后写 Chroma。

结构化档案包含：

```text
name / gender / age
facial_features / signature_features
base_outfit / temperament
negative_prompt / script_id / reference_images
```

固定负面词：

```text
畸形五官，多手指，扭曲脸部，模糊，水印，文字，色差崩坏，表情怪异，多余肢体
```

### 7.3 参考图声明

后续生图 / 视频提示词头部会拼接：

```text
@Image 角色基准参考图（必须严格保持以下角色长相）：
- 林昭：full: /path/林昭_full.png
约束：严格保留参考图里的五官、发型、标志性特征和基础服装；
只允许改变表情、动作、光影、镜头和单镜头临时状态。
```

### 7.4 图像相似度校验

- 有基准图时，新分镜图会提取图像向量；
- 与当前剧本角色参考图比较；
- 不足时最多重试 2 次；
- 超限标记 `character_ooc_failed`；
- 不进入视频、不写 story_memory。

> 当前机器没有本地 CLIP 模型缓存时，会使用 512 维颜色/结构感知向量兜底；
> 有本地 `openai/clip-vit-base-patch32` 缓存时自动使用 CLIP。

---

## 8. 视频节点：熔断 + 降级

`video_gen.py` 内置进程级熔断器：

```text
CLOSED    正常放行
OPEN      连续 3 次失败打开，冷却 40 秒
HALF_OPEN 冷却后只放行 1 次探测
```

返回固定字段：

```json
{
  "video_path": "",
  "video_prompt": "...",
  "status": "success | skipped_degrade | fail_degrade | circuit_open",
  "msg": "..."
}
```

状态含义：

| status | 含义 |
|---|---|
| `success` | 视频生成成功 |
| `skipped_degrade` | 用户关闭视频生成，只保存提示词 |
| `fail_degrade` | 接口失败 / 额度不足，保存提示词 |
| `circuit_open` | 熔断器拦截，未调用接口 |

降级提示词保存：

```text
outputs/<story_id>/第N集_video_prompt.txt
```

> `call_seedance_async` 当前故意留空；接入真实 Seedance 时只改这个函数，
> 熔断、降级、前端状态展示都不需要改。

---

## 9. 成本控制

- 分镜 JSON 文件存在即跳过 LLM；
- system prompt 固定前缀，命中 prompt cache；
- 图片 prompt 默认本地模板，不默认调用 DeepSeek；
- 新剧本 / 续写不自动生图；
- 视频提示词只取 1~3 个镜头；
- 失败重试有上限；
- Qwen 429/402/401/403 自动切 DeepSeek；
- 视频节点熔断，避免持续打不稳定接口。

---

## 10. 快速启动

### 10.1 环境

```powershell
cd C:\Users\Administrator\PycharmProjects\JupyterProject\manju_agent
C:\Users\Administrator\miniconda3\envs\langchain1.2\python.exe -m pip install -r requirements.txt
```

### 10.2 环境变量

在项目根目录 `.env` 配置（不要提交到 Git）：

```dotenv
# 文本模型
DEEPSEEK_API_KEY=sk-xxx
DEEPSEEK_MODEL=deepseek-v4-flash

# 硅基流动：剧本 Qwen / Z-Image
SILICONFLOW_API_KEY=sk-xxx
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1

# 图片
IMAGE_API_URL=https://api.siliconflow.cn/v1/images/generations
IMAGE_API_MODEL=Tongyi-MAI/Z-Image-Turbo
IMAGE_SIZE=1920x1080

# Seedance（预留）
SEEDANCE_KEY=ark-xxx
SEEDANCE_URL=https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks
SEEDANCE_MODEL=doubao-seedance-2-0-fast-260128

# LangSmith（可选）
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_xxx
LANGSMITH_PROJECT=manju-agent
```

### 10.3 Web 启动

终端 1：

```powershell
cd C:\Users\Administrator\PycharmProjects\JupyterProject\manju_agent
C:\Users\Administrator\miniconda3\envs\langchain1.2\python.exe -m uvicorn main:app --reload --port 8000
```

终端 2：

```powershell
cd C:\Users\Administrator\PycharmProjects\JupyterProject\manju_agent
C:\Users\Administrator\miniconda3\envs\langchain1.2\python.exe -m streamlit run web_app.py --server.port 8501
```

打开：

```text
前端：http://127.0.0.1:8501/
后端：http://127.0.0.1:8000/
接口文档：http://127.0.0.1:8000/docs
```

### 10.4 CLI

```powershell
# 交互菜单
python main.py --menu

# 直接输入
python main.py --text "第一章 ..." --story-id demo_story

# 续写
python main.py --text "续写第六章" --story-id demo_story

# 从已有 outputs 恢复记忆
python main.py --sync-outputs

# 清理/规范化 outputs
python main.py --cleanup-outputs
```

---

## 11. API 一览

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 后端首页 |
| GET | `/app` | 跳转 Streamlit |
| GET | `/api/health` | 健康检查 |
| GET | `/api/stories` | 故事列表 |
| GET | `/api/stories/next-id` | 自动生成 `story_00x` |
| POST | `/api/script/new` | 新建剧本 |
| POST | `/api/script/continue` | 续写 |
| POST | `/api/script/confirm` | 置信度确认 |
| GET | `/api/story/{story_id}` | 角色 / 历史摘要 |
| GET | `/api/story/{story_id}/episodes/{episode_no}` | 单集分镜 / 角色 |
| GET | `/api/characters/{story_id}` | 角色与参考图 |
| POST | `/api/characters/reference/generate` | 生成角色基准图 |
| POST | `/api/characters/reference/confirm` | 确认并写 Chroma |
| POST | `/api/storyboard/image` | 手动生成分镜图 |
| POST | `/api/prompt/shot` | 单镜头绘图提示词 |
| POST | `/api/video/prompt` | 按镜头切片视频提示词 |
| POST | `/api/video/generate` | 触发视频节点 |
| GET | `/api/video/circuit` | 熔断状态 |
| POST | `/api/video/circuit/reset` | 重置熔断器 |

### 新建剧本强校验

如果 `story_id` 已存在（Chroma / 本地档案 / outputs 任意一个），返回 409：

```text
该story_id已存在，请更换新ID，或者确认是否要清空旧剧本记忆
```

前端新建剧本区提供：

```text
[自动生成 story_id]
```

例如：

```text
story_001
story_002
story_003
```

---

## 12. 输出目录

```text
outputs/
  latest_result.md
  <story_id>/
    第N章_章节名称.md
    shots/
      ep_01.json
      ep_02.json
    prompts/
      第N章_镜头X.txt
      第N章_视频_镜头A-B.txt
    images/
      第N章_<style>.png
    characters/
      角色_full.png
    videos/
      第N集_seedance.mp4
```

### 旧数据回填

旧故事没有 `ep_XX.json` 时：

```powershell
python backfill_shots.py --story-id demo_story
```

只读已有 Markdown / 视频提示词，不调用 LLM。

---

## 13. 项目结构

```text
manju_agent/
├─ main.py                  # CLI + FastAPI app 入口
├─ api.py                   # FastAPI 路由
├─ web_app.py               # Streamlit 页面
├─ graph.py                 # LangGraph 状态机
├─ state.py                 # State + Pydantic Schema
├─ config.py                # 环境变量与路径
│
├─ node_parse.py            # 原文解析 + 冷启动全局档案
├─ node_script.py           # 剧本改写
├─ node_continue.py         # 续写剧本
├─ node_modify.py           # 局部修改
├─ node_shot_generate.py    # 结构化分镜 + 缓存
├─ node_shot_validate.py    # 分镜 OOC 校验
├─ node_video_prompt.py     # 视频提示词切片
├─ node_branch.py           # 校验路由 / 台账
├─ node_persist.py          # 持久化
├─ node_retrieve.py         # 记忆检索
├─ node_context.py          # 上下文压缩 / 故事圣经
├─ node_intent.py           # 意图识别 / 确认
├─ node_shared.py           # LLM 调用 / 日志 / 结构化重试
├─ node_storyboard.py       # 旧分镜节点（兼容）
├─ node_validate.py         # 旧校验节点（兼容）
│
├─ global_profile.py        # 全局角色 JSON
├─ character_reference.py   # 角色基准图 / 参考库
├─ shot_store.py            # ep_XX.json
├─ json_store.py            # JSON + Markdown 清洗
├─ video_prompt.py          # 纯 Python 视频提示词
├─ image_gen.py             # Z-Image 生图
├─ video_gen.py             # Seedance + 熔断降级
├─ vector_store.py          # Chroma
├─ output_importer.py       # outputs 导入
├─ output_writer.py         # Web 输出落盘
├─ backfill_shots.py        # 旧分镜回填
├─ prompts.py / prompts_global.py / prompts_shots.py
└─ tests/test_offline.py
```

---

## 14. 测试

```powershell
cd C:\Users\Administrator\PycharmProjects\JupyterProject\manju_agent
C:\Users\Administrator\miniconda3\envs\langchain1.2\python.exe -m unittest discover -s tests -q
```

当前：

```text
66 tests OK
```

覆盖：

- 意图路由 / 置信度分流；
- 剧本重写 / 校验回退；
- 分镜 JSON 缓存；
- 分镜 OOC 校验；
- 视频提示词切片；
- 熔断状态机；
- 视频降级；
- story_id 隔离；
- 全局档案清洗；
- 角色参考图存储 / 覆盖。

> 测试以离线 / mock 为主，不是完整真实模型评测。

---

## 15. 已知限制

1. **Seedance 未真正接通**：`call_seedance_async` 为空实现，视频节点默认走降级；
2. **CLIP 依赖本地缓存**：没有模型时使用感知向量兜底，不是严格 CLIP 语义相似度；
3. **embedding 是本地哈希**：适合 Demo / 精确匹配，语义检索能力有限；
4. **LangGraph checkpoint 是内存版**：生产要换 Redis/Postgres；
5. **Streamlit 只适合演示**：不是正式前端；
6. **没有鉴权 / 限流 / 多租户**；
7. **没有大规模 RAG 评测集**：目前主要靠离线测试和人工 smoke。

---

## 16. 设计取舍 

### 为什么不用“全文 + 一个超长 prompt”？

> 因为多集之后上下文会爆炸，而且模型会忽略中间内容。项目采用分层记忆：
> 最近 3 集摘要 + 故事圣经 + 当前 State + 结构化 shot JSON。

### 为什么角色主档放 JSON，不放 Chroma？

> 角色主档是强一致的主数据，需要覆盖、版本、可读；Chroma 负责召回和参考图。
> 这是按数据职责选存储，不是所有东西都塞向量库。

### 为什么分镜不整体进 Chroma？

> 一集 50~60 个镜头，整表召回会污染上下文。分镜落 `ep_XX.json`，
> 需要时按 `episode_no + shot_no` 读取。

### 为什么视频要熔断？

> 视频又慢又贵，失败不应该拖垮剧本和分镜。熔断 + 降级保存提示词，
> 主流程不崩，用户知道当前状态。

### 为什么新剧本 / 续写不自动生图？

> 图片额度贵且不是每次都需要。图片生成独立成手动操作，用户明确需要时才消耗额度。

---

## 17. 常见问题

### Q: 启动时报 `找不到 DEEPSEEK_API_KEY`

检查 `.env` 是否存在，并确认在项目根目录运行。

### Q: 剧本模型 402 / 余额不足

Qwen 余额不足会自动切 DeepSeek；如果两个都失败，检查两个 Key 的余额。

### Q: 图片 451 / 429

接口会等待并换中性 prompt 重试；仍失败时换角色或稍后再试。

### Q: 视频一直 `fail_degrade`

当前 `call_seedance_async` 为空实现，属于预期。接入真实 Seedance 后再测试 success。

### Q: 新建剧本提示 story_id 已存在

换新 ID，或先确认是否真的要清理旧剧本记忆。项目默认禁止覆盖合并。

### Q: 图片生成后看不到历史图

图片生成是手动操作；当前已清理旧图片，重新生成即可。

---

## 18. 后续可扩展

- 真实 Seedance 接入；
- BGE / OpenAI embedding + 向量库迁移；
- RAG recall@k / groundedness 评测；
- Redis/Postgres checkpointer；
- 任务队列与异步视频回调；
- 角色参考图 image-to-image / 多视图一致性；
- 鉴权、限流、多租户；
- 前端从 Streamlit 迁移到 React/Next.js；
- 监控、告警、成本看板。

---

## 19. 免责声明

本项目用于学习、演示和面试展示。生成内容、图片和视频请遵守模型服务商条款及当地法律法规。
