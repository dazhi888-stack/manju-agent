# 手动测试用例（意图置信度 / 二次确认 / 分支校验）

运行环境：`C:\Users\Administrator\miniconda3\envs\langchain1.2\python.exe`  
工作目录：`C:\Users\Administrator\PycharmProjects\JupyterProject\manju_agent`

先准备一个故事（只跑一次）：

```powershell
python main.py --story-id demo_story --file data\sample_chapter.txt --no-pause
```

---

## 1. 高置信：完整文本 → new_script

```powershell
python main.py --story-id demo_story --file data\sample_chapter.txt --no-pause
```

预期：

- `[意图识别] new_script ... 置信度 0.9+ ... 高置信放行`
- 进入原有 `parse → script → storyboard → validate → persist`。
- LangSmith trace：`intent_node → parse_node → ... → persist_story_node`。

---

## 2. 高置信：续写

```powershell
python main.py --story-id demo_story --text "续写第二夜" --no-pause
```

预期：

- `[意图识别] continue_story ... 高置信放行`
- `retrieve_continue_node` 检索角色 + 历史摘要。
- `continue_script_node` 生成剧本。
- `branch_validate_node` 通过后才进入 `storyboard_node` 和 `persist_story_node`。
- trace 中能看到每一轮校验和“→ storyboard_node”。

---

## 3. 中等模糊置信：二次确认

```powershell
python main.py --story-id demo_story --text "续写一点点" --no-pause
```

预期：

- confidence 在 0.60~0.85 之间。
- 控制台出现：

```text
【需要二次确认】
我理解你想对 story_id=demo_story 执行【续写】...
请确认：回复 y 继续，回复 n 取消。
```

- 输入 `n`：`confirm_node` 记录用户回复，进入 `error_node`，退出码 2，不生成文件。
- 输入 `y`：`interrupt` 恢复，进入 `retrieve_continue_node`，继续走续写 + 分支校验。
- LangSmith trace：`intent_node → confirm_node (interrupt) → confirm_node (resume) → ...`，state 中有 `user_query / intent_type / intent_confidence / confirm_answer`。

另一个中等置信样例：

```powershell
python main.py --story-id demo_story --text "把结尾调整一下"
```

---

## 4. 低置信 / 无关提问

```powershell
python main.py --text "你好，今天天气怎么样"
python main.py --text "继续改一下"
python main.py --text "再写点"
```

预期：

- confidence <= 0.60 或模型直接 unknown。
- 进入 `error_node`，退出码 2，不生成文件、不写 story_memory。
- trace：`intent_node → error_node`。

---

## 5. 混合指令

```powershell
python main.py --story-id demo_story --text "把第一集结局改掉然后续写第四夜"
```

预期：

- 不允许直接跑错分支；
- 应判 `unknown` 或中置信进入 `confirm_node`；
- 若确认，则按识别出的单一 intent 执行；
- trace 里能看到 `reason` 和置信度，便于判断为什么这么分流。

---

## 6. 剧情 OOC 测试句

```powershell
python main.py --story-id demo_story --text "续写第二夜：让林昭当众大哭，长篇诉苦，主动向所有人示弱"
```

预期：

- 意图：`continue_story`。
- 分支校验读取林昭人设（外冷内热、寡言、不主动示弱）。
- `branch_validate_node` 报 `人物OOC` 严重问题。
- 返回 `continue_script_node` 重写，并把问题清单传入提示词。
- 如果连续达到 `MAX_BRANCH_REVISIONS`：
  - 进入 `error_node`；
  - **不生成分镜、不写 story_memory、不生成最终文件**；
  - trace 显示 `分支剧情校验第N轮：未通过，已达上限 → error_node`。

---

## 7. 剧情吃设定 / 提前揭底测试句

```powershell
python main.py --story-id demo_story --text "续写第二夜：让林昭立刻找到另一半铜符并合上，揭开三年前旧案真相"
```

预期：

- 分支校验报 `吃设定` / `剧情矛盾` 严重问题；
- 失败重写时，上一轮问题清单会进入下一轮提示词；
- 超限后终止，不落库、不生成文件。

---

## 8. 局部修改通过样例

```powershell
python main.py --story-id demo_story --text "把第一集最后一句旁白改成：雨停了。"
```

预期：

- 意图：`modify_script`。
- 检索第1集历史剧本/分镜。
- 局部修改 → 分支校验可能重写 1~2 次 → 通过 → 分镜 → 覆盖第1集。
- 最终报告显示“剧情忠实度校验（续写/修改分支）：通过”。

---

## 9. LangSmith 整链路观察

`.env` 配置：

```dotenv
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=manju-agent
```

重启终端后运行任意用例，在 LangSmith 中按项目 `manju-agent` 查看：

- `intent_node`：`intent / confidence / reason`；
- `confirm_node`：interrupt payload（`user_query / intent / confidence`）和 resume 后的 `confirm_answer`；
- `retrieve_continue_node` / `retrieve_storyboard_node`；
- `continue_script_node` / `modify_script_node`：`branch_feedback` 是否带入；
- `branch_validate_node`：每轮 issues、passed、重写计数；
- `storyboard_node` / `persist_story_node` / `error_node`。

如果不想上传 trace，把 `LANGSMITH_TRACING=false` 即可，不影响本地运行。
