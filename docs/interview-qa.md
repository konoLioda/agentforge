# AgentForge 面试问答准备

> 配套文档：`docs/architecture.md`。本文所有回答都对应真实代码，涉及的具体类名、方法名、默认值、正则、文件路径均与源码一致，可放心引用。

---

## 一、1 分钟项目自我介绍模板

> 「我做过一个叫 **AgentForge** 的多智能体协作框架，基于 **LangChain + LangGraph**，后端接的是智谱 **GLM-4**。
>
> 它做了两条能力线：第一条是**单智能体的 ReAct 推理**，我自己手写了 Thought → Action → Observation → Final Answer 的循环，用正则去解析模型的输出，再安全地调用 9 个工具，比如用 AST 而不是 eval 做计算、用受限命名空间跑 Python 代码；第二条是**多智能体协作工作流**，用 LangGraph 的状态图把 Planner、Worker、Reviewer 三个角色串起来，Planner 拆任务、Worker 执行、Reviewer 打分审查，不通过就回退重试，靠条件边和 `operator.add` 的状态累加来实现。
>
> 此外我还补齐了工程闭环：短期记忆用 deque 固定窗口、长期记忆用 Chroma 向量库（内置了离线哈希嵌入，零下载）、多用户 JSON 持久化、FastAPI 提供 REST 和 SSE 流式接口、Streamlit 做前端。
>
> 最有意思的是我把**外部 LLM 依赖全部 mock 掉**，写了 38 个单元测试，离线、确定、不需要真实 API Key 就能全绿。
>
> 这个项目让我对『LLM 从一条 prompt 到一次工具调用、再到一张状态图』的完整链路有了落地理解，也让我能说清楚每个设计是在为什么取舍。」

> 面试官若追问「说一个你印象最深的难点」，可用下面第 9 题的回答衔接。

---

## 二、高频面试问题与参考答案

> 共 20 题，按主题分组。每题回答都力求「结论 + 本项目真实做法 + 为什么这么选」。

### A. 架构与选型

#### Q1. 为什么用 LangGraph，而不是只用 LangChain？

**答**：LangChain 擅长「把 LLM、工具、prompt 组装起来」，但它本质上还是**单次调用链**，缺乏对「多步骤、带循环、带状态」的显式建模。我项目的第二条能力线——Planner 拆任务、Worker 逐个子任务执行、Reviewer 审查、不通过就回退——天然是一个**有状态的图**，不是一条直线。

LangGraph 把 Agent 表达成 `StateGraph` + 节点 + 条件边：节点是纯函数，状态通过一个共享的 `TypedDict` 在节点间传递，条件边决定下一步去哪。我在 `agentforge/graph/workflow.py` 里定义了 `WorkflowState`，用两条条件函数 `should_continue`（还有子任务就回 Worker，否则去 Reviewer）和 `should_retry`（没通过且没到上限就回 Planner，否则结束）来驱动循环。

它的核心价值是：**把「流程控制」从业务代码里抽出来，变成一张可读、可扩展、可画图的状态机**。以后要加并行执行、加新角色、加人工确认节点，改图比改一堆 if-else 清晰得多。同时它的状态是显式声明的（`TypedDict`），比 LangChain 早期隐式的 `AgentExecutor` 内部状态更透明、更好测。

#### Q2. 你的项目整体分层是怎样的？

**答**：严格按职责分了六层（详见 `docs/architecture.md` 的分层图）：

1. **展示层**（`agentforge/ui`）：Streamlit，登录、对话、工作流两个 Tab。
2. **接口层**（`agentforge/api`）：FastAPI，REST + SSE。
3. **编排层**（`agentforge/graph`）：LangGraph 状态机。
4. **智能体层**（`agentforge/agents`）：`BaseAgent` 抽象基类 + React/Planner/Worker/Reviewer 四个子类。
5. **能力层**：`tools`（工具）、`memory`（记忆）、`user`（用户持久化）。
6. **基础设施层**：`config`（settings + logging）。

分层的收益是**依赖单向向下**：UI 依赖 API 或直接依赖 Agent，Agent 依赖工具和记忆，但工具/记忆不反向依赖 Agent，这让每个模块都能独立测试、独立替换。比如我可以把 `ReactAgent` 换成别的实现，而 API 层和 UI 层的契约（`AgentResponse`）不变。

#### Q3. 为什么用 Pydantic？它在你项目里承担什么角色？

**答**：Pydantic 在我的项目里有**三种不同的用法**，分别解决三类问题：

1. **数据契约 / 结构化输出**：`AgentResponse`、`PlannerOutput`、`SubTask`、`ReviewResult` 都是 `BaseModel`。比如 `ReviewResult` 定义了 `score: int`、`strengths: List[str]`、`approved: bool` 等字段，Reviewer 输出 JSON 后 `ReviewResult(**data)` 一行就完成了**校验 + 反序列化**——如果模型少给了字段或类型不对，会立刻抛错进入降级逻辑，而不是把一个「脏 dict」传遍全系统。
2. **工具入参 schema**：每个工具（`calculator` 的 `CalculatorInput`、`code_interpreter` 的 `CodeInterpreterInput` 等）都用 `BaseModel` 定义 `args_schema`，LangChain 据此约束工具参数、生成描述，也方便结构化传参 `tool.func(**parsed_input)`。
3. **API 请求/响应校验**：`ChatRequest`、`WorkflowResponse` 等，FastAPI 直接用它做自动参数校验和 OpenAPI 文档生成。

为什么不用手写 dict：因为 LLM 输出是**不可信的外部输入**，用 Pydantic 做**运行时类型护栏**，把「非法数据」挡在系统边界，比事后到处 `if isinstance(...)` 安全得多。同时 `model_dump()` 能方便地在 LangGraph 状态（要求可序列化 dict）之间转换。

#### Q4. 你自己写的 ReAct 循环，和 LangChain 自带的 AgentExecutor 有什么区别？为什么不直接用现成的？

**答**：LangChain 的 `AgentExecutor` 把「提示词、工具绑定、解析、循环上限」都封装好了，但它的解析逻辑（早期版本尤其）比较脆弱，而且内部状态对开发者不太透明。我选择**自己写循环**（`react_agent.py` 的 `think()`），原因有三：

1. **可讲解、可掌控**：ReAct 就三个正则（`_extract_thought` / `_extract_action` / `_extract_final_answer`），面试官能一眼看懂，我也能精确控制解析规则。
2. **定制降级策略**：比如「模型既没给 Action 也没给 Final Answer 时，直接把整段响应当最终答案并 break」，这种兜底在通用框架里很难定制。
3. **可测**：循环逻辑是纯 Python，配合 mock 的 LLM 客户端可以精确断言每一步。

代价我也清楚：放弃了 LangChain 原生工具绑定、记忆自动注入、结构化输出等便利，而且**正则对模型输出格式漂移是脆弱的**——如果模型把 JSON 包进 markdown 代码块、或者字段顺序变了，我的正则可能解析错。所以我在回答时也会主动说「这是显式选择了透明度，放弃了鲁棒性」。

#### Q5. 多智能体（Planner/Worker/Reviewer）之间是怎么协作的？数据怎么流转？

**答**：协作的本质是**通过 LangGraph 共享状态 `WorkflowState` 传递数据**，而不是三个 Agent 互相发消息。

- **Planner** 把原始任务 `original_task` 拆成 `subtasks` 列表（每个子任务有 `id`、`depends_on`、`assigned_to`），写回状态。
- **Worker** 读 `current_task_id` 找到当前子任务，再根据 `depends_on` 从 `completed_outputs` 里取前置任务的输出拼成上下文，调用自己内部持有的 `ReactAgent` 执行，把结果追加进 `completed_outputs`，然后 `current_task_id += 1`。
- **Reviewer** 把所有 `completed_outputs` 拼接起来，对照 `original_task` 打分并给出 `approved`。
- 未通过且 `iteration < max_iterations`，条件边把它送回 Planner 重新规划。

关键点是 `completed_outputs: Annotated[List[str], operator.add]`。因为 LangGraph 里节点返回的 dict 默认是**覆盖**状态，而我需要「多个 Worker 的输出被累积」，所以用 `operator.add` 作为 reducer，让每次返回的 `[content]` 被**列表拼接**而不是替换。这是「多智能体产出如何聚合」的核心机制。

### B. ReAct 与工具

#### Q6. 讲讲 ReAct 的原理，以及你在代码里是怎么解析的？

**答**：ReAct（Reasoning + Acting）的核心思想是**让模型把「推理」和「行动」交替地显式写出来**，而不是一口气给答案。好处是：推理过程可解释，且模型可以通过「行动 → 观察 → 再推理」的闭环获取外部信息、自我纠错。

我的实现（`react_agent.py`）：

1. **系统提示词强制格式**：要求模型严格输出 `Thought:` / `Action:` / `Action Input:` / `Observation:` / `Final Answer:`，并约定「每次只执行一个 Action」。
2. **正则解析**（三个 `_extract_*` 方法）：
   - Thought：`r"Thought:\s*(.+?)(?=Action:|Final Answer:|$)"`，非贪婪，`DOTALL` 支持跨行；
   - Action / Action Input：`r"Action:\s*(.+?)\s*$"`、`r"Action Input:\s*(.+?)\s*$"`，`MULTILINE` 按行匹配；
   - Final Answer：`r"Final Answer:\s*(.+)"`，`DOTALL` 贪婪取到结尾。
3. **执行循环**：`for iteration in range(max_iterations)`（默认 5）——有 Final Answer 就 break；有 Action 就执行工具并把结果以 `Observation: ...` 追加回消息，让模型「看到」工具结果再继续推理；两者都没有就兜底结束，防止死循环。

我会特别强调「为什么用正则而不是 JSON」：因为 ReAct 的交互文本是「自然语言推理 + 结构化字段」混合，正则能容忍中间的随意文本；但代价是格式漂移时脆弱，这也是我列在改进点里的。

#### Q7. 工具是如何被「安全调用」的？举两个例子。

**答**：我项目里「安全」分两层：**调用层的容错** + **工具内部的安全实现**。

调用层（`_execute_action`）：
- 先查表，工具名不在 `AVAILABLE_TOOLS` 里就返回「未知工具 + 可用列表」字符串，绝不抛异常。
- 输入先尝试 `json.loads`：解析成 dict 就 `tool.func(**parsed_input)` 结构化传参，否则降级为纯文本 `tool.run(...)`。
- 全程 try/except，工具异常转成错误字符串回填，让循环继续。

工具内部的安全实现，举两个例子：
- **计算器**：我**不用 `eval()`**，而是用 `ast` 把表达式解析成语法树，只允许白名单里的函数（`sin/cos/tan/sqrt/log` 等）和运算符（加减乘除幂取负），遇到未知节点直接抛「不支持」。这样 `__import__('os').system(...)` 这类注入从根上就进不来。
- **代码解释器**：用**关键字黑名单**（`import/open/exec/eval/subprocess/socket/...`）+ **受限 `__builtins__`**（只给 `print/range/len/sum/...` 等安全函数）+ **重定向 `sys.stdout`** 捕获输出，异常时返回 `执行失败: ...`。

同时我会诚实地说边界：黑名单方案**不是真正的沙箱**，理论上仍可能被绕过（比如 `getattr` 的间接访问），生产环境应该用容器或 `RestrictedPython` + 子进程资源限制。这是「演示级安全 vs 生产级安全」的区别。

#### Q8. 搜索、网页抓取、API 调用这些涉及网络的工具，怎么保证稳定性和超时？

**答**：统一用 `httpx`，并给每个网络工具设置**明确超时**——搜索和抓取是 10 秒（`httpx.Client(timeout=10.0)`），通用 API 调用是 15 秒，图片生成下载是 30 秒。

更重要的是**异常分类处理**：`search_tool.py` 里区分了 `HTTPStatusError`（返回 `HTTP 状态码`）、`TimeoutException`（返回「搜索超时」）和通用异常（返回错误详情）；`web_scraper` 和 `call_api` 也用 try/except 返回友好错误串。这样即使外部服务挂了，工具也不会把异常抛到 ReAct 循环里，而是把「错误」作为 Observation 回填，模型还能据此决定换一个工具或直接告诉用户。

另外搜索工具在**未配置 `SEARCH_API_KEY` 时返回明确的错误提示**，而不是静默失败或崩溃。

### C. 记忆与向量检索

#### Q9. 你的记忆系统是怎么设计的？短期和长期有什么区别？

**答**：我做了**两级记忆**，对应人脑的「工作记忆」和「长期记忆」：

- **短期记忆**（`ShortTermMemory`）：用 `collections.deque(maxlen=20)` 存 `{"role", "content"}` 消息。`deque` 的 `maxlen` 让它**自动淘汰最旧消息**，保证上下文不会无限膨胀。ReactAgent 组装消息时只取最近 10 轮（`get_context(max_turns=10)`）。它的特点是**快、精确、但容量受限、进程内有效**。
- **长期记忆**（`LongTermMemory`）：用 ChromaDB 持久化到磁盘（`PersistentClient`，集合 `agent_memory`，余弦相似度）。写入用递增 ID `mem_{count:06d}`，检索 `search(query, n_results=5)` 返回 `id/content/metadata/distance`。它解决的是「跨会话、超窗口」的记忆。

两者配合的理想流程是：先用短期记忆提供紧邻上下文，再用长期记忆做**语义召回**补充历史相关信息。不过我要坦白一个现状：**长期记忆的检索在当前默认路径里其实是关闭的**——`BaseAgent.long_term` 默认是 `None`，`recall()` 在它为空时直接返回空列表，而 API 和工作流创建 Agent 时都没有注入 `LongTermMemory`。所以这是一个「接口已实现、默认未激活」的点，我把它列为首要改进项。

#### Q10. 讲讲向量检索的原理，以及你为什么默认用「离线哈希嵌入」而不是下载一个 embedding 模型？

**答**：向量检索的原理是：把文本映射成高维向量，让「语义相近」的文本在向量空间里距离更近，检索时把 query 也向量化，用相似度（我的集合用**余弦相似度** `hnsw:space=cosine`）找最近邻，Chroma 内部用 HNSW 近似最近邻索引加速。

我默认用一个自写的 `SimpleEmbedding`（`long_term.py`，64 维）：它把每个字符 `ord(ch) % 64` 累加到对应维度，最后 L2 归一化。选它做**默认**的原因是：

1. **零依赖、零联网**：不用下载 sentence-transformers 这类几百 MB 的模型，clone 就能跑，符合「开箱即用」定位；
2. **确定性**：同一文本永远得到同一向量，测试可复现；
3. **可替换**：构造函数 `embedding_function=None` 时用默认，调用方随时注入更强的语义嵌入。

但我会**主动强调它的致命短板**：字符哈希只能捕捉「字符重叠」，**没有真正的语义**——「猫」和「狗」会被当成几乎无关。所以它适合演示和骨架，生产必须换真正的 embedding 模型。这个「知道自己在妥协什么」的意识，比「会用向量库」更重要。

### D. 流式输出与 API

#### Q11. 流式输出（SSE）是怎么实现的？完整链路是什么？

**答**：完整链路分三段，都对应真实代码：

1. **LLM 层**：`ReactAgent.think_stream()` 调用智谱客户端时传 `stream=True`，得到一个 chunk 迭代器，逐块 `yield chunk.choices[0].delta.content`。注意它是**生成器**，不会一次性等所有 token。
2. **API 层**：`routes.py` 的 `/chat/stream` 用 FastAPI 的 `StreamingResponse`，`media_type="text/event-stream"`，并设置 `Cache-Control: no-cache` 和 `Connection: keep-alive`。内部的 `generate()` 协程遍历 `think_stream`，把每个 chunk 序列化成 SSE 格式 `data: {"content": ...}\n\n`，结束后发一个 `data: [DONE]\n\n` 作为结束哨兵；同时把 chunk 累积成 `full_response`，流结束后统一写入 `ConversationStore`。
3. **UI 层**：Streamlit 用 `st.write_stream(agent.think_stream(...))` 把生成器变成打字机式输出。

选择 SSE 而不是 WebSocket 的原因：**它是单向的、基于 HTTP、实现简单**，对「服务端 → 客户端推送 token」这种场景足够；WebSocket 适合需要客户端频繁双向交互的场景，这里用不上。SSE 还能自动断线重连（标准行为），比裸长轮询更省事。

#### Q12. 你的多用户系统和会话上下文是怎么管理的？

**答**：分「用户身份」和「会话状态」两层：

- **用户身份**（`UserManager`）：持久化到 `data/users.json`，`get_or_create_user` 按用户名复用或新建（`user_{序号:04d}`），并刷新 `last_active`；删除用户时同时用 `shutil.rmtree` 删掉该用户的对话目录。
- **对话持久化**（`ConversationStore`）：每个用户一个 `data/users/{user_id}/conversations.json`，每条消息带时间戳，支持 `get_history/clear/export_text`。
- **会话上下文**（`routes.py` 的 `agents_cache`）：`Dict[str, ReactAgent]` 按 `user_id` 缓存 Agent 实例。因为每个 `ReactAgent` 自带一个 `ShortTermMemory`，所以缓存 Agent 就等于**保持了该用户的对话上下文（短期记忆）**；匿名请求（无 `user_id`）则每次新建 Agent，无状态。

这个设计的优点是好懂、零外部依赖；**限制**我也清楚：`agents_cache` 是进程内字典，**多 worker 或服务重启后上下文就丢了**，也不支持横向扩展。改进方向是迁到 Redis 之类的共享会话存储。

### E. 测试与工程化

#### Q13. 你怎么保证测试的确定性？外部 LLM 依赖是怎么 mock 的？

**答**：核心思路是**把所有不确定性挡在测试边界外**——LLM、网络、文件系统都变成可控的。

具体在 `tests/conftest.py` 里做了一个 `FakeZhipuAI`：

- 定义 `_FakeResponse`（非流式）和 `_FakeStreamChunk`（流式）来模拟 `completions.create` 的返回值；
- `_content_for(messages)` 根据**系统提示词的内容分支**返回不同假响应：看到「任务规划专家」就返回 Planner 需要的 JSON，看到「质量审查专家」就返回 Reviewer 需要的 JSON，否则返回 ReAct 的 `Thought: ... / Final Answer: ...`；
- 用一个 `@pytest.fixture(autouse=True)` + `monkeypatch`，把 5 个模块里的 `ZhipuAI` 全部替换成 `FakeZhipuAI`。

这样做的三个好处：
1. **离线**：不发任何真实请求，不需要 API Key；
2. **确定**：同一个测试永远得到同一个「模型输出」，断言稳定；
3. **覆盖面广**：因为 mock 的是客户端类，ReAct 的解析、Planner/Reviewer 的 JSON 解析、API 全链路、工作流编排全都能被测到。

测试还用了 `tmp_path` fixture 让长期记忆的 Chroma 数据隔离，避免测试间互相污染。

#### Q14. 你这个项目一共多少测试？分别测了什么？

**答**：**38 个单元测试**（`pyproject.toml` 里 `testpaths=["tests"]`，README 也写着 38 个），分布是：

- `test_agents.py`（5 个）：`AgentResponse` 默认值、`BaseAgent.__repr__`、Planner 拆任务、Reviewer 审查格式；
- `test_tools.py`（16 个）：计算器 9 个（加减乘除、幂、负数、浮点、空输入、非法表达式、`tool.run`）、代码解释器 7 个（打印、列表推导、变量、空输入、`import` 禁止、`open` 禁止、数学运算）；
- `test_api.py`（7 个）：`/`、`/health`、创建/列用户、404、`/chat`、`/chat/stream`、`/workflow`；
- `test_memory.py`（10 个）：短期记忆 6 个（增、`maxlen` 截断、取最近、清空、摘要、空摘要）、长期记忆 4 个（增+检索、全量、清空、删除）。

测试的选型思路是：**确定性工具（计算器/代码解释器）直接断言输出，LLM 相关用 mock 保证可控，记忆用 `tmp_path` 隔离**。它证明的不只是「代码能跑」，而是「关键逻辑在无网络、无密钥环境下可复现验证」。

#### Q15. 为什么工具函数要「返回错误字符串」而不是抛异常？

**答**：因为工具的调用方是 **LLM 驱动的循环**，而不是人。如果工具抛异常、循环崩了，用户只会看到 500；但如果工具返回一段**人类可读的错误字符串**（如「未知工具」「搜索超时」「禁止使用: import」），这段文本会作为 `Observation` 回填给模型，模型就能**据此调整策略**——换个工具、换个参数、或者直接告诉用户「搜索不可用」。

这是一种「**把异常降级为数据**」的设计：把「控制流异常」转成「可供推理的内容」。代价是错误信息需要精心设计，否则模型可能误把错误当正确答案。这也是 Agent 工程里一个很典型的思路：**在 LLM 回路里，错误信息是宝贵的信号，不是要被吞掉的东西**。

### F. 难点、不足与改进

#### Q16. 这个项目最大的难点是什么？你是怎么解决的？

**答**：我认为最大的难点是**「让不可靠的 LLM 输出变得可靠」**，它贯穿了三个地方：

1. **ReAct 格式解析**：模型不总按你要求的格式输出。我用正则 + 循环上限 + 「无 Action 无 Final 就兜底」来解决，但我也承认正则本身对格式漂移是脆弱的。
2. **结构化输出**：Planner/Reviewer 要求 JSON，我一方面用低温度（0.3）提升稳定性，另一方面容忍 markdown 代码块包裹（`split("```json")`），并且**解析失败就走降级**（Planner 降级成单任务，Reviewer 降级成默认通过）。
3. **多智能体状态聚合**：多个 Worker 的输出要能被累积，我用 `operator.add` 做 reducer 解决「覆盖 vs 累加」的冲突。

总的来说，难点不在「调通 API」，而在**「为不可控的模型输出建立可控的边界」**——上限（循环次数）、护栏（Pydantic 校验）、兜底（降级策略）三样缺一不可。

#### Q17. 这个项目有哪些不足？如果让你重构/再改进，你会怎么做？

**答**：我对自己项目的短板有清晰的认知，按优先级列：

1. **长期记忆没真正闭环**：`long_term` 默认 `None`、`remember()` 和 `need_memory` 从未被调用，等于长期记忆只是「摆设」。改进：默认注入 `LongTermMemory`，在需要时落库。
2. **字符哈希嵌入无语义**：改进：换成 sentence-transformers 或智谱 Embedding（接口已通过 `embedding_function` 预留）。
3. **代码解释器非真沙箱**：黑名单可被绕过。改进：`RestrictedPython` 或容器 + 子进程 + 资源限制。
4. **子任务无真正拓扑排序**：`depends_on` 只是回读上下文，执行仍按 `id` 线性。改进：做 DAG 拓扑排序，用 LangGraph 的 fan-out 并行执行无依赖子任务。
5. **会话缓存是进程内 dict**：多实例/重启即失效。改进：Redis。
6. **清理死代码**：`ChatRequest.use_memory` 定义了没用到、`extract_zip_tool` 没注册、`AgentResponse.need_memory` 没消费。

这个回答的价值在于：**不是泛泛说「还可以更好」，而是每个改进都对应一个我读代码时发现的、具体的技术债**。

#### Q18. 你项目里有哪些「异常降级策略」？为什么这样设计？

**答**：我的项目几乎在每个「可能失败」的环节都设计了降级，核心原则是「**局部失败不能拖垮整体**」：

| 环节 | 降级策略 |
|------|---------|
| ReAct 无 Action 也无 Final | 把整段响应当最终答案并 break，避免死循环 |
| ReAct 工具未知/执行失败 | 返回错误字符串回填，不抛异常 |
| Planner JSON 解析失败 | 降级成「原任务 = 单个子任务」的 `PlannerOutput` |
| Reviewer JSON 解析失败 | 降级成 `score=7, approved=True`（默认通过） |
| 审查未通过 | 最多重试 `max_iterations` 次，到上限强制结束 |
| 网络工具未配 Key / 超时 / 异常 | 返回友好错误串 |
| API 层异常 | `try/except` → `HTTPException(500, str(exc))` |

这样设计的权衡是：**可用性 > 严格性**。宁可返回一个「不那么完美但可用」的结果，也不让一个子步骤把整个请求变成 500。代价是可能**放行低质量结果**（Reviewer 默认通过尤其如此），所以在生产环境我会加人工确认或更严格的兜底阈值。

#### Q19. 依赖注入（`api_key`）在你的项目里是怎么体现的？有什么好处？

**答**：最典型的体现是 `ReactAgent.__init__(..., api_key: Optional[str] = None)`：它默认 `api_key or settings.ZHIPU_API_KEY`，即「**没传就用环境变量，传了就用传入值**」。Streamlit 侧边栏让用户输入自己的智谱 Key，输入后 `ReactAgent(api_key=user_api_key)` 重建 Agent，**立即生效**，不需要重启进程或改 `.env`。

好处有三：
1. **配置的优先级清晰**：运行时参数 > 环境变量 > 默认值，测试、多租户、临时切换都方便。
2. **可测试性**：测试时不需要真实 Key，直接注入 fake 客户端（我的 mock 甚至替换了整个 `ZhipuAI` 类）。
3. **密钥不外泄**：Key 通过参数在内存里流动，配合 `.env` 被 gitignore，避免硬编码进源码。

我也会诚实指出**不彻底的地方**：目前只有 `ReactAgent` 支持 `api_key` 注入，`PlannerAgent`/`WorkerAgent`/`ReviewerAgent` 仍然直接读 `settings.ZHIPU_API_KEY`，多租户场景下这是个缺口，应该统一走构造参数注入。

#### Q20. 如果让你给这个项目加一个「高可用 / 生产级」特性，你会先加什么？为什么？

**答**：我会先加 **LLM 调用的重试 + 超时 + 限流（backoff）**。原因很实在：我现在的 `_call_llm` 是**一次裸调用，没有任何重试和超时控制**（`temperature=0.7, max_tokens=2048` 固定），一旦智谱 API 抖动，整个请求就 500 了，而且没有指数退避，重试只会把对方打得更惨。

其次是 **把 `agents_cache` 从进程内 dict 迁到 Redis**，解决多实例/重启丢上下文的问题。

选这两项而不是「上更多 fancy 的 Agent 特性」，是因为**它们直接决定系统在生产里「能不能稳定活着」**——一个功能再多但一抖就挂的 Agent 服务，是没有价值的。这也体现我判断优先级的原则：**先解决可靠性和状态，再谈智能性**。

---

## 三、快速记忆清单（面试前扫一眼）

- **ReAct 解析正则**：Thought 用 `(.+?)(?=Action:|Final Answer:|$)`；Action/Input 用 `.+?\s*$`（MULTILINE）；Final Answer 用 `(.+)`（DOTALL）。
- **循环上限**：`max_iterations=5`（ReactAgent）、`max_iterations=3`（工作流）。
- **状态累加**：`completed_outputs: Annotated[List[str], operator.add]`。
- **长期记忆**：Chroma `PersistentClient` + 自写 `SimpleEmbedding`（64 维字符哈希、L2 归一化、余弦空间）；默认 `long_term=None` 未激活。
- **短期记忆**：`deque(maxlen=20)`，取最近 10 轮。
- **安全计算**：`ast` 白名单求值，不用 `eval`。
- **代码执行**：黑名单 + 受限 `__builtins__` + 重定向 stdout。
- **SSE**：`StreamingResponse(media_type="text/event-stream")` + `data: {...}\n\n` + `data: [DONE]`。
- **降级**：Planner 失败→单任务；Reviewer 失败→默认通过（score=7）。
- **温度**：Planner/Reviewer 0.3，ReAct 0.7。
- **测试**：38 个，`conftest.py` 的 `FakeZhipuAI` autouse monkeypatch，离线可跑。
