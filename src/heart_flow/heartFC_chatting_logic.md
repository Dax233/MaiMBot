# HeartFChatting 逻辑详解

`HeartFChatting` 类是心流系统（Heart Flow System）中负责**专注聊天**（`ChatState.FOCUSED`）的核心组件。它的主要职责是在特定的聊天流 (`stream_id`) 中，通过一个持续的 **思考(Think)-规划(Plan)-执行(Execute)** 循环来模拟更自然、更深入的对话交互。当关联的 `SubHeartflow` 状态切换为 `FOCUSED` 时，`HeartFChatting` 实例会被创建并启动；当状态切换为其他（如 `CHAT` 或 `ABSENT`）时，它会被关闭。

## 1. 初始化 (`__init__`, `_initialize`)

-   **依赖注入**: 在创建时，`HeartFChatting` 接收 `chat_id`（即 `stream_id`）、关联的 `SubMind` 实例以及 `Observation` 实例列表作为参数。
-   **核心组件**: 内部初始化了几个关键组件：
    -   `ActionManager`: 管理当前循环可用的动作（如回复文本、回复表情、不回复）。
    -   `HeartFCGenerator`: (`self.gpt_instance`) 用于生成回复文本。
    -   `ToolUser`: (`self.tool_user`) 用于执行 `SubMind` 可能请求的工具调用（虽然在此类中主要用于获取工具定义，实际执行由 `SubMind` 完成）。
    -   `HeartFCSender`: (`self.heart_fc_sender`) 专门负责处理消息发送逻辑，包括管理"正在思考"状态。
    -   `LLMRequest`: (`self.planner_llm`) 配置用于执行规划任务的大语言模型请求。
-   **状态变量**:
    -   `_initialized`: 标记是否完成懒初始化。
    -   `_processing_lock`: 异步锁，确保同一时间只有一个完整的"思考-规划-执行"周期在运行。
    -   `_loop_active`: 标记主循环是否正在运行。
    -   `_loop_task`: 指向主循环的 `asyncio.Task` 对象。
    -   `_cycle_history`: 一个双端队列 (`deque`)，用于存储最近若干次循环的信息 (`CycleInfo`)。
    -   `_current_cycle`: 当前正在执行的循环信息 (`CycleInfo`)。
-   **懒初始化 (`_initialize`)**:
    -   在首次需要访问 `ChatStream` 前调用（通常在 `start` 方法中）。
    -   根据 `stream_id` 从 `chat_manager` 获取对应的 `ChatStream` 实例。
    -   更新日志前缀，使用聊天流的名称以提高可读性。

## 2. 生命周期管理 (`start`, `shutdown`)

-   **启动 (`start`)**:
    -   外部调用此方法来启动 `HeartFChatting` 的工作流程。
    -   内部调用 `_start_loop_if_needed` 来安全地启动主循环任务 (`_hfc_loop`)。
-   **关闭 (`shutdown`)**:
    -   外部调用此方法来优雅地停止 `HeartFChatting`。
    -   取消正在运行的主循环任务 (`_loop_task`)。
    -   清理内部状态（如 `_loop_active`, `_loop_task`）。
    -   释放可能被持有的处理锁 (`_processing_lock`)。

## 3. 核心循环 (`_hfc_loop`)

`_hfc_loop` 是 `HeartFChatting` 的心脏，它以异步方式无限期运行（直到被 `shutdown` 取消），不断执行以下步骤：

1.  **创建循环记录**: 初始化一个新的 `CycleInfo` 对象来记录本次循环的详细信息（ID、开始时间、计时器、动作、思考内容等）。
2.  **获取处理锁**: 使用 `_processing_lock` 确保并发安全。
3.  **执行思考-规划-执行**: 调用 `_think_plan_execute_loop` 方法。
4.  **处理循环延迟**: 根据本次循环是否执行了实际动作以及循环耗时，智能地引入短暂的 `asyncio.sleep`，防止 CPU 空转或过于频繁的循环。
5.  **记录循环信息**: 将完成的 `CycleInfo` 存入 `_cycle_history`，并记录详细的日志，包括循环耗时和各阶段计时。

## 4. 思考-规划-执行周期 (`_think_plan_execute_loop`)

这是每个循环内部的核心逻辑，按顺序执行：

### 4.1. 思考阶段 (`_get_submind_thinking`)

1.  **触发观察**: 调用关联的 `Observation` 实例的 `observe()` 方法，使其更新对环境（如聊天室新消息）的观察。
2.  **触发子思维**: 调用关联的 `SubMind` 实例的 `do_thinking_before_reply()` 方法。**关键**: 会将上一个循环的 `CycleInfo` 传递给 `SubMind`，使其了解上一次行动的决策、理由以及是否发生了重新规划，从而实现更连贯的思考。
3.  **获取思考结果**: `SubMind` 返回其当前的内心想法 (`current_mind`)。

### 4.2. 规划阶段 (`_planner`)

1.  **输入**: 获取 `SubMind` 的当前想法 (`current_mind`)、`SubMind` 通过工具调用收集到的结构化信息 (`structured_info`) 以及观察到的最新消息。
2.  **构建提示词**: 调用 `_build_planner_prompt` 方法，将上述信息以及机器人个性、当前可用动作等整合进一个专门为规划器设计的提示词中。
3.  **定义动作工具**: 使用 `ActionManager.get_planner_tool_definition()` 获取当前可用动作（如 `no_reply`, `text_reply`, `emoji_reply`）的 JSON Schema，将其作为 "工具" 提供给 LLM。
4.  **调用 LLM**: 使用 `self.planner_llm` 向大模型发送请求，**强制要求**模型调用 `decide_reply_action` 这个"工具"，并根据提示词内容决定使用哪个动作以及相应的参数（如 `reasoning`, `emoji_query`）。
5.  **处理 LLM 响应**: 使用 `process_llm_tool_response` 解析 LLM 返回的工具调用请求，提取出决策的动作 (`action`)、理由 (`reasoning`) 和可能的表情查询 (`emoji_query`)。
6.  **检查新消息与重新规划**:
    -   调用 `_check_new_messages` 检查自规划阶段开始以来是否有新消息。
    -   如果检测到新消息，有一定概率（当前为 30%）触发**重新规划**。这会再次调用 `_planner`，但会传入一个特殊的提示词片段（通过 `_build_replan_prompt` 生成），告知 LLM 它之前的决策以及现在需要重新考虑。
7.  **输出**: 返回一个包含最终决策结果（`action`, `reasoning`, `emoji_query` 等）的字典。如果 LLM 调用或解析失败，`action` 会被设为 "error"。

### 4.3. 执行阶段 (`_handle_action`)

根据规划阶段返回的 `action`，分派到不同的处理方法：

-   **`_handle_text_reply` (文本回复)**:
    1.  `_get_anchor_message`: 获取一个用于回复的锚点消息。**注意**: 当前实现是创建一个系统触发的占位符消息作为锚点，而不是实际观察到的最后一条消息。
    2.  `_create_thinking_message`: 调用 `HeartFCSender` 的 `register_thinking` 方法，标记机器人开始思考，并获取一个 `thinking_id`。
    3.  `_replier_work`: 调用回复器生成回复内容。
    4.  `_sender`: 调用发送器发送生成的文本和可能的表情。
-   **`_handle_emoji_reply` (仅表情回复)**:
    1.  获取锚点消息。
    2.  `_handle_emoji`: 获取表情图片并调用 `HeartFCSender` 发送。
-   **`_handle_no_reply` (不回复)**:
    1.  记录不回复的理由。
    2.  `_wait_for_new_message`: 进入等待状态，直到关联的 `Observation` 检测到新消息或超时（当前 300 秒）。

## 5. 回复器逻辑 (`_replier_work`)

-   **输入**: 规划器给出的回复理由 (`reason`)、锚点消息 (`anchor_message`)、思考ID (`thinking_id`)，以及通过 `self.sub_mind` 获取的结构化信息和当前想法。
-   **处理**: 调用 `self.gpt_instance` (`HeartFCGenerator`) 的 `generate_response` 方法。这个方法负责构建最终的生成提示词（结合思考、理由、上下文等），调用 LLM 生成回复文本。
-   **输出**: 返回一个包含多段回复文本的列表 (`List[str]`)，如果生成失败则返回 `None`。

## 6. 发送器逻辑 (`_sender`, `_create_thinking_message`, `_send_response_messages`, `_handle_emoji`)

`HeartFChatting` 类本身不直接处理 WebSocket 发送，而是将发送任务委托给 `HeartFCSender` 实例 (`self.heart_fc_sender`)。

-   **`_create_thinking_message`**: 准备一个 `MessageThinking` 对象，并调用 `sender.register_thinking(thinking_message)`。
-   **`_send_response_messages`**:
    -   检查对应的 `thinking_id` 是否仍然有效（通过 `sender.get_thinking_start_time`）。
    -   遍历 `_replier_work` 返回的回复文本列表 (`response_set`)。
    -   为每一段文本创建一个 `MessageSending` 对象。
    -   调用 `sender.type_and_send_message(bot_message)` 来发送消息。`HeartFCSender` 内部会处理模拟打字延迟、实际发送和消息存储。
    -   发送完成后，调用 `sender.complete_thinking(chat_id, thinking_id)` 来清理思考状态。
    -   记录实际发送的消息 ID 到 `CycleInfo` 中。
-   **`_handle_emoji`**:
    -   使用 `emoji_manager` 根据 `emoji_query` 获取表情图片路径。
    -   将图片转为 Base64。
    -   创建 `MessageSending` 对象（标记为 `is_emoji=True`）。
    -   调用 `sender.send_and_store(bot_message)` 来发送并存储表情消息（这个方法不涉及思考状态）。

## 7. 循环信息记录 (`CycleInfo`)

-   `CycleInfo` 类用于记录每一次思考-规划-执行循环的详细信息，包括：
    -   循环 ID (`cycle_id`)
    -   开始和结束时间 (`start_time`, `end_time`)
    -   是否执行了实际动作 (`action_taken`)
    -   决策的动作类型 (`action_type`) 和理由 (`reasoning`)
    -   各阶段的耗时计时器 (`timers`)
    -   关联的思考消息 ID (`thinking_id`)
    -   是否发生了重新规划 (`replanned`)
    -   详细的响应信息 (`response_info`)，包括生成的文本、表情查询、锚点消息 ID、实际发送的消息 ID 列表以及 `SubMind` 的思考内容。
-   `HeartFChatting` 维护一个 `_cycle_history` 队列来保存最近的循环记录，方便调试和分析。
