# 心流系统 (Heart Flow System)

## 系统架构

### 1. 主心流 (Heartflow)
- 位于 `heartflow.py`
- 作为整个系统的主控制器
- 负责管理和协调多个子心流
- 维护AI的整体思维状态
- 定期进行全局思考更新

### 2. 子心流 (SubHeartflow)
- 位于 `sub_heartflow.py`
- 处理具体的对话场景（如群聊）
- 维护特定场景下的思维状态
- 通过观察者模式接收和处理信息
- 能够进行独立的思考和回复判断

### 3. 观察系统 (Observation)
- 位于 `observation.py`
- 负责收集和处理外部信息
- 支持多种观察类型（如聊天观察）
- 对信息进行实时总结和更新


## 工作流程

1. 主心流启动并创建必要的子心流
2. 子心流通过观察者接收外部信息
3. 系统进行信息处理和思维更新
4. 根据情感状态和思维结果决定是否回复
5. 生成合适的回复并更新思维状态

## 使用说明

### 创建新的子心流
```python
heartflow = Heartflow()
subheartflow = heartflow.create_subheartflow(chat_id)
```

### 添加观察者
```python
observation = ChattingObservation(chat_id)
subheartflow.add_observation(observation)
```

## 配置说明

系统的主要配置参数：
- `sub_heart_flow_stop_time`: 子心流停止时间
- `sub_heart_flow_freeze_time`: 子心流冻结时间
- `heart_flow_update_interval`: 心流更新间隔

## 注意事项

1. 子心流会在长时间不活跃后自动清理
2. 需要合理配置更新间隔以平衡性能和响应速度
3. 观察系统会限制消息处理数量以避免过载 

# HeartFChatting 与主动回复流程说明 (V2)

本文档描述了 `HeartFChatting` 类及其在 `heartFC_controler` 模块中实现的主动、基于兴趣的回复流程。

## 1. `HeartFChatting` 类概述

*   **目标**: 管理特定聊天流 (`stream_id`) 的主动回复逻辑，使其行为更像人类的自然交流。
*   **创建时机**: 当 `HeartFC_Chat` 的兴趣监控任务 (`_interest_monitor_loop`) 检测到某个聊天流的兴趣度 (`InterestChatting`) 达到了触发回复评估的条件 (`should_evaluate_reply`) 时，会为该 `stream_id` 获取或创建唯一的 `HeartFChatting` 实例 (`_get_or_create_heartFC_chat`)。
*   **持有**:
    *   对应的 `sub_heartflow` 实例引用 (通过 `heartflow.get_subheartflow(stream_id)`)。
    *   对应的 `chat_stream` 实例引用。
    *   对 `HeartFC_Chat` 单例的引用 (用于调用发送消息、处理表情等辅助方法)。
*   **初始化**: `HeartFChatting` 实例在创建后会执行异步初始化 (`_initialize`)，这可能包括加载必要的上下文或历史信息（*待确认是否实现了读取历史消息*）。

## 2. 核心回复流程 (由 `HeartFC_Chat` 触发)

当 `HeartFC_Chat` 调用 `HeartFChatting` 实例的方法 (例如 `add_time`) 时，会启动内部的回复决策与执行流程：

1.  **规划 (Planner):**
    *   **输入**: 从关联的 `sub_heartflow` 获取观察结果、思考链、记忆片段等上下文信息。
    *   **决策**:
        *   判断当前是否适合进行回复。
        *   决定回复的形式（纯文本、带表情包等）。
        *   选择合适的回复时机和策略。
    *   **实现**: *此部分逻辑待详细实现，可能利用 LLM 的工具调用能力来增强决策的灵活性和智能性。需要考虑机器人的个性化设定。*

2.  **回复生成 (Replier):**
    *   **输入**: Planner 的决策结果和必要的上下文。
    *   **执行**:
        *   调用 `ResponseGenerator` (`self.gpt`) 或类似组件生成具体的回复文本内容。
        *   可能根据 Planner 的策略生成多个候选回复。
    *   **并发**: 系统支持同时存在多个思考/生成任务（上限由 `global_config.max_concurrent_thinking_messages` 控制）。

3.  **检查 (Checker):**
    *   **时机**: 在回复生成过程中或生成后、发送前执行。
    *   **目的**:
        *   检查自开始生成回复以来，聊天流中是否出现了新的消息。
        *   评估已生成的候选回复在新的上下文下是否仍然合适、相关。
        *   *需要实现相似度比较逻辑，防止发送与近期消息内容相近或重复的回复。*
    *   **处理**: 如果检查结果认为回复不合适，则该回复将被**抛弃**。

4.  **发送协调:**
    *   **执行**: 如果 Checker 通过，`HeartFChatting` 会调用 `HeartFC_Chat` 实例提供的发送接口：
        *   `_create_thinking_message`: 通知 `MessageManager` 显示"正在思考"状态。
        *   `_send_response_messages`: 将最终的回复文本交给 `MessageManager` 进行排队和发送。
        *   `_handle_emoji`: 如果需要发送表情包，调用此方法处理表情包的获取和发送。
    *   **细节**: 实际的消息发送、排队、间隔控制由 `MessageManager` 和 `MessageSender` 负责。

## 3. 与其他模块的交互

*   **`HeartFC_Chat`**:
    *   创建、管理和触发 `HeartFChatting` 实例。
    *   提供发送消息 (`_send_response_messages`)、处理表情 (`_handle_emoji`)、创建思考消息 (`_create_thinking_message`) 的接口给 `HeartFChatting` 调用。
    *   运行兴趣监控循环 (`_interest_monitor_loop`)。
*   **`InterestManager` / `InterestChatting`**:
    *   `InterestManager` 存储每个 `stream_id` 的 `InterestChatting` 实例。
    *   `InterestChatting` 负责计算兴趣衰减和回复概率。
    *   `HeartFC_Chat` 查询 `InterestChatting.should_evaluate_reply()` 来决定是否触发 `HeartFChatting`。
*   **`heartflow` / `sub_heartflow`**:
    *   `HeartFChatting` 从对应的 `sub_heartflow` 获取进行规划所需的核心上下文信息 (观察、思考链等)。
*   **`MessageManager` / `MessageSender`**:
    *   接收来自 `HeartFC_Chat` 的发送请求 (思考消息、文本消息、表情包消息)。
    *   管理消息队列 (`MessageContainer`)，处理消息发送间隔和实际发送 (`MessageSender`)。
*   **`ResponseGenerator` (`gpt`)**:
    *   被 `HeartFChatting` 的 Replier 部分调用，用于生成回复文本。
*   **`MessageStorage`**:
    *   存储所有接收和发送的消息。
*   **`HippocampusManager`**:
    *   `HeartFC_Processor` 使用它计算传入消息的记忆激活率，作为兴趣度计算的输入之一。

## 4. 原有问题与状态更新

1.  **每个 `pfchating` 是否对应一个 `chat_stream`，是否是唯一的？**
    *   **是**。`HeartFC_Chat._get_or_create_heartFC_chat` 确保了每个 `stream_id` 只有一个 `HeartFChatting` 实例。 (已确认)
2.  **`observe_text` 传入进来是纯 str，是不是应该传进来 message 构成的 list?**
    *   **机制已改变**。当前的触发机制是基于 `InterestManager` 的概率判断。`HeartFChatting` 启动后，应从其关联的 `sub_heartflow` 获取更丰富的上下文信息，而非简单的 `observe_text`。
3.  **检查失败的回复应该怎么处理？**
    *   **暂定：抛弃**。这是当前 Checker 逻辑的基础设定。
4.  **如何比较相似度？**
    *   **待实现**。Checker 需要具体的算法来比较候选回复与新消息的相似度。
5.  **Planner 怎么写？**
    *   **待实现**。这是 `HeartFChatting` 的核心决策逻辑，需要结合 `sub_heartflow` 的输出、LLM 工具调用和个性化配置来设计。


## 6. 未来优化点

*   实现 Checker 中的相似度比较算法。
*   详细设计并实现 Planner 的决策逻辑，包括 LLM 工具调用和个性化。
*   确认并完善 `HeartFChatting._initialize()` 中的历史消息加载逻辑。
*   探索更优的检查失败回复处理策略（例如：重新规划、修改回复等）。
*   优化 `HeartFChatting` 与 `sub_heartflow` 的信息交互。



BUG:
2.复读，可能是planner还未校准好
3.planner还未个性化，需要加入bot个性信息，且获取的聊天内容有问题