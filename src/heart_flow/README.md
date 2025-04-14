# 心流系统 (Heart Flow System)

心流系统是一个模拟AI机器人内心思考和情感流动的核心系统。它通过多层次的心流结构，使AI能够对外界信息进行观察、思考和情感反应，从而产生更自然的对话和行为。

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

## 主要功能

### 思维系统
- 定期进行思维更新
- 维护短期记忆和思维连续性
- 支持多层次的思维处理

### 情感系统
- 情绪状态管理
- 回复意愿判断
- 情感因素影响决策

### 交互系统
- 群聊消息处理
- 多场景并行处理
- 智能回复生成

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

### 启动心流系统
```python
await heartflow.heartflow_start_working()
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