# MaiMBot 开发文档

## 📊 系统架构图

```mermaid
graph TD
    A[入口点] --> B[核心模块]
    A --> C[插件系统]
    B --> D[通用功能]
    C --> E[聊天系统]
    C --> F[记忆系统]
    C --> G[情绪系统]
    C --> H[意愿系统]
    C --> I[其他插件]
    
    %% 入口点
    A1[bot.py] --> A
    A2[run.py] --> A
    A3[webui.py] --> A
    
    %% 核心模块
    B1[src/common/logger.py] --> B
    B2[src/common/database.py] --> B
    
    %% 通用功能
    D1[日志系统] --> D
    D2[数据库连接] --> D
    D3[配置管理] --> D
    
    %% 聊天系统
    E1[消息处理] --> E
    E2[提示构建] --> E
    E3[LLM生成] --> E
    E4[关系管理] --> E
    
    %% 记忆系统
    F1[记忆图] --> F
    F2[记忆构建] --> F
    F3[记忆检索] --> F
    F4[记忆遗忘] --> F
    
    %% 情绪系统
    G1[情绪状态] --> G
    G2[情绪更新] --> G
    G3[情绪衰减] --> G
    
    %% 意愿系统
    H1[回复意愿] --> H
    H2[意愿模式] --> H
    H3[概率控制] --> H
    
    %% 其他插件
    I1[远程统计] --> I
    I2[配置重载] --> I
    I3[日程生成] --> I
```

## 📁 核心文件索引

| 功能 | 文件路径 | 描述 |
|------|----------|------|
| **入口点** | `/bot.py` | 主入口，初始化环境和启动服务 |
| | `/run.py` | 安装管理脚本，主要用于Windows |
| | `/webui.py` | Gradio基础的配置UI |
| **配置** | `/template.env` | 环境变量模板 |
| | `/template/bot_config_template.toml` | 机器人配置模板 |
| **核心基础** | `/src/common/database.py` | MongoDB连接管理 |
| | `/src/common/logger.py` | 基于loguru的日志系统 |
| **聊天系统** | `/src/plugins/chat/bot.py` | 消息处理核心逻辑 |
| | `/src/plugins/chat/config.py` | 配置管理与验证 |
| | `/src/plugins/chat/llm_generator.py` | LLM响应生成 |
| | `/src/plugins/chat/prompt_builder.py` | LLM提示构建 |
| **记忆系统** | `/src/plugins/memory_system/memory.py` | 图结构记忆实现 |
| | `/src/plugins/memory_system/draw_memory.py` | 记忆可视化 |
| **情绪系统** | `/src/plugins/moods/moods.py` | 情绪状态管理 |
| **意愿系统** | `/src/plugins/willing/willing_manager.py` | 回复意愿管理 |
| | `/src/plugins/willing/mode_classical.py` | 经典意愿模式 |
| | `/src/plugins/willing/mode_dynamic.py` | 动态意愿模式 |
| | `/src/plugins/willing/mode_custom.py` | 自定义意愿模式 |

## 🔄 模块依赖关系

```mermaid
flowchart TD
    A[bot.py] --> B[src/common/logger.py]
    A --> C[src/plugins/chat/bot.py]
    
    C --> D[src/plugins/chat/config.py]
    C --> E[src/plugins/chat/llm_generator.py]
    C --> F[src/plugins/memory_system/memory.py]
    C --> G[src/plugins/moods/moods.py]
    C --> H[src/plugins/willing/willing_manager.py]
    
    E --> D
    E --> I[src/plugins/chat/prompt_builder.py]
    E --> J[src/plugins/models/utils_model.py]
    
    F --> B
    F --> D
    F --> J
    
    G --> D
    
    H --> B
    H --> D
    H --> K[src/plugins/willing/mode_classical.py]
    H --> L[src/plugins/willing/mode_dynamic.py]
    H --> M[src/plugins/willing/mode_custom.py]
    
    I --> B
    I --> F
    I --> G
    
    J --> B
```

## 🔄 消息处理流程

```mermaid
sequenceDiagram
    participant User
    participant ChatBot
    participant WillingManager
    participant Memory
    participant PromptBuilder
    participant LLMGenerator
    participant MoodManager
    
    User->>ChatBot: 发送消息
    ChatBot->>ChatBot: 消息预处理
    ChatBot->>Memory: 记忆激活
    Memory-->>ChatBot: 激活度
    ChatBot->>WillingManager: 更新回复意愿
    WillingManager-->>ChatBot: 回复决策
    
    alt 决定回复
        ChatBot->>PromptBuilder: 构建提示
        PromptBuilder->>Memory: 获取相关记忆
        Memory-->>PromptBuilder: 相关记忆
        PromptBuilder->>MoodManager: 获取情绪状态
        MoodManager-->>PromptBuilder: 情绪状态
        PromptBuilder-->>ChatBot: 完整提示
        ChatBot->>LLMGenerator: 生成回复
        LLMGenerator-->>ChatBot: AI回复
        ChatBot->>MoodManager: 更新情绪
        ChatBot->>User: 发送回复
    else 不回复
        ChatBot->>WillingManager: 更新未回复状态
    end
```

## 📋 类和功能清单

### 🤖 聊天系统 (`src/plugins/chat/`)

| 类/功能 | 文件 | 描述 |
|--------|------|------|
| `ChatBot` | `bot.py` | 消息处理主类 |
| `ResponseGenerator` | `llm_generator.py` | 响应生成器 |
| `PromptBuilder` | `prompt_builder.py` | 提示构建器 |
| `Message`系列 | `message.py` | 消息表示类 |
| `RelationshipManager` | `relationship_manager.py` | 用户关系管理 |
| `EmojiManager` | `emoji_manager.py` | 表情符号管理 |

### 🧠 记忆系统 (`src/plugins/memory_system/`)

| 类/功能 | 文件 | 描述 |
|--------|------|------|
| `Memory_graph` | `memory.py` | 图结构记忆存储 |
| `Hippocampus` | `memory.py` | 记忆管理主类 |
| `memory_compress()` | `memory.py` | 记忆压缩函数 |
| `get_relevant_memories()` | `memory.py` | 记忆检索函数 |
| `operation_forget_topic()` | `memory.py` | 记忆遗忘函数 |

### 😊 情绪系统 (`src/plugins/moods/`)

| 类/功能 | 文件 | 描述 |
|--------|------|------|
| `MoodManager` | `moods.py` | 情绪管理器单例 |
| `MoodState` | `moods.py` | 情绪状态数据类 |
| `update_mood_from_emotion()` | `moods.py` | 情绪更新函数 |
| `_apply_decay()` | `moods.py` | 情绪衰减函数 |

### 🤔 意愿系统 (`src/plugins/willing/`)

| 类/功能 | 文件 | 描述 |
|--------|------|------|
| `WillingManager` | `willing_manager.py` | 意愿管理工厂类 |
| `ClassicalWillingManager` | `mode_classical.py` | 经典意愿模式 |
| `DynamicWillingManager` | `mode_dynamic.py` | 动态意愿模式 |
| `CustomWillingManager` | `mode_custom.py` | 自定义意愿模式 |

## 🔧 常用命令

- **运行机器人**: `python run.py` 或 `python bot.py`
- **安装依赖**: `pip install --upgrade -r requirements.txt`
- **Docker 部署**: `docker-compose up`
- **代码检查**: `ruff check .`
- **代码格式化**: `ruff format .`
- **内存可视化**: `run_memory_vis.bat` 或 `python -m src.plugins.memory_system.draw_memory`
- **推理过程可视化**: `script/run_thingking.bat`

## 🔧 脚本工具

- **运行MongoDB**: `script/run_db.bat` - 在端口27017启动MongoDB
- **Windows完整启动**: `script/run_windows.bat` - 检查Python版本、设置虚拟环境、安装依赖并运行机器人
- **快速启动**: `script/run_maimai.bat` - 设置UTF-8编码并执行"nb run"命令

## 📝 代码风格

- **Python版本**: 3.9+
- **行长度限制**: 88字符
- **命名规范**:
  - `snake_case` 用于函数和变量
  - `PascalCase` 用于类
  - `_prefix` 用于私有成员
- **导入顺序**: 标准库 → 第三方库 → 本地模块
- **异步编程**: 对I/O操作使用async/await
- **日志记录**: 使用loguru进行一致的日志记录
- **错误处理**: 使用带有具体异常的try/except
- **文档**: 为类和公共函数编写docstrings

## 📋 常见修改点

### 配置修改
- **机器人配置**: `/template/bot_config_template.toml`
- **环境变量**: `/template.env`

### 行为定制
- **个性调整**: `src/plugins/chat/config.py` 中的 BotConfig 类
- **回复意愿算法**: `src/plugins/willing/mode_classical.py`
- **情绪反应模式**: `src/plugins/moods/moods.py`

### 消息处理
- **消息管道**: `src/plugins/chat/message.py`
- **话题识别**: `src/plugins/chat/topic_identifier.py`

### 记忆与学习
- **记忆算法**: `src/plugins/memory_system/memory.py`
- **手动记忆构建**: `src/plugins/memory_system/memory_manual_build.py`

### LLM集成
- **LLM提供商**: `src/plugins/chat/llm_generator.py`
- **模型参数**: `template/bot_config_template.toml` 的 [model] 部分