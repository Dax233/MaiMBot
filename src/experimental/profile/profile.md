### **高级机器人侧写系统：设计文档与发展蓝图**

#### **1\. 系统愿景与目标**

本系统旨在构建一个先进的用户侧写（Profiling）框架，通过深度分析用户数据和交互行为，为机器人（Bot）赋予更深层次的上下文理解能力、更拟人化的情感表达和更真实的人际关系感知。最终目标是让机器人能够像一个有记忆、有洞察力、能共情的伙伴一样与用户互动。

核心目标包括：

* **统一的自然人识别**：为每一个真实用户建立一个稳定、唯一的内部标识 (NaturalPersonID)，并关联其在不同平台的账户。  
* **多维度用户画像构建**：全面记录用户的平台账户信息、身份特征、人格特质、常用绰号、以及系统形成的交互印象。  
* **动态数据导出与Prompt注入**：将结构化的画像数据实时导出，并智能地注入到语言模型的Prompt中，为机器人的决策和回复提供丰富依据。  
* **拟人化交互提升**：使机器人能够：  
  * 准确使用用户在特定情境下的绰号进行称呼。  
  * 根据用户的身份、人格和历史印象调整沟通风格和策略。  
  * 展现更细腻、非线性的情感和关系变化。  
* **模块化与可扩展性**：各项画像分析功能作为独立模块，易于管理、迭代和扩展。

#### **2.核心数据模型：profile\_info 表**

系统将围绕一个核心的MongoDB集合——profile\_info——构建。此表最终将取代现有的 person\_info 表。

* **文档主键 (\_id)**: NaturalPersonID  
  * 类型：字符串 (String)。  
  * 生成方式：对用户的“锚定标识”（例如，系统首次确认此自然人身份时关联的 平台标识-平台用户ID 组合）进行加盐哈希 (e.g., SHA256)。此ID在系统中唯一标识一个自然人。  
  * 作用：取代旧系统中由LLM生成的 person\_name 作为唯一标识。  
* **核心字段**：  
  * platform\_accounts: (对象 Object)  
    * 描述：存储此自然人拥有的、已被系统识别并关联的所有平台账户。  
    * 结构：{ "platform\_name\_1": \["user\_id\_a", "user\_id\_b"\], "platform\_name\_2": \["user\_id\_c"\] }  
    * 示例：{ "qq": \["12345", "67890"\], "wechat": \["wxid\_abc"\] }  
  * identity: (对象 Object)  
    * 描述：用户的身份特征信息，通过分析或用户声明获取。  
    * 结构（暂定）：  
      * cultural\_background: (字符串 String) 文化背景。  
      * professional\_role: (字符串 String) 职业角色。  
      * ethnic\_national\_identity: (字符串 String) 民族与国籍认同。  
      * religious\_belief: (字符串 String) 宗教信仰。  
      * gender\_identity: (字符串 String) 性别认同。  
      * disability\_info: (字符串 String) 残障信息（需谨慎处理隐私）。  
  * personality: (对象 Object)  
    * 描述：用户的人格特质，可基于标准模型（如大五人格）或自定义标签。  
    * 结构（暂定）：{ "openness": 0.7, "conscientiousness": 0.5, ... } 或 { "tags": \["乐观", "内向"\] }  
  * sobriquets\_by\_group: (对象 Object)  
    * 描述：用户在不同平台的不同群组中被使用的绰号及其频次。  
    * 结构：{ "platform-group\_id": { "sobriquets": \[{"name": "绰号A", "count": 10}\] } }  
    * 示例：{ "qq-group123": { "sobriquets": \[{"name": "大佬", "count": 25}, {"name": "阿强", "count": 10}\] } }  
  * impression: (数组 Array of Strings/Objects)  
    * 描述：系统基于长期交互对用户形成的整体印象或关键记忆点。  
    * 结构（暂定）：\["乐于助人", "喜欢开玩笑", {"event": "上次讨论了AI技术", "timestamp": 1678886400}\]  
  * relationship\_metrics: (对象 Object) \- *未来整合*  
    * 描述：替代原有 person\_info 中的线性关系值，可能包含更复杂的关系维度。  
    * 结构（暂定）：{ "familiarity": 0.8, "trust": 0.6, "interaction\_frequency": "high", "last\_interaction\_mood": "positive" }  
  * creation\_timestamp: (日期 Date) 文档创建时间。  
  * last\_updated\_timestamp: (日期 Date) 文档最后更新时间。  
  * person\_info\_pid\_ref: (字符串 String) \- *仅在过渡阶段使用*  
    * 描述：引用旧 person\_info 表中的 person\_id，用于数据迁移和平滑过渡。最终将被移除。

#### **3\. 关键功能模块与流程**

1. **自然人识别与账户链接 (Natural Person Identification & Account Linking)**  
   * 当系统与一个新的 平台-用户ID 交互时，会查询 profile\_info 的 platform\_accounts 字段，判断此账户是否已归属于某个已知的 NaturalPersonID。  
   * 若未找到，则以此 平台-用户ID 为锚点，生成新的 NaturalPersonID，并创建新的 profile\_info 文档。  
   * 未来可开发更高级的账户链接策略（如基于行为分析、用户确认等）来合并可能重复的自然人记录。  
2. **画像数据获取与更新 (Profile Data Acquisition & Update)**  
   * **绰号管理 (Sobriquet Management)**：已部分实现。通过LLM分析聊天记录，提取用户在特定群组的绰号，并更新到对应 NaturalPersonID 文档的 sobriquets\_by\_group 字段中。  
   * **身份信息分析 (Identity Profiling)**：通过LLM分析用户言论、或未来通过用户主动提供的信息，提取并更新 identity 字段。  
   * **人格特质分析 (Personality Profiling)**：通过LLM分析用户沟通风格、内容偏好等，推断并更新 personality 字段。  
   * **交互印象生成 (Impression Generation)**：系统定期或基于特定触发条件（如重要对话、情绪波动），总结交互内容，形成并更新 impression 字段。  
   * 所有这些分析模块都将设计为可独立开关、可配置，并由 ProfileManager 统一调度和管理。  
3. **数据导出与Prompt注入 (Data Export & Prompt Injection)**  
   * ProfileManager 将提供一个核心方法，根据当前的交互上下文（如当前平台、当前用户 user\_id、当前群组 group\_id）以及需要注入的画像维度，从 profile\_info 表中提取相关信息。  
   * **聊天记录格式调整**：机器人在处理和记录聊天时，将从 person\_name: message 格式转变为 user\_id: message 格式。person\_name 将不再是主要的对话者标识。  
   * **Prompt注入格式 (示例)**：
     ```
     // 以下是你了解的交互对象的信息：  
     {  
         "<NaturalPersonID_1>": { // 锚定自然人的唯一ID (加盐哈希值)  
             "current_platform_context": { // 仅包含当前交互平台的相关信息  
                 "<platform_name>": { // 例如 "qq"  
                     "<user_id_on_current_platform>": { // 例如 "12345"  
                         "group_nickname": "大佬", // 当前群组中最常用的针对此用户的绰号  
                         "platform_nickname": "用户昵称" // 用户在该平台的通用昵称  
                     }  
                 }  
             },  
             "all_known_sobriquets_summary": "常被称为'大佬', '阿强'", // 一个综合的绰号描述字符串  
             "identity": {  
                 "professional_role": "软件工程师",  
                 "gender_identity": "男性"  
                 // ... 其他已获取的身份信息 ...  
             },  
             "personality": {  
                 "tags": ["幽默", "技术控"]  
                 // ... 其他已获取的人格信息 ...  
             },  
             "impression": [  
                 "对新技术非常感兴趣",  
                 "上次帮助解决了XX问题"  
                 // ... 其他关键印象 ...  
             ]  
         },  
         "<NaturalPersonID_2>": { // 其他在当前上下文中可能相关的用户  
             // ... 类似结构 ...  
         }  
         // ... 可能包含多个上下文相关用户的画像 ...  
     }
 
   * **注入逻辑**：  
     * NaturalPersonID 作为顶层键，确保所有信息都归属于一个确认的自然人。  
     * current\_platform\_context 字段提供机器人进行准确提及所需的信息：当前平台下此用户的 user\_id，以及在该平台和当前群组中最适合称呼TA的绰号 (group\_sobriquet) 和其平台昵称 (platform\_nickname)。机器人可以优先使用 group\_sobriquet。  
     * all\_known\_sobriquets\_summary 提供一个更全面的绰号概览。  
     * identity, personality, impression 等字段为LLM提供更深层次的背景信息，以调整语气、回应策略，并展现更细腻的情感和关系理解。  
4. **拟人化交互的实现**  
   * **准确称呼**：机器人将使用从 current\_platform\_context 获取的绰号或昵称来称呼用户。  
   * **个性化回应**：基于用户的 identity 和 personality，机器人可以调整其用词、语气和幽默感。  
   * **记忆与连贯性**：impression 字段帮助机器人“记住”过去的交互要点，使对话更连贯和有深度。  
   * **动态关系感知**：未来，结合 relationship\_metrics（替代简单的关系值），机器人可以展现更复杂和真实的关系互动模式。

#### **4\. 模块化架构**

* **ProfileManager**: 作为画像系统的核心调度者。  
  * 负责协调各个画像分析子模块（绰号、身份、人格、印象等）的运行。  
  * 管理 profile\_info 表的读写接口（部分底层操作可能仍由 SobriquetDB 或新的 ProfileDB 模块执行）。  
  * 提供统一的数据导出接口，供Prompt注入使用。  
  * 处理 NaturalPersonID 的生成和账户链接逻辑。  
* **分析子模块**: 每个画像维度（如身份识别、人格分析）可以实现为独立的类或函数集，由 ProfileManager 按需调用。这些模块可以是基于规则、启发式算法或LLM的。  
* **数据库交互模块 (SobriquetDB / ProfileDB)**: 封装与MongoDB profile\_info 集合的底层交互，如增删改查等原子操作。

#### **5\. 发展蓝图与计划表**

**阶段一：基础建设与绰号功能完整迁移 (当前正在进行)**

* **目标**: 建立 profile\_info 表的基本框架，将绰号管理功能完全迁移至新表，并实现基于 person\_info\_pid 的 NaturalPersonID 生成。  
* **已完成/进行中**:  
  * profile\_info 表结构初步定义（包含 \_id as NaturalPersonID (基于person\_info\_pid哈希), person\_info\_pid\_ref, platform\_accounts, sobriquets\_by\_group）。  
  * ProfileManager 初版，实现 NaturalPersonID 生成 (基于 person\_info\_pid) 和绰号数据读取。  
  * SobriquetDB 适配 profile\_info 表的绰号写入。  
  * SobriquetManager 更新，使用 ProfileManager 获取绰号数据，使用 SobriquetDB 写入绰号。  
* **待办任务**:  
  1. **完善 ProfileManager 的数据导出方法**: 实现按照新注入格式导出数据，特别是 current\_platform\_context 和 all\_known\_sobriquets\_summary 部分。  
  2. **聊天记录处理调整**: 修改机器人记录和解析聊天记录的逻辑，从 person\_name: message 迁移到 user\_id: message。  
  3. **机器人回复逻辑调整**: 使机器人能够解析Prompt中注入的 current\_platform\_context，并使用正确的 group\_sobriquet 或 platform\_nickname 来提及用户。  
  4. **账户信息填充**: 确保在创建或更新 profile\_info 文档时，platform\_accounts 字段得到正确填充。  
  5. **全面测试与验证**: 验证绰号功能的完整性和Prompt注入的准确性。

**阶段二：身份与人格画像模块引入**

* **目标**: 在 profile\_info 中增加 identity 和 personality 字段，并开发初步的分析模块。  
* **任务**:  
  1. **Schema扩展**: 在 profile\_info 表中添加 identity 和 personality 的字段结构。  
  2. **ProfileManager扩展**:  
     * 增加对 identity 和 personality 数据的读写接口。  
     * 集成初步的身份/人格分析模块（可基于LLM分析用户发言，或简单规则）。  
     * 将 identity 和 personality 信息包含到Prompt导出数据中。  
  3. **模块化设计**: 确保身份和人格分析功能作为可配置、可开关的模块。  
  4. **效果评估**: 初步评估新增画像信息对机器人交互拟人化的影响。

**阶段三：交互印象生成与关系模型初步探索**

* **目标**: 开发交互印象生成模块，并开始探索更丰富的关系表达方式。  
* **任务**:  
  1. **Schema扩展**: 在 profile\_info 表中添加 impression 字段。  
  2. **ProfileManager扩展**:  
     * 增加对 impression 数据的读写接口。  
     * 开发印象生成模块（可基于LLM总结对话，或记录关键交互事件）。  
     * 将 impression 信息包含到Prompt导出数据中。  
  3. **关系模型研究**:  
     * 调研和设计新的 relationship\_metrics 结构，以替代简单的线性关系值。  
     * 探索如何利用 identity, personality, impression 等画像数据来影响这些新的关系指标。  
  4. **机器人行为调整**: 初步尝试让机器人根据印象和更丰富的关系指标调整其行为。

**阶段四：person\_info 表弃用与 NaturalPersonID 体系完全独立**

* **目标**: 完全迁移到以 profile\_info 和 NaturalPersonID 为核心的体系，弃用 person\_info。  
* **任务**:  
  1. **NaturalPersonID 生成机制最终化**:  
     * 将 NaturalPersonID 的生成方式从依赖 person\_info\_pid 哈希，改为基于\*\*首次遇到的平台账户信息（锚定标识）\*\*进行哈希。  
     * 实现或完善账户链接机制，用于处理同一自然人的多个平台账户，确保它们都指向同一个 NaturalPersonID。  
  2. **数据迁移**:  
     * 将 person\_info 表中所有仍有价值的、与自然人相关的核心数据（如历史关系数据、关键标识等，如果尚未迁移）迁移到 profile\_info 表的相应字段中（可能需要新增 relationship\_metrics 等字段）。  
  3. **代码重构**:  
     * 移除代码中对 person\_info\_manager 和 person\_info 表的直接依赖（除了可能的最终数据读取和迁移脚本）。  
     * 确保所有模块（如 RelationshipManager \- 如果它还存在并管理新的 relationship\_metrics）都直接使用 NaturalPersonID 和 ProfileManager。  
     * 移除 profile\_info 表中的 person\_info\_pid\_ref 字段。  
  4. **系统验证**: 全面测试系统在新的ID体系下的稳定性和功能完整性。

**阶段五：持续优化与高级功能拓展**

* **目标**: 不断提升画像的准确性、丰富度和应用的智能化水平。  
* **任务**:  
  * **算法优化**: 持续改进LLM Prompt和用于各项画像分析的算法。  
  * **机器学习集成**: 探索使用更专业的机器学习模型进行特定画像维度的预测和分析。  
  * **用户反馈与自学习**: 引入机制，让机器人可以根据用户反馈调整其对用户画像的理解。  
  * **隐私与伦理**: 持续关注数据隐私保护，确保系统设计和数据使用符合伦理规范。  
  * **新画像维度**: 根据需求，逐步引入更多的画像维度（如用户兴趣、知识领域、沟通风格偏好等）。

#### **6\. 总结**

该高级侧写系统代表了构建真正智能化、个性化和拟人化机器人的重要一步。通过统一的自然人识别、多维度的用户画像和智能的Prompt注入机制，机器人将能够与用户建立更深厚、更真实的情感连接和交互关系。分阶段的实施方案有助于逐步实现这一宏伟蓝图，并在每个阶段都能带来可感知的交互体验提升。
