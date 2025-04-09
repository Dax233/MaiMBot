import os
import sys
import toml
import customtkinter as ctk
from tkinter import messagebox, StringVar, filedialog
import json
import datetime
import shutil

# 设置主题
ctk.set_appearance_mode("System")  # 系统主题
ctk.set_default_color_theme("blue")  # 蓝色主题

# 配置项的中文翻译映射
SECTION_TRANSLATIONS = {
    "inner": "内部配置",
    "bot": "机器人设置",
    "groups": "群组设置",
    "personality": "人格设置",
    "identity": "身份设置",
    "schedule": "日程设置",
    "platforms": "平台设置",
    "response": "回复设置",
    "heartflow": "心流设置",
    "message": "消息设置",
    "willing": "意愿设置",
    "emoji": "表情设置",
    "memory": "记忆设置",
    "mood": "情绪设置",
    "keywords_reaction": "关键词反应",
    "chinese_typo": "中文错别字",
    "response_spliter": "回复分割器",
    "remote": "远程设置",
    "experimental": "实验功能",
    "model": "模型设置",
}

# 配置项的中文描述
CONFIG_DESCRIPTIONS = {
    # bot设置
    "bot.qq": "机器人的QQ号码",
    "bot.nickname": "机器人的昵称",
    "bot.alias_names": "机器人的别名列表",
    # 群组设置
    "groups.talk_allowed": "允许机器人回复消息的群号列表",
    "groups.talk_frequency_down": "降低回复频率的群号列表",
    "groups.ban_user_id": "禁止回复和读取消息的QQ号列表",
    # 人格设置
    "personality.personality_core": "人格核心描述，建议20字以内",
    "personality.personality_sides": "人格特点列表",
    # 身份设置
    "identity.identity_detail": "身份细节描述列表",
    "identity.height": "身高（厘米）",
    "identity.weight": "体重（千克）",
    "identity.age": "年龄",
    "identity.gender": "性别",
    "identity.appearance": "外貌特征",
    # 日程设置
    "schedule.enable_schedule_gen": "是否启用日程表生成",
    "schedule.prompt_schedule_gen": "日程表生成提示词",
    "schedule.schedule_doing_update_interval": "日程表更新间隔（秒）",
    "schedule.schedule_temperature": "日程表温度，建议0.3-0.6",
    "schedule.time_zone": "时区设置",
    # 平台设置
    "platforms.nonebot-qq": "QQ平台适配器链接",
    # 回复设置
    "response.response_mode": "回复策略（heart_flow：心流，reasoning：推理）",
    "response.model_r1_probability": "主要回复模型使用概率",
    "response.model_v3_probability": "次要回复模型使用概率",
    # 心流设置
    "heartflow.sub_heart_flow_update_interval": "子心流更新频率（秒）",
    "heartflow.sub_heart_flow_freeze_time": "子心流冻结时间（秒）",
    "heartflow.sub_heart_flow_stop_time": "子心流停止时间（秒）",
    "heartflow.heart_flow_update_interval": "心流更新频率（秒）",
    # 消息设置
    "message.max_context_size": "获取的上下文数量",
    "message.emoji_chance": "使用表情包的概率",
    "message.thinking_timeout": "思考时间（秒）",
    "message.max_response_length": "回答的最大token数",
    "message.message_buffer": "是否启用消息缓冲器",
    "message.ban_words": "禁用词列表",
    "message.ban_msgs_regex": "禁用消息正则表达式列表",
    # 意愿设置
    "willing.willing_mode": "回复意愿模式",
    "willing.response_willing_amplifier": "回复意愿放大系数",
    "willing.response_interested_rate_amplifier": "回复兴趣度放大系数",
    "willing.down_frequency_rate": "降低回复频率的群组回复意愿降低系数",
    "willing.emoji_response_penalty": "表情包回复惩罚系数",
    # 表情设置
    "emoji.max_emoji_num": "表情包最大数量",
    "emoji.max_reach_deletion": "达到最大数量时是否删除表情包",
    "emoji.check_interval": "检查表情包的时间间隔",
    "emoji.auto_save": "是否保存表情包和图片",
    "emoji.enable_check": "是否启用表情包过滤",
    "emoji.check_prompt": "表情包过滤要求",
    # 记忆设置
    "memory.build_memory_interval": "记忆构建间隔（秒）",
    "memory.build_memory_distribution": "记忆构建分布参数",
    "memory.build_memory_sample_num": "采样数量",
    "memory.build_memory_sample_length": "采样长度",
    "memory.memory_compress_rate": "记忆压缩率",
    "memory.forget_memory_interval": "记忆遗忘间隔（秒）",
    "memory.memory_forget_time": "记忆遗忘时间（小时）",
    "memory.memory_forget_percentage": "记忆遗忘比例",
    "memory.memory_ban_words": "记忆禁用词列表",
    # 情绪设置
    "mood.mood_update_interval": "情绪更新间隔（秒）",
    "mood.mood_decay_rate": "情绪衰减率",
    "mood.mood_intensity_factor": "情绪强度因子",
    # 关键词反应
    "keywords_reaction.enable": "是否启用关键词反应功能",
    # 中文错别字
    "chinese_typo.enable": "是否启用中文错别字生成器",
    "chinese_typo.error_rate": "单字替换概率",
    "chinese_typo.min_freq": "最小字频阈值",
    "chinese_typo.tone_error_rate": "声调错误概率",
    "chinese_typo.word_replace_rate": "整词替换概率",
    # 回复分割器
    "response_spliter.enable_response_spliter": "是否启用回复分割器",
    "response_spliter.response_max_length": "回复允许的最大长度",
    "response_spliter.response_max_sentence_num": "回复允许的最大句子数",
    # 远程设置
    "remote.enable": "是否启用远程统计",
    # 实验功能
    "experimental.enable_friend_chat": "是否启用好友聊天",
    "experimental.pfc_chatting": "是否启用PFC聊天",
    # 模型设置
    "model.llm_reasoning.name": "推理模型名称",
    "model.llm_reasoning.provider": "推理模型提供商",
    "model.llm_reasoning.pri_in": "推理模型输入价格",
    "model.llm_reasoning.pri_out": "推理模型输出价格",
    "model.llm_normal.name": "回复模型名称",
    "model.llm_normal.provider": "回复模型提供商",
    "model.llm_normal.pri_in": "回复模型输入价格",
    "model.llm_normal.pri_out": "回复模型输出价格",
    "model.llm_emotion_judge.name": "表情判断模型名称",
    "model.llm_emotion_judge.provider": "表情判断模型提供商",
    "model.llm_emotion_judge.pri_in": "表情判断模型输入价格",
    "model.llm_emotion_judge.pri_out": "表情判断模型输出价格",
    "model.llm_topic_judge.name": "主题判断模型名称",
    "model.llm_topic_judge.provider": "主题判断模型提供商",
    "model.llm_topic_judge.pri_in": "主题判断模型输入价格",
    "model.llm_topic_judge.pri_out": "主题判断模型输出价格",
    "model.llm_summary_by_topic.name": "概括模型名称",
    "model.llm_summary_by_topic.provider": "概括模型提供商",
    "model.llm_summary_by_topic.pri_in": "概括模型输入价格",
    "model.llm_summary_by_topic.pri_out": "概括模型输出价格",
    "model.moderation.name": "内容审核模型名称",
    "model.moderation.provider": "内容审核模型提供商",
    "model.moderation.pri_in": "内容审核模型输入价格",
    "model.moderation.pri_out": "内容审核模型输出价格",
    "model.vlm.name": "图像识别模型名称",
    "model.vlm.provider": "图像识别模型提供商",
    "model.vlm.pri_in": "图像识别模型输入价格",
    "model.vlm.pri_out": "图像识别模型输出价格",
    "model.embedding.name": "嵌入模型名称",
    "model.embedding.provider": "嵌入模型提供商",
    "model.embedding.pri_in": "嵌入模型输入价格",
    "model.embedding.pri_out": "嵌入模型输出价格",
    "model.llm_observation.name": "观察模型名称",
    "model.llm_observation.provider": "观察模型提供商",
    "model.llm_observation.pri_in": "观察模型输入价格",
    "model.llm_observation.pri_out": "观察模型输出价格",
    "model.llm_sub_heartflow.name": "子心流模型名称",
    "model.llm_sub_heartflow.provider": "子心流模型提供商",
    "model.llm_sub_heartflow.pri_in": "子心流模型输入价格",
    "model.llm_sub_heartflow.pri_out": "子心流模型输出价格",
    "model.llm_heartflow.name": "心流模型名称",
    "model.llm_heartflow.provider": "心流模型提供商",
    "model.llm_heartflow.pri_in": "心流模型输入价格",
    "model.llm_heartflow.pri_out": "心流模型输出价格",
}


# 获取翻译
def get_translation(key):
    return SECTION_TRANSLATIONS.get(key, key)


# 获取配置项描述
def get_description(key):
    return CONFIG_DESCRIPTIONS.get(key, "")


# 获取根目录路径
def get_root_dir():
    try:
        # 获取当前脚本所在目录
        if getattr(sys, "frozen", False):
            # 如果是打包后的应用
            current_dir = os.path.dirname(sys.executable)
        else:
            # 如果是脚本运行
            current_dir = os.path.dirname(os.path.abspath(__file__))

        # 获取根目录（假设当前脚本在temp_utils_ui目录下或者是可执行文件在根目录）
        if os.path.basename(current_dir) == "temp_utils_ui":
            root_dir = os.path.dirname(current_dir)
        else:
            root_dir = current_dir

        # 检查是否存在config目录
        config_dir = os.path.join(root_dir, "config")
        if not os.path.exists(config_dir):
            os.makedirs(config_dir, exist_ok=True)

        return root_dir
    except Exception as e:
        print(f"获取根目录路径失败: {e}")
        # 返回当前目录作为备选
        return os.getcwd()


# 配置文件路径
CONFIG_PATH = os.path.join(get_root_dir(), "config", "bot_config.toml")


# 保存配置
def save_config(config_data):
    try:
        # 首先备份原始配置文件
        if os.path.exists(CONFIG_PATH):
            # 创建备份目录
            backup_dir = os.path.join(os.path.dirname(CONFIG_PATH), "old")
            if not os.path.exists(backup_dir):
                os.makedirs(backup_dir)

            # 生成备份文件名（使用时间戳）
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"bot_config_{timestamp}.toml.bak"
            backup_path = os.path.join(backup_dir, backup_filename)

            # 复制文件
            with open(CONFIG_PATH, "r", encoding="utf-8") as src:
                with open(backup_path, "w", encoding="utf-8") as dst:
                    dst.write(src.read())

        # 保存新配置
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            toml.dump(config_data, f)
        return True
    except Exception as e:
        print(f"保存配置失败: {e}")
        return False


# 加载配置
def load_config():
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return toml.load(f)
        else:
            print(f"配置文件不存在: {CONFIG_PATH}")
            return {}
    except Exception as e:
        print(f"加载配置失败: {e}")
        return {}


# 多行文本输入框
class ScrollableTextFrame(ctk.CTkFrame):
    def __init__(self, master, initial_text="", height=100, width=400, **kwargs):
        super().__init__(master, **kwargs)

        self.text_var = StringVar(value=initial_text)

        # 文本框
        self.text_box = ctk.CTkTextbox(self, height=height, width=width, wrap="word")
        self.text_box.pack(fill="both", expand=True, padx=5, pady=5)
        self.text_box.insert("1.0", initial_text)

        # 绑定更改事件
        self.text_box.bind("<KeyRelease>", self.update_var)

    def update_var(self, event=None):
        self.text_var.set(self.text_box.get("1.0", "end-1c"))

    def get(self):
        return self.text_box.get("1.0", "end-1c")

    def set(self, text):
        self.text_box.delete("1.0", "end")
        self.text_box.insert("1.0", text)
        self.update_var()


# 配置UI
class ConfigUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        # 窗口设置
        self.title("麦麦配置修改器")
        self.geometry("1100x750")

        # 加载配置
        self.config_data = load_config()
        if not self.config_data:
            messagebox.showerror("错误", "无法加载配置文件！将创建空白配置文件。")
            # 如果配置加载失败，创建一个最小化的空配置
            self.config_data = {"inner": {"version": "1.0.0"}}

        # 保存原始配置，用于检测变更
        self.original_config = json.dumps(self.config_data, sort_keys=True)

        # 自动保存状态
        self.auto_save = ctk.BooleanVar(value=False)

        # 创建主框架
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(padx=10, pady=10, fill="both", expand=True)

        # 创建顶部工具栏
        self.create_toolbar()

        # 创建标签和输入框的字典，用于后续保存配置
        self.config_vars = {}

        # 创建左侧导航和右侧内容区域
        self.create_split_view()

        # 创建底部状态栏
        self.status_label = ctk.CTkLabel(self, text="就绪", anchor="w")
        self.status_label.pack(fill="x", padx=10, pady=(0, 5))

        # 绑定关闭事件
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # 设置最小窗口大小
        self.minsize(800, 600)

        # 居中显示窗口
        self.center_window()

    def center_window(self):
        """将窗口居中显示"""
        try:
            self.update_idletasks()
            width = self.winfo_width()
            height = self.winfo_height()
            x = (self.winfo_screenwidth() // 2) - (width // 2)
            y = (self.winfo_screenheight() // 2) - (height // 2)
            self.geometry(f"{width}x{height}+{x}+{y}")
        except Exception as e:
            print(f"居中窗口时出错: {e}")
            # 使用默认位置
            pass

    def create_toolbar(self):
        toolbar = ctk.CTkFrame(self.main_frame, height=40)
        toolbar.pack(fill="x", padx=5, pady=5)

        # 保存按钮
        save_btn = ctk.CTkButton(toolbar, text="保存配置", command=self.save_config, width=100)
        save_btn.pack(side="left", padx=5)

        # 自动保存选项
        auto_save_cb = ctk.CTkCheckBox(toolbar, text="自动保存", variable=self.auto_save)
        auto_save_cb.pack(side="left", padx=15)

        # 重新加载按钮
        reload_btn = ctk.CTkButton(toolbar, text="重新加载", command=self.reload_config, width=100)
        reload_btn.pack(side="left", padx=5)

        # 手动备份按钮
        backup_btn = ctk.CTkButton(toolbar, text="手动备份", command=self.backup_config, width=100)
        backup_btn.pack(side="left", padx=5)

        # 查看备份按钮
        view_backup_btn = ctk.CTkButton(toolbar, text="查看备份", command=self.view_backups, width=100)
        view_backup_btn.pack(side="left", padx=5)

        # 导入导出菜单按钮
        import_export_btn = ctk.CTkButton(toolbar, text="导入/导出", command=self.show_import_export_menu, width=100)
        import_export_btn.pack(side="left", padx=5)

        # 关于按钮
        about_btn = ctk.CTkButton(toolbar, text="关于", command=self.show_about, width=80)
        about_btn.pack(side="right", padx=5)

    def create_split_view(self):
        # 创建分隔视图框架
        split_frame = ctk.CTkFrame(self.main_frame)
        split_frame.pack(fill="both", expand=True, padx=5, pady=5)

        # 左侧分类列表
        self.category_frame = ctk.CTkFrame(split_frame, width=220)
        self.category_frame.pack(side="left", fill="y", padx=(0, 5), pady=0)
        self.category_frame.pack_propagate(False)  # 固定宽度

        # 右侧内容区域
        self.content_frame = ctk.CTkScrollableFrame(split_frame)
        self.content_frame.pack(side="right", fill="both", expand=True)

        # 创建类别列表
        self.create_category_list()

    def create_category_list(self):
        # 标题和搜索框
        header_frame = ctk.CTkFrame(self.category_frame)
        header_frame.pack(fill="x", padx=5, pady=(10, 5))

        ctk.CTkLabel(header_frame, text="配置分类", font=("Arial", 14, "bold")).pack(side="left", padx=5, pady=5)

        # 搜索按钮
        search_btn = ctk.CTkButton(
            header_frame,
            text="🔍",
            width=30,
            command=self.show_search_dialog,
            fg_color="transparent",
            hover_color=("gray80", "gray30"),
        )
        search_btn.pack(side="right", padx=5, pady=5)

        # 分类按钮
        self.category_buttons = {}
        self.active_category = None

        # 分类按钮容器
        buttons_frame = ctk.CTkScrollableFrame(self.category_frame, height=600)
        buttons_frame.pack(fill="both", expand=True, padx=5, pady=5)

        for section in self.config_data:
            # 跳过inner部分，这个不应该被用户修改
            if section == "inner":
                continue

            # 获取翻译
            section_name = f"{section} ({get_translation(section)})"

            btn = ctk.CTkButton(
                buttons_frame,
                text=section_name,
                fg_color="transparent",
                text_color=("gray10", "gray90"),
                anchor="w",
                height=35,
                command=lambda s=section: self.show_category(s),
            )
            btn.pack(fill="x", padx=5, pady=2)
            self.category_buttons[section] = btn

        # 默认显示第一个分类
        first_section = next((s for s in self.config_data.keys() if s != "inner"), None)
        if first_section:
            self.show_category(first_section)

    def show_category(self, category):
        # 清除当前内容
        for widget in self.content_frame.winfo_children():
            widget.destroy()

        # 更新按钮状态
        for section, btn in self.category_buttons.items():
            if section == category:
                btn.configure(fg_color=("gray75", "gray25"))
                self.active_category = section
            else:
                btn.configure(fg_color="transparent")

        # 获取翻译
        category_name = f"{category} ({get_translation(category)})"

        # 添加标题
        ctk.CTkLabel(self.content_frame, text=f"{category_name} 配置", font=("Arial", 16, "bold")).pack(
            anchor="w", padx=10, pady=(5, 15)
        )

        # 添加配置项
        self.add_config_section(self.content_frame, category, self.config_data[category])

    def add_config_section(self, parent, section_path, section_data, indent=0):
        # 递归添加配置项
        for key, value in section_data.items():
            full_path = f"{section_path}.{key}" if indent > 0 else f"{section_path}.{key}"

            # 获取描述
            description = get_description(full_path)

            if isinstance(value, dict):
                # 如果是字典，创建一个分组框架并递归添加子项
                group_frame = ctk.CTkFrame(parent)
                group_frame.pack(fill="x", expand=True, padx=10, pady=10)

                # 添加标题
                header_frame = ctk.CTkFrame(group_frame, fg_color=("gray85", "gray25"))
                header_frame.pack(fill="x", padx=0, pady=0)

                label = ctk.CTkLabel(header_frame, text=f"{key}", font=("Arial", 13, "bold"), anchor="w")
                label.pack(anchor="w", padx=10, pady=5)

                # 如果有描述，添加提示图标
                if description:
                    # 创建工具提示窗口显示函数
                    def show_tooltip(event, text, widget):
                        x, y, _, _ = widget.bbox("all")
                        x += widget.winfo_rootx() + 25
                        y += widget.winfo_rooty() + 25

                        # 创建工具提示窗口
                        tipwindow = ctk.CTkToplevel(widget)
                        tipwindow.wm_overrideredirect(True)
                        tipwindow.wm_geometry(f"+{x}+{y}")
                        tipwindow.lift()

                        label = ctk.CTkLabel(tipwindow, text=text, justify="left", wraplength=300)
                        label.pack(padx=5, pady=5)

                        # 自动关闭
                        def close_tooltip():
                            tipwindow.destroy()

                        widget.after(3000, close_tooltip)
                        return tipwindow

                    # 在标题后添加提示图标
                    tip_label = ctk.CTkLabel(
                        header_frame, text="ℹ️", font=("Arial", 12), text_color="light blue", width=20
                    )
                    tip_label.pack(side="right", padx=5)

                    # 绑定鼠标悬停事件
                    tip_label.bind("<Enter>", lambda e, t=description, w=tip_label: show_tooltip(e, t, w))

                # 添加内容
                content_frame = ctk.CTkFrame(group_frame)
                content_frame.pack(fill="x", expand=True, padx=5, pady=5)

                self.add_config_section(content_frame, full_path, value, indent + 1)

            elif isinstance(value, list):
                # 如果是列表，创建一个文本框用于编辑JSON格式的列表
                frame = ctk.CTkFrame(parent)
                frame.pack(fill="x", expand=True, padx=5, pady=5)

                # 标签和输入框在一行
                label_frame = ctk.CTkFrame(frame)
                label_frame.pack(fill="x", padx=5, pady=(5, 0))

                # 标签包含描述提示
                label_text = f"{key}:"
                if description:
                    label_text = f"{key}: ({description})"

                label = ctk.CTkLabel(label_frame, text=label_text, font=("Arial", 12), anchor="w")
                label.pack(anchor="w", padx=5 + indent * 10, pady=0)

                # 添加提示信息
                info_label = ctk.CTkLabel(label_frame, text="(列表格式: JSON)", font=("Arial", 9), text_color="gray50")
                info_label.pack(anchor="w", padx=5 + indent * 10, pady=(0, 5))

                # 确定文本框高度，根据列表项数量决定
                list_height = max(100, min(len(value) * 20 + 40, 200))

                # 将列表转换为JSON字符串，美化格式
                json_str = json.dumps(value, ensure_ascii=False, indent=2)

                # 使用多行文本框
                text_frame = ScrollableTextFrame(frame, initial_text=json_str, height=list_height, width=550)
                text_frame.pack(fill="x", padx=10 + indent * 10, pady=5)

                self.config_vars[full_path] = (text_frame.text_var, "list")

                # 绑定变更事件，用于自动保存
                text_frame.text_box.bind("<FocusOut>", lambda e, path=full_path: self.on_field_change(path))

            elif isinstance(value, bool):
                # 如果是布尔值，创建一个复选框
                frame = ctk.CTkFrame(parent)
                frame.pack(fill="x", expand=True, padx=5, pady=5)

                var = ctk.BooleanVar(value=value)
                self.config_vars[full_path] = (var, "bool")

                # 复选框文本包含描述
                checkbox_text = key
                if description:
                    checkbox_text = f"{key} ({description})"

                checkbox = ctk.CTkCheckBox(
                    frame, text=checkbox_text, variable=var, command=lambda path=full_path: self.on_field_change(path)
                )
                checkbox.pack(anchor="w", padx=10 + indent * 10, pady=5)

            elif isinstance(value, (int, float)):
                # 如果是数字，创建一个数字输入框
                frame = ctk.CTkFrame(parent)
                frame.pack(fill="x", expand=True, padx=5, pady=5)

                # 标签包含描述
                label_text = f"{key}:"
                if description:
                    label_text = f"{key}: ({description})"

                label = ctk.CTkLabel(frame, text=label_text, font=("Arial", 12), anchor="w")
                label.pack(anchor="w", padx=10 + indent * 10, pady=(5, 0))

                var = StringVar(value=str(value))
                self.config_vars[full_path] = (var, "number", type(value))

                # 判断数值的长度，决定输入框宽度
                entry_width = max(200, min(len(str(value)) * 15, 300))

                entry = ctk.CTkEntry(frame, width=entry_width, textvariable=var)
                entry.pack(anchor="w", padx=10 + indent * 10, pady=5)

                # 绑定变更事件，用于自动保存
                entry.bind("<FocusOut>", lambda e, path=full_path: self.on_field_change(path))

            else:
                # 对于字符串，创建一个文本输入框
                frame = ctk.CTkFrame(parent)
                frame.pack(fill="x", expand=True, padx=5, pady=5)

                # 标签包含描述
                label_text = f"{key}:"
                if description:
                    label_text = f"{key}: ({description})"

                label = ctk.CTkLabel(frame, text=label_text, font=("Arial", 12), anchor="w")
                label.pack(anchor="w", padx=10 + indent * 10, pady=(5, 0))

                var = StringVar(value=str(value))
                self.config_vars[full_path] = (var, "string")

                # 判断文本长度，决定输入框的类型和大小
                text_len = len(str(value))

                if text_len > 80 or "\n" in str(value):
                    # 对于长文本或多行文本，使用多行文本框
                    text_height = max(80, min(str(value).count("\n") * 20 + 40, 150))

                    text_frame = ScrollableTextFrame(frame, initial_text=str(value), height=text_height, width=550)
                    text_frame.pack(fill="x", padx=10 + indent * 10, pady=5)
                    self.config_vars[full_path] = (text_frame.text_var, "string")

                    # 绑定变更事件，用于自动保存
                    text_frame.text_box.bind("<FocusOut>", lambda e, path=full_path: self.on_field_change(path))
                else:
                    # 对于短文本，使用单行输入框
                    # 根据内容长度动态调整输入框宽度
                    entry_width = max(400, min(text_len * 10, 550))

                    entry = ctk.CTkEntry(frame, width=entry_width, textvariable=var)
                    entry.pack(anchor="w", padx=10 + indent * 10, pady=5, fill="x")

                    # 绑定变更事件，用于自动保存
                    entry.bind("<FocusOut>", lambda e, path=full_path: self.on_field_change(path))

    def on_field_change(self, path):
        """当字段值改变时调用，用于自动保存"""
        if self.auto_save.get():
            self.save_config(show_message=False)
            self.status_label.configure(text=f"已自动保存更改 ({path})")

    def save_config(self, show_message=True):
        """保存配置文件"""
        # 更新配置数据
        updated = False
        _error_path = None

        for path, (var, var_type, *args) in self.config_vars.items():
            parts = path.split(".")

            # 如果路径有多层级
            target = self.config_data
            for p in parts[:-1]:
                if p not in target:
                    target[p] = {}
                target = target[p]

            # 根据变量类型更新值
            try:
                if var_type == "bool":
                    if target[parts[-1]] != var.get():
                        target[parts[-1]] = var.get()
                        updated = True
                elif var_type == "number":
                    # 获取原始类型（int或float）
                    num_type = args[0] if args else int
                    new_value = num_type(var.get())
                    if target[parts[-1]] != new_value:
                        target[parts[-1]] = new_value
                        updated = True

                elif var_type == "list":
                    # 解析JSON字符串为列表
                    new_value = json.loads(var.get())
                    if json.dumps(target[parts[-1]], sort_keys=True) != json.dumps(new_value, sort_keys=True):
                        target[parts[-1]] = new_value
                        updated = True

                else:
                    if target[parts[-1]] != var.get():
                        target[parts[-1]] = var.get()
                        updated = True
            except ValueError as e:
                if show_message:
                    messagebox.showerror("格式错误", str(e))
                else:
                    self.status_label.configure(text=f"保存失败: {e}")
                return False

        if not updated and show_message:
            self.status_label.configure(text="无更改，无需保存")
            return True

        # 保存配置
        if save_config(self.config_data):
            if show_message:
                messagebox.showinfo("成功", "配置已保存！")
            self.original_config = json.dumps(self.config_data, sort_keys=True)
            return True
        else:
            if show_message:
                messagebox.showerror("错误", "保存配置失败！")
            else:
                self.status_label.configure(text="保存失败！")
            return False

    def reload_config(self):
        """重新加载配置"""
        if self.check_unsaved_changes():
            self.config_data = load_config()
            if not self.config_data:
                messagebox.showerror("错误", "无法加载配置文件！")
                return

            # 保存原始配置，用于检测变更
            self.original_config = json.dumps(self.config_data, sort_keys=True)

            # 重新显示当前分类
            self.show_category(self.active_category)

            self.status_label.configure(text="配置已重新加载")

    def check_unsaved_changes(self):
        """检查是否有未保存的更改"""
        # 临时更新配置数据以进行比较
        temp_config = self.config_data.copy()

        try:
            for path, (var, var_type, *args) in self.config_vars.items():
                parts = path.split(".")

                target = temp_config
                for p in parts[:-1]:
                    target = target[p]

                if var_type == "bool":
                    target[parts[-1]] = var.get()
                elif var_type == "number":
                    num_type = args[0] if args else int
                    target[parts[-1]] = num_type(var.get())
                elif var_type == "list":
                    target[parts[-1]] = json.loads(var.get())
                else:
                    target[parts[-1]] = var.get()
        except (ValueError, json.JSONDecodeError):
            # 如果有无效输入，认为有未保存更改
            return False

        # 比较原始配置和当前配置
        current_config = json.dumps(temp_config, sort_keys=True)

        if current_config != self.original_config:
            result = messagebox.askyesnocancel("未保存的更改", "有未保存的更改，是否保存？", icon="warning")

            if result is None:  # 取消
                return False
            elif result:  # 是
                return self.save_config()

        return True

    def show_about(self):
        """显示关于对话框"""
        about_window = ctk.CTkToplevel(self)
        about_window.title("关于")
        about_window.geometry("400x200")
        about_window.resizable(False, False)
        about_window.grab_set()  # 模态对话框

        # 居中
        x = self.winfo_x() + (self.winfo_width() - 400) // 2
        y = self.winfo_y() + (self.winfo_height() - 200) // 2
        about_window.geometry(f"+{x}+{y}")

        # 内容
        ctk.CTkLabel(about_window, text="麦麦配置修改器", font=("Arial", 16, "bold")).pack(pady=(20, 10))

        ctk.CTkLabel(about_window, text="用于修改MaiBot-Core的配置文件\n配置文件路径: config/bot_config.toml").pack(
            pady=5
        )

        ctk.CTkLabel(about_window, text="注意: 修改配置前请备份原始配置文件", text_color=("red", "light coral")).pack(
            pady=5
        )

        ctk.CTkButton(about_window, text="确定", command=about_window.destroy, width=100).pack(pady=15)

    def on_closing(self):
        """关闭窗口前检查未保存更改"""
        if self.check_unsaved_changes():
            self.destroy()

    def backup_config(self):
        """手动备份当前配置文件"""
        try:
            # 检查配置文件是否存在
            if not os.path.exists(CONFIG_PATH):
                messagebox.showerror("错误", "配置文件不存在！")
                return False

            # 创建备份目录
            backup_dir = os.path.join(os.path.dirname(CONFIG_PATH), "old")
            if not os.path.exists(backup_dir):
                os.makedirs(backup_dir)

            # 生成备份文件名（使用时间戳）
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"bot_config_{timestamp}.toml.bak"
            backup_path = os.path.join(backup_dir, backup_filename)

            # 复制文件
            with open(CONFIG_PATH, "r", encoding="utf-8") as src:
                with open(backup_path, "w", encoding="utf-8") as dst:
                    dst.write(src.read())

            messagebox.showinfo("成功", f"配置已备份到:\n{backup_path}")
            self.status_label.configure(text=f"手动备份已创建: {backup_filename}")
            return True
        except Exception as e:
            messagebox.showerror("备份失败", f"备份配置文件失败: {e}")
            return False

    def view_backups(self):
        """查看备份文件列表"""
        # 创建备份目录
        backup_dir = os.path.join(os.path.dirname(CONFIG_PATH), "old")
        if not os.path.exists(backup_dir):
            os.makedirs(backup_dir)

        # 查找备份文件
        backup_files = []
        for filename in os.listdir(backup_dir):
            if filename.startswith("bot_config_") and filename.endswith(".toml.bak"):
                backup_path = os.path.join(backup_dir, filename)
                mod_time = os.path.getmtime(backup_path)
                backup_files.append((filename, backup_path, mod_time))

        if not backup_files:
            messagebox.showinfo("提示", "未找到备份文件")
            return

        # 按修改时间排序，最新的在前
        backup_files.sort(key=lambda x: x[2], reverse=True)

        # 创建备份查看窗口
        backup_window = ctk.CTkToplevel(self)
        backup_window.title("备份文件")
        backup_window.geometry("600x400")
        backup_window.grab_set()  # 模态对话框

        # 居中
        x = self.winfo_x() + (self.winfo_width() - 600) // 2
        y = self.winfo_y() + (self.winfo_height() - 400) // 2
        backup_window.geometry(f"+{x}+{y}")

        # 创建说明标签
        ctk.CTkLabel(backup_window, text="备份文件列表 (双击可恢复)", font=("Arial", 14, "bold")).pack(
            pady=(10, 5), padx=10, anchor="w"
        )

        # 创建列表框
        backup_frame = ctk.CTkScrollableFrame(backup_window, width=580, height=300)
        backup_frame.pack(padx=10, pady=10, fill="both", expand=True)

        # 添加备份文件项
        for _i, (filename, filepath, mod_time) in enumerate(backup_files):
            # 格式化时间为可读格式
            time_str = datetime.datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M:%S")

            # 创建一个框架用于每个备份项
            item_frame = ctk.CTkFrame(backup_frame)
            item_frame.pack(fill="x", padx=5, pady=5)

            # 显示备份文件信息
            ctk.CTkLabel(item_frame, text=f"{time_str}", font=("Arial", 12, "bold"), width=200).pack(
                side="left", padx=10, pady=10
            )

            # 文件名
            name_label = ctk.CTkLabel(item_frame, text=filename, font=("Arial", 11))
            name_label.pack(side="left", fill="x", expand=True, padx=5, pady=10)

            # 恢复按钮
            restore_btn = ctk.CTkButton(
                item_frame, text="恢复", width=80, command=lambda path=filepath: self.restore_backup(path)
            )
            restore_btn.pack(side="right", padx=10, pady=10)

            # 绑定双击事件
            for widget in (item_frame, name_label):
                widget.bind("<Double-1>", lambda e, path=filepath: self.restore_backup(path))

        # 关闭按钮
        ctk.CTkButton(backup_window, text="关闭", command=backup_window.destroy, width=100).pack(pady=10)

    def restore_backup(self, backup_path):
        """从备份文件恢复配置"""
        if not os.path.exists(backup_path):
            messagebox.showerror("错误", "备份文件不存在！")
            return False

        # 确认还原
        confirm = messagebox.askyesno(
            "确认",
            f"确定要从以下备份文件恢复配置吗？\n{os.path.basename(backup_path)}\n\n这将覆盖当前的配置！",
            icon="warning",
        )

        if not confirm:
            return False

        try:
            # 先备份当前配置
            self.backup_config()

            # 恢复配置
            with open(backup_path, "r", encoding="utf-8") as src:
                with open(CONFIG_PATH, "w", encoding="utf-8") as dst:
                    dst.write(src.read())

            messagebox.showinfo("成功", "配置已从备份恢复！")

            # 重新加载配置
            self.reload_config()
            return True
        except Exception as e:
            messagebox.showerror("恢复失败", f"恢复配置失败: {e}")
            return False

    def show_search_dialog(self):
        """显示搜索对话框"""
        try:
            search_window = ctk.CTkToplevel(self)
            search_window.title("搜索配置项")
            search_window.geometry("500x400")
            search_window.grab_set()  # 模态对话框

            # 居中
            x = self.winfo_x() + (self.winfo_width() - 500) // 2
            y = self.winfo_y() + (self.winfo_height() - 400) // 2
            search_window.geometry(f"+{x}+{y}")

            # 搜索框
            search_frame = ctk.CTkFrame(search_window)
            search_frame.pack(fill="x", padx=10, pady=10)

            search_var = StringVar()
            search_entry = ctk.CTkEntry(
                search_frame, placeholder_text="输入关键词搜索...", width=380, textvariable=search_var
            )
            search_entry.pack(side="left", padx=5, pady=5, fill="x", expand=True)

            # 结果列表框
            results_frame = ctk.CTkScrollableFrame(search_window, width=480, height=300)
            results_frame.pack(padx=10, pady=5, fill="both", expand=True)

            # 搜索结果标签
            results_label = ctk.CTkLabel(results_frame, text="请输入关键词进行搜索", anchor="w")
            results_label.pack(fill="x", padx=10, pady=10)

            # 结果项列表
            results_items = []

            # 搜索函数
            def perform_search():
                # 清除之前的结果
                for item in results_items:
                    item.destroy()
                results_items.clear()

                keyword = search_var.get().lower()
                if not keyword:
                    results_label.configure(text="请输入关键词进行搜索")
                    return

                # 收集所有匹配的配置项
                matches = []

                def search_config(section_path, config_data):
                    for key, value in config_data.items():
                        full_path = f"{section_path}.{key}" if section_path else key

                        # 检查键名是否匹配
                        if keyword in key.lower():
                            matches.append((full_path, value))

                        # 检查描述是否匹配
                        description = get_description(full_path)
                        if description and keyword in description.lower():
                            matches.append((full_path, value))

                        # 检查值是否匹配(仅字符串类型)
                        if isinstance(value, str) and keyword in value.lower():
                            matches.append((full_path, value))

                        # 递归搜索子项
                        if isinstance(value, dict):
                            search_config(full_path, value)

                # 开始搜索
                search_config("", self.config_data)

                if not matches:
                    results_label.configure(text=f"未找到包含 '{keyword}' 的配置项")
                    return

                results_label.configure(text=f"找到 {len(matches)} 个匹配项")

                # 显示搜索结果
                for full_path, value in matches:
                    # 创建一个框架用于每个结果项
                    item_frame = ctk.CTkFrame(results_frame)
                    item_frame.pack(fill="x", padx=5, pady=3)
                    results_items.append(item_frame)

                    # 配置项路径
                    path_parts = full_path.split(".")
                    section = path_parts[0] if len(path_parts) > 0 else ""
                    _key = path_parts[-1] if len(path_parts) > 0 else ""

                    # 获取描述
                    description = get_description(full_path)
                    desc_text = f" ({description})" if description else ""

                    # 显示完整路径
                    path_label = ctk.CTkLabel(
                        item_frame,
                        text=f"{full_path}{desc_text}",
                        font=("Arial", 11, "bold"),
                        anchor="w",
                        wraplength=450,
                    )
                    path_label.pack(anchor="w", padx=10, pady=(5, 0), fill="x")

                    # 显示值的预览（截断过长的值）
                    value_str = str(value)
                    if len(value_str) > 50:
                        value_str = value_str[:50] + "..."

                    value_label = ctk.CTkLabel(
                        item_frame, text=f"值: {value_str}", font=("Arial", 10), anchor="w", wraplength=450
                    )
                    value_label.pack(anchor="w", padx=10, pady=(0, 5), fill="x")

                    # 添加"转到"按钮
                    goto_btn = ctk.CTkButton(
                        item_frame,
                        text="转到",
                        width=60,
                        height=25,
                        command=lambda s=section: self.goto_config_item(s, search_window),
                    )
                    goto_btn.pack(side="right", padx=10, pady=5)

                    # 绑定双击事件
                    for widget in (item_frame, path_label, value_label):
                        widget.bind("<Double-1>", lambda e, s=section: self.goto_config_item(s, search_window))

            # 搜索按钮
            search_button = ctk.CTkButton(search_frame, text="搜索", width=80, command=perform_search)
            search_button.pack(side="right", padx=5, pady=5)

            # 绑定回车键
            search_entry.bind("<Return>", lambda e: perform_search())

            # 初始聚焦到搜索框
            search_window.after(100, lambda: self.safe_focus(search_entry))
        except Exception as e:
            print(f"显示搜索对话框出错: {e}")
            messagebox.showerror("错误", f"显示搜索对话框失败: {e}")

    def safe_focus(self, widget):
        """安全地设置焦点，避免应用崩溃"""
        try:
            if widget.winfo_exists():
                widget.focus_set()
        except Exception as e:
            print(f"设置焦点出错: {e}")
            # 忽略错误

    def goto_config_item(self, section, dialog=None):
        """跳转到指定的配置项"""
        if dialog:
            dialog.destroy()

        # 切换到相应的分类
        if section in self.category_buttons:
            self.show_category(section)

    def show_import_export_menu(self):
        """显示导入导出菜单"""
        menu_window = ctk.CTkToplevel(self)
        menu_window.title("导入/导出配置")
        menu_window.geometry("300x200")
        menu_window.resizable(False, False)
        menu_window.grab_set()  # 模态对话框

        # 居中
        x = self.winfo_x() + (self.winfo_width() - 300) // 2
        y = self.winfo_y() + (self.winfo_height() - 200) // 2
        menu_window.geometry(f"+{x}+{y}")

        # 创建按钮
        ctk.CTkLabel(menu_window, text="配置导入导出", font=("Arial", 16, "bold")).pack(pady=(20, 10))

        # 导出按钮
        export_btn = ctk.CTkButton(
            menu_window, text="导出配置到文件", command=lambda: self.export_config(menu_window), width=200
        )
        export_btn.pack(pady=10)

        # 导入按钮
        import_btn = ctk.CTkButton(
            menu_window, text="从文件导入配置", command=lambda: self.import_config(menu_window), width=200
        )
        import_btn.pack(pady=10)

        # 取消按钮
        cancel_btn = ctk.CTkButton(menu_window, text="取消", command=menu_window.destroy, width=100)
        cancel_btn.pack(pady=10)

    def export_config(self, parent_window=None):
        """导出配置到文件"""
        # 先保存当前配置
        if not self.save_config(show_message=False):
            if messagebox.askyesno("警告", "当前配置存在错误，是否仍要导出？"):
                pass
            else:
                return

        # 选择保存位置
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"bot_config_export_{timestamp}.toml"

        file_path = filedialog.asksaveasfilename(
            title="导出配置",
            filetypes=[("TOML 文件", "*.toml"), ("所有文件", "*.*")],
            defaultextension=".toml",
            initialfile=default_filename,
        )

        if not file_path:
            return

        try:
            # 复制当前配置文件到选择的位置
            shutil.copy2(CONFIG_PATH, file_path)

            messagebox.showinfo("成功", f"配置已导出到:\n{file_path}")
            self.status_label.configure(text=f"配置已导出到: {file_path}")

            if parent_window:
                parent_window.destroy()

            return True
        except Exception as e:
            messagebox.showerror("导出失败", f"导出配置失败: {e}")
            return False

    def import_config(self, parent_window=None):
        """从文件导入配置"""
        # 先检查是否有未保存的更改
        if not self.check_unsaved_changes():
            return

        # 选择要导入的文件
        file_path = filedialog.askopenfilename(
            title="导入配置", filetypes=[("TOML 文件", "*.toml"), ("所有文件", "*.*")]
        )

        if not file_path:
            return

        try:
            # 尝试加载TOML文件以验证格式
            with open(file_path, "r", encoding="utf-8") as f:
                import_data = toml.load(f)

            # 验证导入文件的基本结构
            if "inner" not in import_data:
                raise ValueError("导入的配置文件没有inner部分，格式不正确")

            if "version" not in import_data["inner"]:
                raise ValueError("导入的配置文件没有版本信息，格式不正确")

            # 确认导入
            confirm = messagebox.askyesno(
                "确认导入", f"确定要导入此配置文件吗？\n{file_path}\n\n这将替换当前的配置！", icon="warning"
            )

            if not confirm:
                return

            # 先备份当前配置
            self.backup_config()

            # 复制导入的文件到配置位置
            shutil.copy2(file_path, CONFIG_PATH)

            messagebox.showinfo("成功", "配置已导入，请重新加载以应用更改")

            # 重新加载配置
            self.reload_config()

            if parent_window:
                parent_window.destroy()

            return True
        except Exception as e:
            messagebox.showerror("导入失败", f"导入配置失败: {e}")
            return False


# 主函数
def main():
    try:
        app = ConfigUI()
        app.mainloop()
    except Exception as e:
        print(f"程序发生错误: {e}")
        # 显示错误对话框

        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("程序错误", f"程序运行时发生错误:\n{e}")
        root.destroy()


if __name__ == "__main__":
    main()
