# import customtkinter as ctk
# import subprocess
# import threading
# import queue
# import re
# import os
# import signal
# from collections import deque
# import sys

# # 设置应用的外观模式和默认颜色主题
# ctk.set_appearance_mode("dark")
# ctk.set_default_color_theme("blue")


# class LogViewerApp(ctk.CTk):
#     """日志查看器应用的主类，继承自customtkinter的CTk类"""

#     def __init__(self):
#         """初始化日志查看器应用的界面和状态"""
#         super().__init__()
#         self.title("日志查看器")
#         self.geometry("1200x800")

#         # 标记GUI是否运行中
#         self.is_running = True

#         # 程序关闭时的清理操作
#         self.protocol("WM_DELETE_WINDOW", self._on_closing)

#         # 初始化进程、日志队列、日志数据等变量
#         self.process = None
#         self.log_queue = queue.Queue()
#         self.log_data = deque(maxlen=10000)  # 使用固定长度队列
#         self.available_levels = set()
#         self.available_modules = set()
#         self.sorted_modules = []
#         self.module_checkboxes = {}  # 存储模块复选框的字典

#         # 日志颜色配置
#         self.color_config = {
#             "time": "#888888",
#             "DEBUG": "#2196F3",
#             "INFO": "#4CAF50",
#             "WARNING": "#FF9800",
#             "ERROR": "#F44336",
#             "module": "#D4D0AB",
#             "default": "#FFFFFF",
#         }

#         # 列可见性配置
#         self.column_visibility = {"show_time": True, "show_level": True, "show_module": True}

#         # 选中的日志等级和模块
#         self.selected_levels = set()
#         self.selected_modules = set()

#         # 创建界面组件并启动日志队列处理
#         self.create_widgets()
#         self.after(100, self.process_log_queue)

#     def create_widgets(self):
#         """创建应用界面的各个组件"""
#         self.grid_columnconfigure(0, weight=1)
#         self.grid_rowconfigure(1, weight=1)

#         # 控制面板
#         control_frame = ctk.CTkFrame(self)
#         control_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=5)

#         self.start_btn = ctk.CTkButton(control_frame, text="启动", command=self.start_process)
#         self.start_btn.pack(side="left", padx=5)

#         self.stop_btn = ctk.CTkButton(control_frame, text="停止", command=self.stop_process, state="disabled")
#         self.stop_btn.pack(side="left", padx=5)

#         self.clear_btn = ctk.CTkButton(control_frame, text="清屏", command=self.clear_logs)
#         self.clear_btn.pack(side="left", padx=5)

#         column_filter_frame = ctk.CTkFrame(control_frame)
#         column_filter_frame.pack(side="left", padx=20)

#         self.time_check = ctk.CTkCheckBox(column_filter_frame, text="显示时间", command=self.refresh_logs)
#         self.time_check.pack(side="left", padx=5)
#         self.time_check.select()

#         self.level_check = ctk.CTkCheckBox(column_filter_frame, text="显示等级", command=self.refresh_logs)
#         self.level_check.pack(side="left", padx=5)
#         self.level_check.select()

#         self.module_check = ctk.CTkCheckBox(column_filter_frame, text="显示模块", command=self.refresh_logs)
#         self.module_check.pack(side="left", padx=5)
#         self.module_check.select()

#         # 筛选面板
#         filter_frame = ctk.CTkFrame(self)
#         filter_frame.grid(row=0, column=1, rowspan=2, sticky="ns", padx=5)

#         ctk.CTkLabel(filter_frame, text="日志等级筛选").pack(pady=5)
#         self.level_scroll = ctk.CTkScrollableFrame(filter_frame, width=150, height=200)
#         self.level_scroll.pack(fill="both", expand=True, padx=5)

#         ctk.CTkLabel(filter_frame, text="模块筛选").pack(pady=5)
#         self.module_filter_entry = ctk.CTkEntry(filter_frame, placeholder_text="输入模块过滤词")
#         self.module_filter_entry.pack(pady=5)
#         self.module_filter_entry.bind("<KeyRelease>", self.update_module_filter)

#         self.module_scroll = ctk.CTkScrollableFrame(filter_frame, width=300, height=200)
#         self.module_scroll.pack(fill="both", expand=True, padx=5)

#         self.log_text = ctk.CTkTextbox(self, wrap="word")
#         self.log_text.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)

#         self.init_text_tags()

#     def update_module_filter(self, event):
#         """根据模块过滤词更新模块复选框的显示"""
#         filter_text = self.module_filter_entry.get().strip().lower()
#         for module, checkbox in self.module_checkboxes.items():
#             if filter_text in module.lower():
#                 checkbox.pack(anchor="w", padx=5, pady=2)
#             else:
#                 checkbox.pack_forget()

#     def update_filters(self, level, module):
#         """更新日志等级和模块的筛选器"""
#         if level not in self.available_levels:
#             self.available_levels.add(level)
#             self.add_checkbox(self.level_scroll, level, "level")

#         module_key = self.get_module_key(module)
#         if module_key not in self.available_modules:
#             self.available_modules.add(module_key)
#             self.sorted_modules = sorted(self.available_modules, key=lambda x: x.lower())
#             self.rebuild_module_checkboxes()

#     def rebuild_module_checkboxes(self):
#         """重新构建模块复选框"""
#         # 清空现有复选框
#         for widget in self.module_scroll.winfo_children():
#             widget.destroy()
#         self.module_checkboxes.clear()

#         # 重建排序后的复选框
#         for module in self.sorted_modules:
#             self.add_checkbox(self.module_scroll, module, "module")

#     def add_checkbox(self, parent, text, type_):
#         """在指定父组件中添加复选框"""

#         def update_filter():
#             current = cb.get()
#             if type_ == "level":
#                 (self.selected_levels.add if current else self.selected_levels.discard)(text)
#             else:
#                 (self.selected_modules.add if current else self.selected_modules.discard)(text)
#             self.refresh_logs()

#         cb = ctk.CTkCheckBox(parent, text=text, command=update_filter)
#         cb.select()  # 初始选中

#         # 手动同步初始状态到集合（关键修复）
#         if type_ == "level":
#             self.selected_levels.add(text)
#         else:
#             self.selected_modules.add(text)

#         if type_ == "module":
#             self.module_checkboxes[text] = cb
#         cb.pack(anchor="w", padx=5, pady=2)
#         return cb

#     def check_filter(self, entry):
#         """检查日志条目是否符合当前筛选条件"""
#         level_ok = not self.selected_levels or entry["level"] in self.selected_levels
#         module_key = self.get_module_key(entry["module"])
#         module_ok = not self.selected_modules or module_key in self.selected_modules
#         return level_ok and module_ok

#     def init_text_tags(self):
#         """初始化日志文本的颜色标签"""
#         for tag, color in self.color_config.items():
#             self.log_text.tag_config(tag, foreground=color)
#         self.log_text.tag_config("default", foreground=self.color_config["default"])

#     def start_process(self):
#         """启动日志进程并开始读取输出"""
#         self.process = subprocess.Popen(
#             ["nb", "run"],
#             stdout=subprocess.PIPE,
#             stderr=subprocess.STDOUT,
#             text=True,
#             bufsize=1,
#             encoding="utf-8",
#             errors="ignore",
#         )
#         self.start_btn.configure(state="disabled")
#         self.stop_btn.configure(state="normal")
#         threading.Thread(target=self.read_output, daemon=True).start()

#     def stop_process(self):
#         """停止日志进程并清理相关资源"""
#         if self.process:
#             try:
#                 if hasattr(self.process, "pid"):
#                     if os.name == "nt":
#                         subprocess.run(
#                             ["taskkill", "/F", "/T", "/PID", str(self.process.pid)], check=True, capture_output=True
#                         )
#                     else:
#                         os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
#             except (subprocess.CalledProcessError, ProcessLookupError, OSError) as e:
#                 print(f"终止进程失败: {e}")
#             finally:
#                 self.process = None
#         self.log_queue.queue.clear()
#         self.start_btn.configure(state="normal")
#         self.stop_btn.configure(state="disabled")
#         self.refresh_logs()

#     def read_output(self):
#         """读取日志进程的输出并放入队列"""
#         try:
#             while self.process and self.process.poll() is None and self.is_running:
#                 line = self.process.stdout.readline()
#                 if line:
#                     self.log_queue.put(line)
#                 else:
#                     break  # 避免空循环
#             self.process.stdout.close()  # 确保关闭文件描述符
#         except ValueError:  # 处理可能的I/O操作异常
#             pass

#     def process_log_queue(self):
#         """处理日志队列中的日志条目"""
#         while not self.log_queue.empty():
#             line = self.log_queue.get()
#             self.process_log_line(line)

#         # 仅在GUI仍在运行时继续处理队列
#         if self.is_running:
#             self.after(100, self.process_log_queue)

#     def process_log_line(self, line):
#         """解析单行日志并更新日志数据和筛选器"""
#         match = re.match(
#             r"""^
#             (?:(?P<time>\d{2}:\d{2}(?::\d{2})?)\s*\|\s*)?
#             (?P<level>\w+)\s*\|\s*
#             (?P<module>.*?)
#             \s*[-|]\s*
#             (?P<message>.*)
#             $""",
#             line.strip(),
#             re.VERBOSE,
#         )

#         if match:
#             groups = match.groupdict()
#             time = groups.get("time", "")
#             level = groups.get("level", "OTHER")
#             module = groups.get("module", "UNKNOWN").strip()
#             message = groups.get("message", "").strip()
#             raw_line = line
#         else:
#             time, level, module, message = "", "OTHER", "UNKNOWN", line
#             raw_line = line

#         self.update_filters(level, module)
#         log_entry = {"raw": raw_line, "time": time, "level": level, "module": module, "message": message}
#         self.log_data.append(log_entry)

#         if self.check_filter(log_entry):
#             self.display_log(log_entry)

#     def get_module_key(self, module_name):
#         """获取模块名称的标准化键"""
#         cleaned = module_name.strip()
#         return re.sub(r":\d+$", "", cleaned)

#     def display_log(self, entry):
#         """在日志文本框中显示日志条目"""
#         parts = []
#         tags = []

#         if self.column_visibility["show_time"] and entry["time"]:
#             parts.append(f"{entry['time']} ")
#             tags.append("time")

#         if self.column_visibility["show_level"]:
#             level_tag = entry["level"] if entry["level"] in self.color_config else "default"
#             parts.append(f"{entry['level']:<8} ")
#             tags.append(level_tag)

#         if self.column_visibility["show_module"]:
#             parts.append(f"{entry['module']} ")
#             tags.append("module")

#         parts.append(f"- {entry['message']}\n")
#         tags.append("default")

#         self.log_text.configure(state="normal")
#         for part, tag in zip(parts, tags):
#             self.log_text.insert("end", part, tag)
#         self.log_text.see("end")
#         self.log_text.configure(state="disabled")

#     def refresh_logs(self):
#         """刷新日志显示，根据筛选条件重新显示日志"""
#         self.column_visibility = {
#             "show_time": self.time_check.get(),
#             "show_level": self.level_check.get(),
#             "show_module": self.module_check.get(),
#         }

#         self.log_text.configure(state="normal")
#         self.log_text.delete("1.0", "end")

#         filtered_logs = [entry for entry in self.log_data if self.check_filter(entry)]

#         for entry in filtered_logs:
#             parts = []
#             tags = []

#             if self.column_visibility["show_time"] and entry["time"]:
#                 parts.append(f"{entry['time']} ")
#                 tags.append("time")

#             if self.column_visibility["show_level"]:
#                 level_tag = entry["level"] if entry["level"] in self.color_config else "default"
#                 parts.append(f"{entry['level']:<8} ")
#                 tags.append(level_tag)

#             if self.column_visibility["show_module"]:
#                 parts.append(f"{entry['module']} ")
#                 tags.append("module")

#             parts.append(f"- {entry['message']}\n")
#             tags.append("default")

#             for part, tag in zip(parts, tags):
#                 self.log_text.insert("end", part, tag)

#         self.log_text.see("end")
#         self.log_text.configure(state="disabled")

#     def clear_logs(self):
#         """清空日志文本框中的内容"""
#         self.log_text.configure(state="normal")
#         self.log_text.delete("1.0", "end")
#         self.log_text.configure(state="disabled")

#     def _on_closing(self):
#         """处理窗口关闭事件，安全清理资源"""
#         # 标记GUI已关闭
#         self.is_running = False

#         # 停止日志进程
#         self.stop_process()

#         # 安全清理tkinter变量
#         for attr_name in list(self.__dict__.keys()):
#             if isinstance(getattr(self, attr_name), (ctk.Variable, ctk.StringVar, ctk.IntVar, ctk.DoubleVar, ctk.BooleanVar)):
#                 try:
#                     var = getattr(self, attr_name)
#                     var.set(None)
#                 except Exception:
#                     pass
#                 setattr(self, attr_name, None)

#         self.quit()
#         sys.exit(0)


# if __name__ == "__main__":
#     # 启动日志查看器应用
#     app = LogViewerApp()
#     app.mainloop()
