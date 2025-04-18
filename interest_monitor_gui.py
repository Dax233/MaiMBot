import tkinter as tk
from tkinter import ttk
import time
import os
from datetime import datetime, timedelta
import random
from collections import deque
import json  # 引入 json

# --- 引入 Matplotlib ---
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.dates as mdates  # 用于处理日期格式
import matplotlib  # 导入 matplotlib

# --- 配置 ---
LOG_FILE_PATH = os.path.join("logs", "interest", "interest_history.log")  # 指向历史日志文件
REFRESH_INTERVAL_MS = 200  # 刷新间隔 (毫秒) - 可以适当调长，因为读取文件可能耗时
WINDOW_TITLE = "Interest Monitor (Live History)"
MAX_HISTORY_POINTS = 1000  # 图表上显示的最大历史点数 (可以增加)
MAX_STREAMS_TO_DISPLAY = 15  # 最多显示多少个聊天流的折线图 (可以增加)

# *** 添加 Matplotlib 中文字体配置 ***
# 尝试使用 'SimHei' 或 'Microsoft YaHei'，如果找不到，matplotlib 会回退到默认字体
# 确保你的系统上安装了这些字体
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False  # 解决负号'-'显示为方块的问题


class InterestMonitorApp:
    def __init__(self, root):
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.root.geometry("1800x800")  # 调整窗口大小以适应图表

        # --- 数据存储 ---
        # 使用 deque 来存储有限的历史数据点
        # key: stream_id, value: deque([(timestamp, interest_level), ...])
        self.stream_history = {}
        # key: stream_id, value: deque([(timestamp, reply_probability), ...]) # <--- 新增：存储概率历史
        self.probability_history = {}
        self.stream_colors = {}  # 为每个 stream 分配颜色
        self.stream_display_names = {}  # *** New: Store display names (group_name) ***
        self.selected_stream_id = tk.StringVar()  # 用于 Combobox 绑定

        # --- UI 元素 ---
        # 创建 Notebook (选项卡控件)
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(pady=10, padx=10, fill=tk.BOTH, expand=1)

        # --- 第一个选项卡：所有流 ---
        self.frame_all = ttk.Frame(self.notebook, padding="5 5 5 5")
        self.notebook.add(self.frame_all, text="所有聊天流")

        # 状态标签
        self.status_label = tk.Label(root, text="Initializing...", anchor="w", fg="grey")
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=2)

        # Matplotlib 图表设置 (用于第一个选项卡)
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        # 配置在 update_plot 中进行，避免重复

        # 创建 Tkinter 画布嵌入 Matplotlib 图表 (用于第一个选项卡)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame_all)  # <--- 放入 frame_all
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # --- 第二个选项卡：单个流 ---
        self.frame_single = ttk.Frame(self.notebook, padding="5 5 5 5")
        self.notebook.add(self.frame_single, text="单个聊天流详情")

        # 单个流选项卡的上部控制区域
        self.control_frame_single = ttk.Frame(self.frame_single)
        self.control_frame_single.pack(side=tk.TOP, fill=tk.X, pady=5)

        ttk.Label(self.control_frame_single, text="选择聊天流:").pack(side=tk.LEFT, padx=(0, 5))
        self.stream_selector = ttk.Combobox(
            self.control_frame_single, textvariable=self.selected_stream_id, state="readonly", width=50
        )
        self.stream_selector.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.stream_selector.bind("<<ComboboxSelected>>", self.on_stream_selected)

        # Matplotlib 图表设置 (用于第二个选项卡)
        self.fig_single = Figure(figsize=(5, 4), dpi=100)
        # 修改：创建两个子图，一个显示兴趣度，一个显示概率
        self.ax_single_interest = self.fig_single.add_subplot(211)  # 2行1列的第1个
        self.ax_single_probability = self.fig_single.add_subplot(
            212, sharex=self.ax_single_interest
        )  # 2行1列的第2个，共享X轴

        # 创建 Tkinter 画布嵌入 Matplotlib 图表 (用于第二个选项卡)
        self.canvas_single = FigureCanvasTkAgg(self.fig_single, master=self.frame_single)  # <--- 放入 frame_single
        self.canvas_widget_single = self.canvas_single.get_tk_widget()
        self.canvas_widget_single.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # --- 初始化和启动刷新 ---
        self.update_display()  # 首次加载并开始刷新循环

    def on_stream_selected(self, event=None):
        """当 Combobox 选择改变时调用，更新单个流的图表"""
        self.update_single_stream_plot()

    def get_random_color(self):
        """生成随机颜色用于区分线条"""
        return "#{:06x}".format(random.randint(0, 0xFFFFFF))

    def load_and_update_history(self):
        """从 history log 文件加载数据并更新历史记录"""
        if not os.path.exists(LOG_FILE_PATH):
            self.set_status(f"Error: Log file not found at {LOG_FILE_PATH}", "red")
            # 如果文件不存在，不清空现有数据，以便显示最后一次成功读取的状态
            return

        # *** Reset display names each time we reload ***
        new_stream_history = {}
        new_stream_display_names = {}
        new_probability_history = {}  # <--- 重置概率历史
        read_count = 0
        error_count = 0
        # *** Calculate the timestamp threshold for the last 30 minutes ***
        current_time = time.time()
        time_threshold = current_time - (15 * 60)  # 30 minutes in seconds

        try:
            with open(LOG_FILE_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    read_count += 1
                    try:
                        log_entry = json.loads(line.strip())
                        timestamp = log_entry.get("timestamp")

                        # *** Add time filtering ***
                        if timestamp is None or float(timestamp) < time_threshold:
                            continue  # Skip old or invalid entries

                        stream_id = log_entry.get("stream_id")
                        interest_level = log_entry.get("interest_level")
                        group_name = log_entry.get(
                            "group_name", stream_id
                        )  # *** Get group_name, fallback to stream_id ***
                        reply_probability = log_entry.get("reply_probability")  # <--- 获取概率值

                        # *** Check other required fields AFTER time filtering ***
                        if stream_id is None or interest_level is None:
                            error_count += 1
                            continue  # 跳过无效行

                        # 如果是第一次读到这个 stream_id，则创建 deque
                        if stream_id not in new_stream_history:
                            new_stream_history[stream_id] = deque(maxlen=MAX_HISTORY_POINTS)
                            new_probability_history[stream_id] = deque(maxlen=MAX_HISTORY_POINTS)  # <--- 创建概率 deque
                            # 检查是否已有颜色，没有则分配
                            if stream_id not in self.stream_colors:
                                self.stream_colors[stream_id] = self.get_random_color()

                        # *** Store the latest display name found for this stream_id ***
                        new_stream_display_names[stream_id] = group_name

                        # 添加数据点
                        new_stream_history[stream_id].append((float(timestamp), float(interest_level)))
                        # 添加概率数据点 (如果存在)
                        if reply_probability is not None:
                            try:
                                new_probability_history[stream_id].append((float(timestamp), float(reply_probability)))
                            except (TypeError, ValueError):
                                # 如果概率值无效，可以跳过或记录一个默认值，这里跳过
                                pass

                    except json.JSONDecodeError:
                        error_count += 1
                        # logger.warning(f"Skipping invalid JSON line: {line.strip()}")
                        continue  # 跳过无法解析的行
                    except (TypeError, ValueError):
                        error_count += 1
                        # logger.warning(f"Skipping line due to data type error ({e}): {line.strip()}")
                        continue  # 跳过数据类型错误的行

            # 读取完成后，用新数据替换旧数据
            self.stream_history = new_stream_history
            self.stream_display_names = new_stream_display_names  # *** Update display names ***
            self.probability_history = new_probability_history  # <--- 更新概率历史
            status_msg = f"Data loaded at {datetime.now().strftime('%H:%M:%S')}. Lines read: {read_count}."
            if error_count > 0:
                status_msg += f" Skipped {error_count} invalid lines."
                self.set_status(status_msg, "orange")
            else:
                self.set_status(status_msg, "green")

        except IOError as e:
            self.set_status(f"Error reading file {LOG_FILE_PATH}: {e}", "red")
        except Exception as e:
            self.set_status(f"An unexpected error occurred during loading: {e}", "red")

        # --- 更新 Combobox ---
        self.update_stream_selector()

    def update_stream_selector(self):
        """更新单个流选项卡中的 Combobox 列表"""
        # 创建 (display_name, stream_id) 对的列表，按 display_name 排序
        available_streams = sorted(
            [
                (name, sid)
                for sid, name in self.stream_display_names.items()
                if sid in self.stream_history and self.stream_history[sid]
            ],
            key=lambda item: item[0],  # 按显示名称排序
        )

        # 更新 Combobox 的值 (仅显示 display_name)
        self.stream_selector["values"] = [name for name, sid in available_streams]

        # 检查当前选中的 stream_id 是否仍然有效
        current_selection_name = self.selected_stream_id.get()
        current_selection_valid = any(name == current_selection_name for name, sid in available_streams)

        if not current_selection_valid and available_streams:
            # 如果当前选择无效，并且有可选流，则默认选中第一个
            self.selected_stream_id.set(available_streams[0][0])
            # 手动触发一次更新，因为 set 不会触发 <<ComboboxSelected>>
            self.update_single_stream_plot()
        elif not available_streams:
            # 如果没有可选流，清空选择
            self.selected_stream_id.set("")
            self.update_single_stream_plot()  # 清空图表

    def update_all_streams_plot(self):
        """更新第一个选项卡的 Matplotlib 图表 (显示所有流)"""
        self.ax.clear()  # 清除旧图
        # *** 设置中文标题和标签 ***
        self.ax.set_title("兴趣度随时间变化图 (所有活跃流)")
        self.ax.set_xlabel("时间")
        self.ax.set_ylabel("兴趣度")
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        self.ax.grid(True)
        self.ax.set_ylim(0, 10)  # 固定 Y 轴范围 0-10

        # 只绘制最新的 N 个 stream (按最后记录的兴趣度排序)
        # 注意：现在是基于文件读取的快照排序，可能不是实时最新
        active_streams = sorted(
            self.stream_history.items(),
            key=lambda item: item[1][-1][1] if item[1] else 0,  # 按最后兴趣度排序
            reverse=True,
        )[:MAX_STREAMS_TO_DISPLAY]

        all_times = []  # 用于确定 X 轴范围

        for stream_id, history in active_streams:
            if not history:
                continue

            timestamps, interests = zip(*history)
            # 将 time.time() 时间戳转换为 matplotlib 可识别的日期格式
            try:
                mpl_dates = [datetime.fromtimestamp(ts) for ts in timestamps]
                all_times.extend(mpl_dates)  # 收集所有时间点

                # *** Use display name for label ***
                display_label = self.stream_display_names.get(stream_id, stream_id)

                self.ax.plot(
                    mpl_dates,
                    interests,
                    label=display_label,  # *** Use display_label ***
                    color=self.stream_colors.get(stream_id, "grey"),
                    marker=".",
                    markersize=3,
                    linestyle="-",
                    linewidth=1,
                )
            except ValueError as e:
                print(f"Skipping plot for {stream_id} due to invalid timestamp: {e}")
                continue

        if all_times:
            # 根据数据动态调整 X 轴范围，留一点边距
            min_time = min(all_times)
            max_time = max(all_times)
            # delta = max_time - min_time
            # self.ax.set_xlim(min_time - delta * 0.05, max_time + delta * 0.05)
            self.ax.set_xlim(min_time, max_time)

            # 自动格式化X轴标签
            self.fig.autofmt_xdate()
        else:
            # 如果没有数据，设置一个默认的时间范围，例如最近一小时
            now = datetime.now()
            one_hour_ago = now - timedelta(hours=1)
            self.ax.set_xlim(one_hour_ago, now)

        # 添加图例
        if active_streams:
            # 调整图例位置和大小
            # 字体已通过全局 matplotlib.rcParams 设置
            self.ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0.0, fontsize="x-small")
            # 调整布局，确保图例不被裁剪
            self.fig.tight_layout(rect=[0, 0, 0.85, 1])  # 右侧留出空间给图例

        self.canvas.draw()  # 重绘画布

    def update_single_stream_plot(self):
        """更新第二个选项卡的 Matplotlib 图表 (显示单个选定的流)"""
        self.ax_single_interest.clear()
        self.ax_single_probability.clear()

        # 设置子图标题和标签
        self.ax_single_interest.set_title("兴趣度")
        self.ax_single_interest.set_ylabel("兴趣度")
        self.ax_single_interest.grid(True)
        self.ax_single_interest.set_ylim(0, 10)  # 固定 Y 轴范围 0-10

        self.ax_single_probability.set_title("回复评估概率")
        self.ax_single_probability.set_xlabel("时间")
        self.ax_single_probability.set_ylabel("概率")
        self.ax_single_probability.grid(True)
        self.ax_single_probability.set_ylim(0, 1.05)  # 固定 Y 轴范围 0-1
        self.ax_single_probability.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))

        selected_name = self.selected_stream_id.get()
        selected_sid = None

        # --- 新增：根据选中的名称找到 stream_id ---
        if selected_name:
            for sid, name in self.stream_display_names.items():
                if name == selected_name:
                    selected_sid = sid
                    break

        all_times = []  # 用于确定 X 轴范围

        # --- 新增：绘制兴趣度图 ---
        if selected_sid and selected_sid in self.stream_history and self.stream_history[selected_sid]:
            history = self.stream_history[selected_sid]
            timestamps, interests = zip(*history)
            try:
                mpl_dates = [datetime.fromtimestamp(ts) for ts in timestamps]
                all_times.extend(mpl_dates)
                self.ax_single_interest.plot(
                    mpl_dates,
                    interests,
                    color=self.stream_colors.get(selected_sid, "blue"),
                    marker=".",
                    markersize=3,
                    linestyle="-",
                    linewidth=1,
                )
            except ValueError as e:
                print(f"Skipping interest plot for {selected_sid} due to invalid timestamp: {e}")

        # --- 新增：绘制概率图 ---
        if selected_sid and selected_sid in self.probability_history and self.probability_history[selected_sid]:
            prob_history = self.probability_history[selected_sid]
            prob_timestamps, probabilities = zip(*prob_history)
            try:
                prob_mpl_dates = [datetime.fromtimestamp(ts) for ts in prob_timestamps]
                # 注意：概率图的时间点可能与兴趣度不同，也需要加入 all_times
                all_times.extend(prob_mpl_dates)
                self.ax_single_probability.plot(
                    prob_mpl_dates,
                    probabilities,
                    color=self.stream_colors.get(selected_sid, "green"),  # 可以用不同颜色
                    marker=".",
                    markersize=3,
                    linestyle="-",
                    linewidth=1,
                )
            except ValueError as e:
                print(f"Skipping probability plot for {selected_sid} due to invalid timestamp: {e}")

        # --- 新增：调整 X 轴范围和格式 ---
        if all_times:
            min_time = min(all_times)
            max_time = max(all_times)
            # 设置共享的 X 轴范围
            self.ax_single_interest.set_xlim(min_time, max_time)
            # self.ax_single_probability.set_xlim(min_time, max_time) # sharex 会自动同步
            # 自动格式化X轴标签 (应用到共享轴的最后一个子图上通常即可)
            self.fig_single.autofmt_xdate()
        else:
            # 如果没有数据，设置一个默认的时间范围
            now = datetime.now()
            one_hour_ago = now - timedelta(hours=1)
            self.ax_single_interest.set_xlim(one_hour_ago, now)
            # self.ax_single_probability.set_xlim(one_hour_ago, now) # sharex 会自动同步

        # --- 新增：重新绘制画布 ---
        self.canvas_single.draw()

    def update_display(self):
        """主更新循环"""
        try:
            self.load_and_update_history()  # 从文件加载数据并更新内部状态
            # *** 修改：分别调用两个图表的更新方法 ***
            self.update_all_streams_plot()  # 更新所有流的图表
            self.update_single_stream_plot()  # 更新单个流的图表
        except Exception as e:
            # 提供更详细的错误信息
            import traceback

            error_msg = f"Error during update: {e}\n{traceback.format_exc()}"
            self.set_status(error_msg, "red")
            print(error_msg)  # 打印详细错误到控制台

        # 安排下一次刷新
        self.root.after(REFRESH_INTERVAL_MS, self.update_display)

    def set_status(self, message: str, color: str = "grey"):
        """更新状态栏标签"""
        # 限制状态栏消息长度
        max_len = 150
        display_message = (message[:max_len] + "...") if len(message) > max_len else message
        self.status_label.config(text=display_message, fg=color)


if __name__ == "__main__":
    # 导入 timedelta 用于默认时间范围
    from datetime import timedelta

    root = tk.Tk()
    app = InterestMonitorApp(root)
    root.mainloop()
