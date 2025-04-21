import traceback
from typing import Optional, Dict
import asyncio
import threading  # 导入 threading
from ...moods.moods import MoodManager
from ...chat.emoji_manager import emoji_manager
from .heartFC_generator import ResponseGenerator
from .messagesender import MessageManager
from src.heart_flow.heartflow import heartflow
from src.common.logger import get_module_logger, CHAT_STYLE_CONFIG, LogConfig
from src.plugins.person_info.relationship_manager import relationship_manager
from src.do_tool.tool_use import ToolUser
from .interest import InterestManager
from src.plugins.chat.chat_stream import chat_manager
from .pf_chatting import PFChatting


# 定义日志配置
chat_config = LogConfig(
    console_format=CHAT_STYLE_CONFIG["console_format"],
    file_format=CHAT_STYLE_CONFIG["file_format"],
)

logger = get_module_logger("HeartFCController", config=chat_config)

# 检测群聊兴趣的间隔时间
INTEREST_MONITOR_INTERVAL_SECONDS = 1


# 合并后的版本：使用 __new__ + threading.Lock 实现线程安全单例，类名为 HeartFCController
class HeartFCController:
    _instance = None
    _lock = threading.Lock()  # 使用 threading.Lock 保证 __new__ 线程安全
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                # Double-checked locking
                if cls._instance is None:
                    logger.debug("创建 HeartFCController 单例实例...")
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # 使用 _initialized 标志确保 __init__ 只执行一次
        if self._initialized:
            return
        # 虽然 __new__ 保证了只有一个实例，但为了防止意外重入或多线程下的初始化竞争，
        # 再次使用类锁保护初始化过程是更严谨的做法。
        # 如果确定 __init__ 逻辑本身是幂等的或非关键的，可以省略这里的锁。
        # 但为了保持原始逻辑的意图（防止重复初始化），这里保留检查。
        with self.__class__._lock:  # 确保初始化逻辑线程安全
            if self._initialized:  # 再次检查，防止锁等待期间其他线程已完成初始化
                return

            logger.info("正在初始化 HeartFCController 单例...")
            self.gpt = ResponseGenerator()
            self.mood_manager = MoodManager.get_instance()
            # 注意：mood_manager 的 start_mood_update 可能需要在应用主循环启动后调用，
            # 或者确保其内部实现是安全的。这里保持原状。
            self.mood_manager.start_mood_update()
            self.tool_user = ToolUser()
            # 注意：InterestManager() 可能是另一个单例或需要特定初始化。
            # 假设 InterestManager() 返回的是正确配置的实例。
            self.interest_manager = InterestManager()
            self._interest_monitor_task: Optional[asyncio.Task] = None
            self.pf_chatting_instances: Dict[str, PFChatting] = {}
            # _pf_chatting_lock 用于保护 pf_chatting_instances 的异步操作
            self._pf_chatting_lock = asyncio.Lock()  # 这个是 asyncio.Lock，用于异步上下文
            self.emoji_manager = emoji_manager  # 假设是全局或已初始化的实例
            self.relationship_manager = relationship_manager  # 假设是全局或已初始化的实例
            # MessageManager 可能是类本身或单例实例，根据其设计确定
            self.MessageManager = MessageManager
            self._initialized = True
            logger.info("HeartFCController 单例初始化完成。")

    @classmethod
    def get_instance(cls):
        """获取 HeartFCController 的单例实例。"""
        # 如果实例尚未创建，调用构造函数（这将触发 __new__ 和 __init__）
        if cls._instance is None:
            # 在首次调用 get_instance 时创建实例。
            # __new__ 中的锁会确保线程安全。
            cls()
            # 添加日志记录，说明实例是在 get_instance 调用时创建的
            logger.info("HeartFCController 实例在首次 get_instance 时创建。")
        elif not cls._initialized:
            # 实例已创建但可能未初始化完成（理论上不太可能发生，除非 __init__ 异常）
            logger.warning("HeartFCController 实例存在但尚未完成初始化。")
        return cls._instance

    # --- 新增：检查 PFChatting 状态的方法 --- #
    def is_pf_chatting_active(self, stream_id: str) -> bool:
        """检查指定 stream_id 的 PFChatting 循环是否处于活动状态。"""
        # 注意：这里直接访问字典，不加锁，因为读取通常是安全的，
        # 并且 PFChatting 实例的 _loop_active 状态由其自身的异步循环管理。
        # 如果需要更强的保证，可以在访问 pf_instance 前获取 _pf_chatting_lock
        pf_instance = self.pf_chatting_instances.get(stream_id)
        if pf_instance and pf_instance._loop_active:  # 直接检查 PFChatting 实例的 _loop_active 属性
            return True
        return False

    # --- 结束新增 --- #

    async def start(self):
        """启动异步任务,如回复启动器"""
        logger.debug("HeartFCController 正在启动异步任务...")
        self._initialize_monitor_task()
        logger.info("HeartFCController 异步任务启动完成")

    def _initialize_monitor_task(self):
        """启动后台兴趣监控任务，可以检查兴趣是否足以开启心流对话"""
        if self._interest_monitor_task is None or self._interest_monitor_task.done():
            try:
                loop = asyncio.get_running_loop()
                self._interest_monitor_task = loop.create_task(self._interest_monitor_loop())
            except RuntimeError:
                logger.error("创建兴趣监控任务失败：没有运行中的事件循环。")
                raise
        else:
            logger.warning("跳过兴趣监控任务创建：任务已存在或正在运行。")

    # --- Added PFChatting Instance Manager ---
    async def _get_or_create_pf_chatting(self, stream_id: str) -> Optional[PFChatting]:
        """获取现有PFChatting实例或创建新实例。"""
        async with self._pf_chatting_lock:
            if stream_id not in self.pf_chatting_instances:
                logger.info(f"为流 {stream_id} 创建新的PFChatting实例")
                # 传递 self (HeartFCController 实例) 进行依赖注入
                instance = PFChatting(stream_id, self)
                # 执行异步初始化
                if not await instance._initialize():
                    logger.error(f"为流 {stream_id} 初始化PFChatting失败")
                    return None
                self.pf_chatting_instances[stream_id] = instance
            return self.pf_chatting_instances[stream_id]

    # --- End Added PFChatting Instance Manager ---

    async def _interest_monitor_loop(self):
        """后台任务，定期检查兴趣度变化并触发回复"""
        logger.info("兴趣监控循环开始...")
        while True:
            await asyncio.sleep(INTEREST_MONITOR_INTERVAL_SECONDS)
            try:
                # 从心流中获取活跃流
                active_stream_ids = list(heartflow.get_all_subheartflows_streams_ids())
                for stream_id in active_stream_ids:
                    stream_name = chat_manager.get_stream_name(stream_id) or stream_id  # 获取流名称
                    sub_hf = heartflow.get_subheartflow(stream_id)
                    if not sub_hf:
                        logger.warning(f"监控循环: 无法获取活跃流 {stream_name} 的 sub_hf")
                        continue

                    should_trigger = False
                    try:
                        interest_chatting = self.interest_manager.get_interest_chatting(stream_id)
                        if interest_chatting:
                            should_trigger = interest_chatting.should_evaluate_reply()
                        else:
                            logger.trace(
                                f"[{stream_name}] 没有找到对应的 InterestChatting 实例，跳过基于兴趣的触发检查。"
                            )
                    except Exception as e:
                        logger.error(f"检查兴趣触发器时出错 流 {stream_name}: {e}")
                        logger.error(traceback.format_exc())

                    if should_trigger:
                        # 启动一次麦麦聊天
                        pf_instance = await self._get_or_create_pf_chatting(stream_id)
                        if pf_instance:
                            asyncio.create_task(pf_instance.add_time())
                        else:
                            logger.error(f"[{stream_name}] 无法获取或创建PFChatting实例。跳过触发。")

            except asyncio.CancelledError:
                logger.info("兴趣监控循环已取消。")
                break
            except Exception as e:
                logger.error(f"兴趣监控循环错误: {e}")
                logger.error(traceback.format_exc())
                await asyncio.sleep(5)  # 发生错误时等待
