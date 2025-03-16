import asyncio
from typing import Dict
from ..chat.chat_stream import ChatStream

class WillingManager:
    def __init__(self):
        self.chat_reply_willing: Dict[str, float] = {}  # 存储每个聊天流的回复意愿
        self._decay_task = None
        self._started = False
        
    async def _decay_reply_willing(self):
        """定期衰减回复意愿"""
        while True:
            await asyncio.sleep(3)
            for chat_id in self.chat_reply_willing:
                # 每分钟衰减10%的回复意愿
                self.chat_reply_willing[chat_id] = max(0, self.chat_reply_willing[chat_id] * 0.6)
                
    def get_willing(self, chat_stream: ChatStream) -> float:
        """获取指定聊天流的回复意愿"""
        if chat_stream:
            return self.chat_reply_willing.get(chat_stream.stream_id, 0)
        return 0
    
    def set_willing(self, chat_id: str, willing: float):
        """设置指定聊天流的回复意愿"""
        self.chat_reply_willing[chat_id] = willing
        
    async def change_reply_willing_received(self, 
                                          chat_stream: ChatStream,
                                          topic: str = None,
                                          is_mentioned_bot: bool = False,
                                          config = None,
                                          is_emoji: bool = False,
                                          interested_rate: float = 0,
                                          sender_id: str = None) -> float:
        """改变指定聊天流的回复意愿并返回回复概率"""
        chat_id = chat_stream.stream_id
        current_willing = self.chat_reply_willing.get(chat_id, 0)
        
        if topic and current_willing < 1:
            current_willing += 0.2
        elif topic:
            current_willing += 0.05
            
        if is_mentioned_bot and current_willing < 1.0:
            current_willing += 0.9
        elif is_mentioned_bot:
            current_willing += 0.05
        
        if is_emoji:
            current_willing *= 0.2
        
        self.chat_reply_willing[chat_id] = min(current_willing, 3.0)
        
        reply_probability = (current_willing - 0.5) * 2

        # 检查群组权限（如果是群聊）
        if chat_stream.group_info and config:
            if chat_stream.group_info.group_id not in config.talk_allowed_groups:
                current_willing = 0
                reply_probability = 0
                
            if chat_stream.group_info.group_id in config.talk_frequency_down_groups:
                reply_probability = reply_probability / 3.5
                
            if is_mentioned_bot and sender_id == "1026294844":
                reply_probability = 1
            
        return reply_probability
    
    def change_reply_willing_sent(self, chat_stream: ChatStream):
        """发送消息后降低聊天流的回复意愿"""
        if chat_stream:
            chat_id = chat_stream.stream_id
            current_willing = self.chat_reply_willing.get(chat_id, 0)
            self.chat_reply_willing[chat_id] = max(0, current_willing - 1.8)
    
    def change_reply_willing_not_sent(self, chat_stream: ChatStream):
        """未发送消息后降低聊天流的回复意愿"""
        if chat_stream:
            chat_id = chat_stream.stream_id
            current_willing = self.chat_reply_willing.get(chat_id, 0)
            self.chat_reply_willing[chat_id] = max(0, current_willing - 0)
        
    def change_reply_willing_after_sent(self, chat_stream: ChatStream):
        """发送消息后提高聊天流的回复意愿"""
        if chat_stream:
            chat_id = chat_stream.stream_id
            current_willing = self.chat_reply_willing.get(chat_id, 0)
            if current_willing < 1:
                self.chat_reply_willing[chat_id] = min(1, current_willing + 0.4)
        
    async def ensure_started(self):
        """确保衰减任务已启动"""
        if not self._started:
            if self._decay_task is None:
                self._decay_task = asyncio.create_task(self._decay_reply_willing())
            self._started = True

# 创建全局实例
willing_manager = WillingManager() 