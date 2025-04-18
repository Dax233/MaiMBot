新写一个类，叫做pfchating
这个类初始化时会输入一个chat_stream或者stream_id
这个类会包含对应的sub_hearflow和一个chat_stream

pfchating有以下几个组成部分：
规划器：决定是否要进行回复（根据sub_heartflow中的observe内容），可以选择不回复，回复文字或者回复表情包，你可以使用llm的工具调用来实现
回复器：可以根据信息产生回复，这部分代码将大部分与trigger_reply_generation(stream_id, observed_messages)一模一样
（回复器可能同时运行多个(0-3个)，这些回复器会根据不同时刻的规划器产生不同回复
检查器：由于生成回复需要时间，检查器会检查在有了新的消息内容之后，回复是否还适合，如果合适就转给发送器
如果一条消息被发送了，其他回复在检查时也要增加这条消息的信息，防止重复发送内容相近的回复
发送器，将回复发送到聊天，这部分主体不需要再pfcchating中实现，只需要使用原有的self._send_response_messages(anchor_message, response_set, thinking_id)


当_process_triggered_reply(self, stream_id: str, observed_messages: List[dict]):触发时，并不会单独进行一次回复


问题：
1.每个pfchating是否对应一个caht_stream，是否是唯一的？(fix)
2.observe_text传入进来是纯str，是不是应该传进来message构成的list?(fix)
3.检查失败的回复应该怎么处理？(先抛弃)
4.如何比较相似度？
5.planner怎么写？（好像可以先不加入这部分）

BUG:
1.第一条激活消息没有被读取，进入pfc聊天委托时应该读取一下之前的上文
2.复读，可能是planner还未校准好
3.planner还未个性化，需要加入bot个性信息，且获取的聊天内容有问题
4.心流好像过短，而且有时候没有等待更新
5.表情包有可能会发两次