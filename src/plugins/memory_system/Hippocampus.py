# -*- coding: utf-8 -*-
import datetime
import math
import random
import time
import re

import jieba
import networkx as nx

# from nonebot import get_driver
from ...common.database import db
# from ..chat.config import global_config
from ..chat.utils import (
    calculate_information_content,
    cosine_similarity,
    get_closest_chat_from_db,
    text_to_vector,
)
from ..models.utils_model import LLM_request
from src.common.logger import get_module_logger, LogConfig, MEMORY_STYLE_CONFIG
from src.plugins.memory_system.sample_distribution import MemoryBuildScheduler #分布生成器
from .config import MemoryConfig

# 定义日志配置
memory_config = LogConfig(
    # 使用海马体专用样式
    console_format=MEMORY_STYLE_CONFIG["console_format"],
    file_format=MEMORY_STYLE_CONFIG["file_format"],
)


logger = get_module_logger("memory_system", config=memory_config)


class Memory_graph:
    def __init__(self):
        self.G = nx.Graph()  # 使用 networkx 的图结构

    def connect_dot(self, concept1, concept2):
        # 避免自连接
        if concept1 == concept2:
            return

        current_time = datetime.datetime.now().timestamp()

        # 如果边已存在,增加 strength
        if self.G.has_edge(concept1, concept2):
            self.G[concept1][concept2]["strength"] = self.G[concept1][concept2].get("strength", 1) + 1
            # 更新最后修改时间
            self.G[concept1][concept2]["last_modified"] = current_time
        else:
            # 如果是新边,初始化 strength 为 1
            self.G.add_edge(
                concept1,
                concept2,
                strength=1,
                created_time=current_time,  # 添加创建时间
                last_modified=current_time,
            )  # 添加最后修改时间

    def add_dot(self, concept, memory):
        current_time = datetime.datetime.now().timestamp()

        if concept in self.G:
            if "memory_items" in self.G.nodes[concept]:
                if not isinstance(self.G.nodes[concept]["memory_items"], list):
                    self.G.nodes[concept]["memory_items"] = [self.G.nodes[concept]["memory_items"]]
                self.G.nodes[concept]["memory_items"].append(memory)
                # 更新最后修改时间
                self.G.nodes[concept]["last_modified"] = current_time
            else:
                self.G.nodes[concept]["memory_items"] = [memory]
                # 如果节点存在但没有memory_items,说明是第一次添加memory,设置created_time
                if "created_time" not in self.G.nodes[concept]:
                    self.G.nodes[concept]["created_time"] = current_time
                self.G.nodes[concept]["last_modified"] = current_time
        else:
            # 如果是新节点,创建新的记忆列表
            self.G.add_node(
                concept,
                memory_items=[memory],
                created_time=current_time,  # 添加创建时间
                last_modified=current_time,
            )  # 添加最后修改时间

    def get_dot(self, concept):
        # 检查节点是否存在于图中
        if concept in self.G:
            # 从图中获取节点数据
            node_data = self.G.nodes[concept]
            return concept, node_data
        return None

    def get_related_item(self, topic, depth=1):
        if topic not in self.G:
            return [], []

        first_layer_items = []
        second_layer_items = []

        # 获取相邻节点
        neighbors = list(self.G.neighbors(topic))

        # 获取当前节点的记忆项
        node_data = self.get_dot(topic)
        if node_data:
            concept, data = node_data
            if "memory_items" in data:
                memory_items = data["memory_items"]
                if isinstance(memory_items, list):
                    first_layer_items.extend(memory_items)
                else:
                    first_layer_items.append(memory_items)

        # 只在depth=2时获取第二层记忆
        if depth >= 2:
            # 获取相邻节点的记忆项
            for neighbor in neighbors:
                node_data = self.get_dot(neighbor)
                if node_data:
                    concept, data = node_data
                    if "memory_items" in data:
                        memory_items = data["memory_items"]
                        if isinstance(memory_items, list):
                            second_layer_items.extend(memory_items)
                        else:
                            second_layer_items.append(memory_items)

        return first_layer_items, second_layer_items

    @property
    def dots(self):
        # 返回所有节点对应的 Memory_dot 对象
        return [self.get_dot(node) for node in self.G.nodes()]

    def forget_topic(self, topic):
        """随机删除指定话题中的一条记忆，如果话题没有记忆则移除该话题节点"""
        if topic not in self.G:
            return None

        # 获取话题节点数据
        node_data = self.G.nodes[topic]

        # 如果节点存在memory_items
        if "memory_items" in node_data:
            memory_items = node_data["memory_items"]

            # 确保memory_items是列表
            if not isinstance(memory_items, list):
                memory_items = [memory_items] if memory_items else []

            # 如果有记忆项可以删除
            if memory_items:
                # 随机选择一个记忆项删除
                removed_item = random.choice(memory_items)
                memory_items.remove(removed_item)

                # 更新节点的记忆项
                if memory_items:
                    self.G.nodes[topic]["memory_items"] = memory_items
                else:
                    # 如果没有记忆项了，删除整个节点
                    self.G.remove_node(topic)

                return removed_item

        return None

#负责海马体与其他部分的交互
class EntorhinalCortex:
    def __init__(self, hippocampus):
        self.hippocampus = hippocampus
        self.memory_graph = hippocampus.memory_graph
        self.config = hippocampus.config

    def get_memory_sample(self):
        """从数据库获取记忆样本"""
        # 硬编码：每条消息最大记忆次数
        max_memorized_time_per_msg = 3

        # 创建双峰分布的记忆调度器
        scheduler = MemoryBuildScheduler(
            n_hours1=self.config.memory_build_distribution[0],
            std_hours1=self.config.memory_build_distribution[1],
            weight1=self.config.memory_build_distribution[2],
            n_hours2=self.config.memory_build_distribution[3],
            std_hours2=self.config.memory_build_distribution[4],
            weight2=self.config.memory_build_distribution[5],
            total_samples=self.config.build_memory_sample_num
        )

        timestamps = scheduler.get_timestamp_array()
        logger.info(f"回忆往事: {[time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts)) for ts in timestamps]}")
        chat_samples = []
        for timestamp in timestamps:
            messages = self.random_get_msg_snippet(
                timestamp, 
                self.config.build_memory_sample_length, 
                max_memorized_time_per_msg
            )
            if messages:
                time_diff = (datetime.datetime.now().timestamp() - timestamp) / 3600
                logger.debug(f"成功抽取 {time_diff:.1f} 小时前的消息样本，共{len(messages)}条")
                chat_samples.append(messages)
            else:
                logger.debug(f"时间戳 {timestamp} 的消息样本抽取失败")

        return chat_samples

    def random_get_msg_snippet(self, target_timestamp: float, chat_size: int, max_memorized_time_per_msg: int) -> list:
        """从数据库中随机获取指定时间戳附近的消息片段"""
        try_count = 0
        while try_count < 3:
            messages = get_closest_chat_from_db(length=chat_size, timestamp=target_timestamp)
            if messages:
                for message in messages:
                    if message["memorized_times"] >= max_memorized_time_per_msg:
                        messages = None
                        break
                if messages:
                    for message in messages:
                        db.messages.update_one(
                            {"_id": message["_id"]}, {"$set": {"memorized_times": message["memorized_times"] + 1}}
                        )
                    return messages
            try_count += 1
        return None

    async def sync_memory_to_db(self):
        """将记忆图同步到数据库"""
        # 获取数据库中所有节点和内存中所有节点
        db_nodes = list(db.graph_data.nodes.find())
        memory_nodes = list(self.memory_graph.G.nodes(data=True))

        # 转换数据库节点为字典格式,方便查找
        db_nodes_dict = {node["concept"]: node for node in db_nodes}

        # 检查并更新节点
        for concept, data in memory_nodes:
            memory_items = data.get("memory_items", [])
            if not isinstance(memory_items, list):
                memory_items = [memory_items] if memory_items else []

            # 计算内存中节点的特征值
            memory_hash = self.hippocampus.calculate_node_hash(concept, memory_items)

            # 获取时间信息
            created_time = data.get("created_time", datetime.datetime.now().timestamp())
            last_modified = data.get("last_modified", datetime.datetime.now().timestamp())

            if concept not in db_nodes_dict:
                # 数据库中缺少的节点,添加
                node_data = {
                    "concept": concept,
                    "memory_items": memory_items,
                    "hash": memory_hash,
                    "created_time": created_time,
                    "last_modified": last_modified,
                }
                db.graph_data.nodes.insert_one(node_data)
            else:
                # 获取数据库中节点的特征值
                db_node = db_nodes_dict[concept]
                db_hash = db_node.get("hash", None)

                # 如果特征值不同,则更新节点
                if db_hash != memory_hash:
                    db.graph_data.nodes.update_one(
                        {"concept": concept},
                        {
                            "$set": {
                                "memory_items": memory_items,
                                "hash": memory_hash,
                                "created_time": created_time,
                                "last_modified": last_modified,
                            }
                        },
                    )

        # 处理边的信息
        db_edges = list(db.graph_data.edges.find())
        memory_edges = list(self.memory_graph.G.edges(data=True))

        # 创建边的哈希值字典
        db_edge_dict = {}
        for edge in db_edges:
            edge_hash = self.hippocampus.calculate_edge_hash(edge["source"], edge["target"])
            db_edge_dict[(edge["source"], edge["target"])] = {"hash": edge_hash, "strength": edge.get("strength", 1)}

        # 检查并更新边
        for source, target, data in memory_edges:
            edge_hash = self.hippocampus.calculate_edge_hash(source, target)
            edge_key = (source, target)
            strength = data.get("strength", 1)

            # 获取边的时间信息
            created_time = data.get("created_time", datetime.datetime.now().timestamp())
            last_modified = data.get("last_modified", datetime.datetime.now().timestamp())

            if edge_key not in db_edge_dict:
                # 添加新边
                edge_data = {
                    "source": source,
                    "target": target,
                    "strength": strength,
                    "hash": edge_hash,
                    "created_time": created_time,
                    "last_modified": last_modified,
                }
                db.graph_data.edges.insert_one(edge_data)
            else:
                # 检查边的特征值是否变化
                if db_edge_dict[edge_key]["hash"] != edge_hash:
                    db.graph_data.edges.update_one(
                        {"source": source, "target": target},
                        {
                            "$set": {
                                "hash": edge_hash,
                                "strength": strength,
                                "created_time": created_time,
                                "last_modified": last_modified,
                            }
                        },
                    )

    def sync_memory_from_db(self):
        """从数据库同步数据到内存中的图结构"""
        current_time = datetime.datetime.now().timestamp()
        need_update = False

        # 清空当前图
        self.memory_graph.G.clear()

        # 从数据库加载所有节点
        nodes = list(db.graph_data.nodes.find())
        for node in nodes:
            concept = node["concept"]
            memory_items = node.get("memory_items", [])
            if not isinstance(memory_items, list):
                memory_items = [memory_items] if memory_items else []

            # 检查时间字段是否存在
            if "created_time" not in node or "last_modified" not in node:
                need_update = True
                # 更新数据库中的节点
                update_data = {}
                if "created_time" not in node:
                    update_data["created_time"] = current_time
                if "last_modified" not in node:
                    update_data["last_modified"] = current_time

                db.graph_data.nodes.update_one({"concept": concept}, {"$set": update_data})
                logger.info(f"[时间更新] 节点 {concept} 添加缺失的时间字段")

            # 获取时间信息(如果不存在则使用当前时间)
            created_time = node.get("created_time", current_time)
            last_modified = node.get("last_modified", current_time)

            # 添加节点到图中
            self.memory_graph.G.add_node(
                concept, memory_items=memory_items, created_time=created_time, last_modified=last_modified
            )

        # 从数据库加载所有边
        edges = list(db.graph_data.edges.find())
        for edge in edges:
            source = edge["source"]
            target = edge["target"]
            strength = edge.get("strength", 1)

            # 检查时间字段是否存在
            if "created_time" not in edge or "last_modified" not in edge:
                need_update = True
                # 更新数据库中的边
                update_data = {}
                if "created_time" not in edge:
                    update_data["created_time"] = current_time
                if "last_modified" not in edge:
                    update_data["last_modified"] = current_time

                db.graph_data.edges.update_one({"source": source, "target": target}, {"$set": update_data})
                logger.info(f"[时间更新] 边 {source} - {target} 添加缺失的时间字段")

            # 获取时间信息(如果不存在则使用当前时间)
            created_time = edge.get("created_time", current_time)
            last_modified = edge.get("last_modified", current_time)

            # 只有当源节点和目标节点都存在时才添加边
            if source in self.memory_graph.G and target in self.memory_graph.G:
                self.memory_graph.G.add_edge(
                    source, target, strength=strength, created_time=created_time, last_modified=last_modified
                )

        if need_update:
            logger.success("[数据库] 已为缺失的时间字段进行补充")

#负责整合，遗忘，合并记忆
class ParahippocampalGyrus:
    def __init__(self, hippocampus):
        self.hippocampus = hippocampus
        self.memory_graph = hippocampus.memory_graph
        self.config = hippocampus.config

    async def memory_compress(self, messages: list, compress_rate=0.1):
        """压缩和总结消息内容，生成记忆主题和摘要。

        Args:
            messages (list): 消息列表，每个消息是一个字典，包含以下字段：
                - time: float, 消息的时间戳
                - detailed_plain_text: str, 消息的详细文本内容
            compress_rate (float, optional): 压缩率，用于控制生成的主题数量。默认为0.1。

        Returns:
            tuple: (compressed_memory, similar_topics_dict)
                - compressed_memory: set, 压缩后的记忆集合，每个元素是一个元组 (topic, summary)
                    - topic: str, 记忆主题
                    - summary: str, 主题的摘要描述
                - similar_topics_dict: dict, 相似主题字典，key为主题，value为相似主题列表
                    每个相似主题是一个元组 (similar_topic, similarity)
                    - similar_topic: str, 相似的主题
                    - similarity: float, 相似度分数（0-1之间）

        Process:
            1. 合并消息文本并生成时间信息
            2. 使用LLM提取关键主题
            3. 过滤掉包含禁用关键词的主题
            4. 为每个主题生成摘要
            5. 查找与现有记忆中的相似主题
        """
        if not messages:
            return set(), {}

        # 合并消息文本，同时保留时间信息
        input_text = ""
        time_info = ""
        # 计算最早和最晚时间
        earliest_time = min(msg["time"] for msg in messages)
        latest_time = max(msg["time"] for msg in messages)

        earliest_dt = datetime.datetime.fromtimestamp(earliest_time)
        latest_dt = datetime.datetime.fromtimestamp(latest_time)

        # 如果是同一年
        if earliest_dt.year == latest_dt.year:
            earliest_str = earliest_dt.strftime("%m-%d %H:%M:%S")
            latest_str = latest_dt.strftime("%m-%d %H:%M:%S")
            time_info += f"是在{earliest_dt.year}年，{earliest_str} 到 {latest_str} 的对话:\n"
        else:
            earliest_str = earliest_dt.strftime("%Y-%m-%d %H:%M:%S")
            latest_str = latest_dt.strftime("%Y-%m-%d %H:%M:%S")
            time_info += f"是从 {earliest_str} 到 {latest_str} 的对话:\n"

        for msg in messages:
            input_text += f"{msg['detailed_plain_text']}\n"

        logger.debug(input_text)

        topic_num = self.hippocampus.calculate_topic_num(input_text, compress_rate)
        topics_response = await self.hippocampus.llm_topic_judge.generate_response(self.hippocampus.find_topic_llm(input_text, topic_num))

        # 使用正则表达式提取<>中的内容
        topics = re.findall(r'<([^>]+)>', topics_response[0])
        
        # 如果没有找到<>包裹的内容，返回['none']
        if not topics:
            topics = ['none']
        else:
            # 处理提取出的话题
            topics = [
                topic.strip()
                for topic in ','.join(topics).replace("，", ",").replace("、", ",").replace(" ", ",").split(",")
                if topic.strip()
            ]

        # 过滤掉包含禁用关键词的topic
        filtered_topics = [
            topic for topic in topics 
            if not any(keyword in topic for keyword in self.config.memory_ban_words)
        ]

        logger.debug(f"过滤后话题: {filtered_topics}")

        # 创建所有话题的请求任务
        tasks = []
        for topic in filtered_topics:
            topic_what_prompt = self.hippocampus.topic_what(input_text, topic, time_info)
            task = self.hippocampus.llm_summary_by_topic.generate_response_async(topic_what_prompt)
            tasks.append((topic.strip(), task))

        # 等待所有任务完成
        compressed_memory = set()
        similar_topics_dict = {}
        
        for topic, task in tasks:
            response = await task
            if response:
                compressed_memory.add((topic, response[0]))
                
                existing_topics = list(self.memory_graph.G.nodes())
                similar_topics = []

                for existing_topic in existing_topics:
                    topic_words = set(jieba.cut(topic))
                    existing_words = set(jieba.cut(existing_topic))

                    all_words = topic_words | existing_words
                    v1 = [1 if word in topic_words else 0 for word in all_words]
                    v2 = [1 if word in existing_words else 0 for word in all_words]

                    similarity = cosine_similarity(v1, v2)

                    if similarity >= 0.7:
                        similar_topics.append((existing_topic, similarity))

                similar_topics.sort(key=lambda x: x[1], reverse=True)
                similar_topics = similar_topics[:3]
                similar_topics_dict[topic] = similar_topics

        return compressed_memory, similar_topics_dict

    async def operation_build_memory(self):
        logger.debug("------------------------------------开始构建记忆--------------------------------------")
        start_time = time.time()
        memory_samples = self.hippocampus.entorhinal_cortex.get_memory_sample()
        all_added_nodes = []
        all_connected_nodes = []
        all_added_edges = []
        for i, messages in enumerate(memory_samples, 1):
            all_topics = []
            progress = (i / len(memory_samples)) * 100
            bar_length = 30
            filled_length = int(bar_length * i // len(memory_samples))
            bar = "█" * filled_length + "-" * (bar_length - filled_length)
            logger.debug(f"进度: [{bar}] {progress:.1f}% ({i}/{len(memory_samples)})")

            compress_rate = self.config.memory_compress_rate
            compressed_memory, similar_topics_dict = await self.memory_compress(messages, compress_rate)
            logger.debug(f"压缩后记忆数量: {compressed_memory}，似曾相识的话题: {similar_topics_dict}")

            current_time = datetime.datetime.now().timestamp()
            logger.debug(f"添加节点: {', '.join(topic for topic, _ in compressed_memory)}")
            all_added_nodes.extend(topic for topic, _ in compressed_memory)
            
            for topic, memory in compressed_memory:
                self.memory_graph.add_dot(topic, memory)
                all_topics.append(topic)

                if topic in similar_topics_dict:
                    similar_topics = similar_topics_dict[topic]
                    for similar_topic, similarity in similar_topics:
                        if topic != similar_topic:
                            strength = int(similarity * 10)
                            
                            logger.debug(f"连接相似节点: {topic} 和 {similar_topic} (强度: {strength})")
                            all_added_edges.append(f"{topic}-{similar_topic}")
                            
                            all_connected_nodes.append(topic)
                            all_connected_nodes.append(similar_topic)
                            
                            self.memory_graph.G.add_edge(
                                topic,
                                similar_topic,
                                strength=strength,
                                created_time=current_time,
                                last_modified=current_time,
                            )

            for i in range(len(all_topics)):
                for j in range(i + 1, len(all_topics)):
                    logger.debug(f"连接同批次节点: {all_topics[i]} 和 {all_topics[j]}")
                    all_added_edges.append(f"{all_topics[i]}-{all_topics[j]}")
                    self.memory_graph.connect_dot(all_topics[i], all_topics[j])

        logger.success(f"更新记忆: {', '.join(all_added_nodes)}")
        logger.debug(f"强化连接: {', '.join(all_added_edges)}")
        logger.info(f"强化连接节点: {', '.join(all_connected_nodes)}")
        
        await self.hippocampus.entorhinal_cortex.sync_memory_to_db()
        
        end_time = time.time()
        logger.success(
            f"---------------------记忆构建耗时: {end_time - start_time:.2f} "
            "秒---------------------"
        )

    async def operation_forget_topic(self, percentage=0.1):
        logger.info("[遗忘] 开始检查数据库...")

        all_nodes = list(self.memory_graph.G.nodes())
        all_edges = list(self.memory_graph.G.edges())

        if not all_nodes and not all_edges:
            logger.info("[遗忘] 记忆图为空,无需进行遗忘操作")
            return

        check_nodes_count = max(1, int(len(all_nodes) * percentage))
        check_edges_count = max(1, int(len(all_edges) * percentage))

        nodes_to_check = random.sample(all_nodes, check_nodes_count)
        edges_to_check = random.sample(all_edges, check_edges_count)

        edge_changes = {"weakened": 0, "removed": 0}
        node_changes = {"reduced": 0, "removed": 0}

        current_time = datetime.datetime.now().timestamp()

        logger.info("[遗忘] 开始检查连接...")
        for source, target in edges_to_check:
            edge_data = self.memory_graph.G[source][target]
            last_modified = edge_data.get("last_modified")

            if current_time - last_modified > 3600 * self.config.memory_forget_time:
                current_strength = edge_data.get("strength", 1)
                new_strength = current_strength - 1

                if new_strength <= 0:
                    self.memory_graph.G.remove_edge(source, target)
                    edge_changes["removed"] += 1
                    logger.info(f"[遗忘] 连接移除: {source} -> {target}")
                else:
                    edge_data["strength"] = new_strength
                    edge_data["last_modified"] = current_time
                    edge_changes["weakened"] += 1
                    logger.info(f"[遗忘] 连接减弱: {source} -> {target} (强度: {current_strength} -> {new_strength})")

        logger.info("[遗忘] 开始检查节点...")
        for node in nodes_to_check:
            node_data = self.memory_graph.G.nodes[node]
            last_modified = node_data.get("last_modified", current_time)

            if current_time - last_modified > 3600 * 24:
                memory_items = node_data.get("memory_items", [])
                if not isinstance(memory_items, list):
                    memory_items = [memory_items] if memory_items else []

                if memory_items:
                    current_count = len(memory_items)
                    removed_item = random.choice(memory_items)
                    memory_items.remove(removed_item)

                    if memory_items:
                        self.memory_graph.G.nodes[node]["memory_items"] = memory_items
                        self.memory_graph.G.nodes[node]["last_modified"] = current_time
                        node_changes["reduced"] += 1
                        logger.info(f"[遗忘] 记忆减少: {node} (数量: {current_count} -> {len(memory_items)})")
                    else:
                        self.memory_graph.G.remove_node(node)
                        node_changes["removed"] += 1
                        logger.info(f"[遗忘] 节点移除: {node}")

        if any(count > 0 for count in edge_changes.values()) or any(count > 0 for count in node_changes.values()):
            await self.hippocampus.entorhinal_cortex.sync_memory_to_db()
            logger.info("[遗忘] 统计信息:")
            logger.info(f"[遗忘] 连接变化: {edge_changes['weakened']} 个减弱, {edge_changes['removed']} 个移除")
            logger.info(f"[遗忘] 节点变化: {node_changes['reduced']} 个减少记忆, {node_changes['removed']} 个移除")
        else:
            logger.info("[遗忘] 本次检查没有节点或连接满足遗忘条件")

# 海马体
class Hippocampus:
    def __init__(self):
        self.memory_graph = Memory_graph()
        self.llm_topic_judge = None
        self.llm_summary_by_topic = None
        self.entorhinal_cortex = None
        self.parahippocampal_gyrus = None
        self.config = None

    def initialize(self, global_config):
        self.config = MemoryConfig.from_global_config(global_config)
        # 初始化子组件
        self.entorhinal_cortex = EntorhinalCortex(self)
        self.parahippocampal_gyrus = ParahippocampalGyrus(self)
        # 从数据库加载记忆图
        self.entorhinal_cortex.sync_memory_from_db()
        self.llm_topic_judge = self.config.llm_topic_judge
        self.llm_summary_by_topic = self.config.llm_summary_by_topic

    def get_all_node_names(self) -> list:
        """获取记忆图中所有节点的名字列表"""
        return list(self.memory_graph.G.nodes())

    def calculate_node_hash(self, concept, memory_items) -> int:
        """计算节点的特征值"""
        if not isinstance(memory_items, list):
            memory_items = [memory_items] if memory_items else []
        sorted_items = sorted(memory_items)
        content = f"{concept}:{'|'.join(sorted_items)}"
        return hash(content)

    def calculate_edge_hash(self, source, target) -> int:
        """计算边的特征值"""
        nodes = sorted([source, target])
        return hash(f"{nodes[0]}:{nodes[1]}")

    def find_topic_llm(self, text, topic_num):
        prompt = (
            f"这是一段文字：{text}。请你从这段话中总结出最多{topic_num}个关键的概念，可以是名词，动词，或者特定人物，帮我列出来，"
            f"将主题用逗号隔开，并加上<>,例如<主题1>,<主题2>......尽可能精简。只需要列举最多{topic_num}个话题就好，不要有序号，不要告诉我其他内容。"
            f"如果找不出主题或者没有明显主题，返回<none>。"
        )
        return prompt

    def topic_what(self, text, topic, time_info):
        prompt = (
            f'这是一段文字，{time_info}：{text}。我想让你基于这段文字来概括"{topic}"这个概念，帮我总结成一句自然的话，'
            f"可以包含时间和人物，以及具体的观点。只输出这句话就好"
        )
        return prompt

    def calculate_topic_num(self, text, compress_rate):
        """计算文本的话题数量"""
        information_content = calculate_information_content(text)
        topic_by_length = text.count("\n") * compress_rate
        topic_by_information_content = max(1, min(5, int((information_content - 3) * 2)))
        topic_num = int((topic_by_length + topic_by_information_content) / 2)
        logger.debug(
            f"topic_by_length: {topic_by_length}, topic_by_information_content: {topic_by_information_content}, "
            f"topic_num: {topic_num}"
        )
        return topic_num

    def get_memory_from_keyword(self, keyword: str, max_depth: int = 2) -> list:
        """从关键词获取相关记忆。

        Args:
            keyword (str): 关键词
            max_depth (int, optional): 记忆检索深度，默认为2。1表示只获取直接相关的记忆，2表示获取间接相关的记忆。

        Returns:
            list: 记忆列表，每个元素是一个元组 (topic, memory_items, similarity)
                - topic: str, 记忆主题
                - memory_items: list, 该主题下的记忆项列表
                - similarity: float, 与关键词的相似度
        """
        if not keyword:
            return []

        # 获取所有节点
        all_nodes = list(self.memory_graph.G.nodes())
        memories = []

        # 计算关键词的词集合
        keyword_words = set(jieba.cut(keyword))

        # 遍历所有节点，计算相似度
        for node in all_nodes:
            node_words = set(jieba.cut(node))
            all_words = keyword_words | node_words
            v1 = [1 if word in keyword_words else 0 for word in all_words]
            v2 = [1 if word in node_words else 0 for word in all_words]
            similarity = cosine_similarity(v1, v2)

            # 如果相似度超过阈值，获取该节点的记忆
            if similarity >= 0.3:  # 可以调整这个阈值
                node_data = self.memory_graph.G.nodes[node]
                memory_items = node_data.get("memory_items", [])
                if not isinstance(memory_items, list):
                    memory_items = [memory_items] if memory_items else []
                
                memories.append((node, memory_items, similarity))

        # 按相似度降序排序
        memories.sort(key=lambda x: x[2], reverse=True)
        return memories

    async def get_memory_from_text(self, text: str, num: int = 5, max_depth: int = 2, 
                                 fast_retrieval: bool = False) -> list:
        """从文本中提取关键词并获取相关记忆。

        Args:
            text (str): 输入文本
            num (int, optional): 需要返回的记忆数量。默认为5。
            max_depth (int, optional): 记忆检索深度。默认为2。
            fast_retrieval (bool, optional): 是否使用快速检索。默认为False。
                如果为True，使用jieba分词和TF-IDF提取关键词，速度更快但可能不够准确。
                如果为False，使用LLM提取关键词，速度较慢但更准确。

        Returns:
            list: 记忆列表，每个元素是一个元组 (topic, memory_items, similarity)
                - topic: str, 记忆主题
                - memory_items: list, 该主题下的记忆项列表
                - similarity: float, 与文本的相似度
        """
        if not text:
            return []

        if fast_retrieval:
            # 使用jieba分词提取关键词
            words = jieba.cut(text)
            # 过滤掉停用词和单字词
            keywords = [word for word in words if len(word) > 1]
            # 去重
            keywords = list(set(keywords))
            # 限制关键词数量
            keywords = keywords[:5]
        else:
            # 使用LLM提取关键词
            topic_num = min(5, max(1, int(len(text) * 0.1)))  # 根据文本长度动态调整关键词数量
            topics_response = await self.llm_topic_judge.generate_response(
                self.find_topic_llm(text, topic_num)
            )

            # 提取关键词
            keywords = re.findall(r'<([^>]+)>', topics_response[0])
            if not keywords:
                keywords = ['none']
            else:
                keywords = [
                    keyword.strip()
                    for keyword in ','.join(keywords).replace("，", ",").replace("、", ",").replace(" ", ",").split(",")
                    if keyword.strip()
                ]

        # 从每个关键词获取记忆
        all_memories = []
        for keyword in keywords:
            memories = self.get_memory_from_keyword(keyword, max_depth)
            all_memories.extend(memories)

        # 去重（基于主题）
        seen_topics = set()
        unique_memories = []
        for topic, memory_items, similarity in all_memories:
            if topic not in seen_topics:
                seen_topics.add(topic)
                unique_memories.append((topic, memory_items, similarity))

        # 按相似度排序并返回前num个
        unique_memories.sort(key=lambda x: x[2], reverse=True)
        return unique_memories[:num]

# driver = get_driver()
# config = driver.config

start_time = time.time()

# 创建记忆图
memory_graph = Memory_graph()
# 创建海马体
hippocampus = Hippocampus()

# 从全局配置初始化记忆系统
from ..chat.config import global_config
hippocampus.initialize(global_config=global_config)

end_time = time.time()
logger.success(f"加载海马体耗时: {end_time - start_time:.2f} 秒")
