from src.common.logger_manager import get_logger
from src.common.database import db # 全局数据库对象
from src.config.config import global_config # 全局配置
from src.chat.person_info.person_info import person_info_manager # 用于获取 person_id 和 person_name
# 假设 relationship_manager 中有通用的 get_person_names_batch，或者这个功能已移至 person_info_manager
from src.chat.person_info.relationship_manager import relationship_manager

from typing import List, Dict, Any, Optional

logger = get_logger("ProfileManager")

class ProfileManager:
    """
    管理用户画像信息（profile_info），特别是绰号数据的读取和格式化。
    """

    def __init__(self):
        # profile_info 集合的名称，从配置读取，默认为 "profile_info"
        self.profile_info_collection_name = global_config.group_sobriquet.get(
            "profile_info_collection_name", "profile_info"
        )
        self.profile_collection = getattr(db, self.profile_info_collection_name, None)

        if self.profile_collection is None:
            logger.error(
                f"未能从数据库获取 '{self.profile_info_collection_name}' 集合。"
                "ProfileManager 的功能将受限。"
            )
        else:
            logger.info(
                f"ProfileManager 初始化成功，使用集合: '{self.profile_info_collection_name}'"
            )

    def is_available(self) -> bool:
        """检查 profile_info 集合是否可用。"""
        return self.profile_collection is not None

    async def get_users_group_sobriquets_for_prompt_injection_data(
        self, platform: str, user_ids: List[str], group_id: str
    ) -> Dict[str, Dict[str, Any]]:
        """
        批量获取多个用户在指定群组的绰号信息，用于 Prompt 注入。
        数据源：profile_info 集合 (用于绰号) 和 person_info 相关方法 (用于 person_name)。

        Args:
            platform (str): 平台名称。
            user_ids (List[str]): 用户平台特定 ID 列表 (字符串形式)。
            group_id (str): 群组 ID (字符串形式)。

        Returns:
            Dict[str, Dict[str, Any]]:
                键为 person_name (如果获取不到则可能跳过该用户或使用 fallback),
                值为 {"user_id": "原始平台用户ID字符串", "sobriquets": [{"绰号A": 次数}, ...]}
        """
        if not self.is_available():
            logger.error("ProfileManager: profile_info 集合不可用，无法获取绰号数据。")
            return {}
        if not user_ids or not group_id:
            logger.debug("ProfileManager: user_ids 或 group_id 为空，返回空字典。")
            return {}

        user_ids_str = [str(uid) for uid in user_ids]
        group_id_str = str(group_id)
        
        # 1. 获取 person_id 列表 和 user_id 到 person_id 的映射
        user_id_to_person_id_map: Dict[str, str] = {}
        person_ids_to_query: List[str] = []
        for uid_str in user_ids_str:
            # 使用 person_info_manager 获取 person_id
            pid = person_info_manager.get_person_id(platform, uid_str)
            if pid:
                user_id_to_person_id_map[uid_str] = pid
                if pid not in person_ids_to_query:
                    person_ids_to_query.append(pid)
        
        if not person_ids_to_query:
            logger.debug(f"ProfileManager: 未能从 user_ids: {user_ids_str} (平台: {platform}) 获取到任何有效的 person_id。")
            return {}

        # 2. 批量获取 person_names
        # 假设 relationship_manager.get_person_names_batch 或 person_info_manager.get_person_names_batch 可用
        # 返回格式: {user_id_str: person_name}
        try:
            # 注意: relationship_manager.get_person_names_batch 内部会处理 person_id 的转换
            person_names_map: Dict[str, str] = await relationship_manager.get_person_names_batch(platform, user_ids_str)
        except Exception as e:
            logger.error(f"ProfileManager: 调用 get_person_names_batch 时出错: {e}", exc_info=True)
            person_names_map = {}


        # 3. 查询 profile_info 集合获取绰号数据
        sobriquets_result_map: Dict[str, Dict[str, Any]] = {}
        group_key_in_db = f"{platform}-{group_id_str}" # 群组在数据库中的键

        try:
            # logger.debug(f"ProfileManager: 查询 profile_info 获取绰号 for person_ids: {person_ids_to_query}, group_key: {group_key_in_db}")
            profile_cursor = self.profile_collection.find(
                {"_id": {"$in": person_ids_to_query}}, # _id 在 profile_info 中是 person_id
                {
                    "_id": 1, # 返回 person_id (_id)
                    f"sobriquets_by_group.{group_key_in_db}.sobriquets": 1 # 只获取特定群组的绰号列表
                }
            )
            
            async for profile_doc in profile_cursor: # 假设使用的是异步MongoDB驱动 (如 Motor)
                person_id_from_doc = profile_doc.get("_id")
                
                # 安全地获取绰号列表
                group_data = profile_doc.get("sobriquets_by_group", {}).get(group_key_in_db)
                raw_sobriquets_list = group_data.get("sobriquets", []) if group_data else []

                if not raw_sobriquets_list:
                    continue # 此用户在此群组没有绰号记录

                # 格式化绰号为 [{"绰号名": 次数}, ...]
                formatted_sobriquets = []
                for item in raw_sobriquets_list:
                    if (isinstance(item, dict) and 
                        isinstance(item.get("name"), str) and 
                        isinstance(item.get("count"), int) and item["count"] > 0):
                        formatted_sobriquets.append({item["name"]: item["count"]})
                    # else:
                        # logger.warning(f"ProfileManager: profile_info 中 person_id '{person_id_from_doc}' 群组 '{group_key_in_db}' 的绰号格式无效: {item}")
                
                if not formatted_sobriquets:
                    continue

                # 找到此 person_id 对应的原始 user_id
                original_user_id: Optional[str] = None
                for uid_map_key, pid_map_val in user_id_to_person_id_map.items():
                    if pid_map_val == person_id_from_doc:
                        original_user_id = uid_map_key
                        break
                
                if not original_user_id:
                    # logger.warning(f"ProfileManager: 在 profile_info 中找到 person_id '{person_id_from_doc}' 的绰号，但无法映射回原始 user_id。")
                    continue

                # 获取 person_name
                person_name = person_names_map.get(original_user_id)
                if not person_name:
                    # logger.warning(f"ProfileManager: 用户 (ID: {original_user_id}, PersonID: {person_id_from_doc}) 在群组 '{group_id_str}' 有绰号，但未能获取其 person_name。将跳过此用户。")
                    # 或者使用 fallback 名称:
                    # person_name = f"用户({original_user_id[-4:]})"
                    continue # 如果严格要求 person_name 作为键，则跳过

                sobriquets_result_map[person_name] = {
                    "user_id": original_user_id,
                    "sobriquets": formatted_sobriquets # 确保键名是 "sobriquets"
                }
            
            # logger.debug(
            #    f"ProfileManager: 批量获取群组 {group_id_str} 中 {len(user_ids_str)} 个用户的绰号，"
            #    f"格式化并返回 {len(sobriquets_result_map)} 个用户的数据。"
            # )

        except AttributeError as e: # getattr(db, ...) 可能失败
            logger.error(f"ProfileManager: 访问数据库集合 '{self.profile_info_collection_name}' 时出错: {e}。")
        except Exception as e:
            logger.error(f"ProfileManager: 批量获取群组绰号时发生意外错误: {e}", exc_info=True)

        return sobriquets_result_map

# 创建 ProfileManager 的单例实例 (如果适用，或者在使用它的地方创建)
profile_manager = ProfileManager()
