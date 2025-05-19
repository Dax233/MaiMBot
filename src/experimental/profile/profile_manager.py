import hashlib
from src.common.logger_manager import get_logger
from src.common.database import db
from src.config.config import global_config
from src.chat.person_info.person_info import person_info_manager # 用于获取 person_info_pid
from src.chat.person_info.relationship_manager import relationship_manager # 用于获取 person_name

from typing import List, Dict, Any, Optional

logger = get_logger("ProfileManager")

class ProfileManager:
    """
    管理用户画像信息（profile_info）。
    在过渡阶段，profile_info 的文档ID (_id) 基于 person_info_pid 生成，
    并且主要迁移绰号相关功能。
    """

    def __init__(self):
        self.profile_info_collection_name = global_config.profile.get(
            "profile_info_collection_name", "profile_info"
        )
        self.profile_collection = getattr(db, self.profile_info_collection_name, None)
        
        self.profile_id_salt = global_config.security.get("profile_id_salt", "default_salt_please_change_me")
        if self.profile_id_salt == "default_salt_please_change_me":
            logger.warning("安全警告：正在使用默认的 profile_id_salt。请在配置中设置一个强盐值！")

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

    def generate_profile_document_id(self, person_info_pid: str) -> str:
        """
        为 profile_info 文档生成 _id，基于 person_info_pid 的加盐哈希。

        Args:
            person_info_pid (str): 从 person_info 表获取的代表人格的ID。

        Returns:
            str: 生成的哈希字符串，用作 profile_info 文档的 _id。
        """
        if not person_info_pid:
            logger.error("生成 profile_document_id 时，person_info_pid 为空。")
            raise ValueError("person_info_pid cannot be empty for ID generation.")
            
        salted_input = f"{self.profile_id_salt}-{person_info_pid}"
        hashed_id = hashlib.sha256(salted_input.encode('utf-8')).hexdigest()
        return hashed_id

    async def get_users_group_sobriquets_for_prompt_injection_data(
        self, platform: str, platform_user_ids: List[str], group_id: str
    ) -> Dict[str, Dict[str, Any]]:
        """
        批量获取多个用户在指定群组的绰号信息，用于 Prompt 注入。
        数据源：profile_info 集合 (用于绰号), person_info 相关 (用于 person_name 和 person_info_pid)。

        Args:
            platform (str): 平台名称。
            platform_user_ids (List[str]): 用户平台特定 ID 列表 (字符串形式)。
            group_id (str): 群组 ID (字符串形式)。

        Returns:
            Dict[str, Dict[str, Any]]:
                键为 person_name (从 person_info 获取),
                值为 {"user_id": "原始平台用户ID字符串", "sobriquets": [{"绰号A": 次数}, ...]}
        """
        if not self.is_available():
            logger.error("ProfileManager: profile_info 集合不可用，无法获取绰号数据。")
            return {}
        if not platform_user_ids or not group_id:
            logger.debug("ProfileManager: platform_user_ids 或 group_id 为空，返回空字典。")
            return {}

        user_ids_str = [str(uid) for uid in platform_user_ids]
        group_id_str = str(group_id)
        
        # 1. 获取 person_info_pid 并生成 profile_document_id 列表
        #    同时建立 profile_document_id 到原始平台用户ID的映射
        profile_doc_ids_to_query: List[str] = []
        profile_doc_id_to_original_uid_map: Dict[str, str] = {}

        for uid_str in user_ids_str:
            person_info_pid = person_info_manager.get_person_id(platform, uid_str) # 获取现有的自然人ID
            if person_info_pid:
                try:
                    profile_doc_id = self.generate_profile_document_id(person_info_pid)
                    if profile_doc_id not in profile_doc_ids_to_query:
                        profile_doc_ids_to_query.append(profile_doc_id)
                    profile_doc_id_to_original_uid_map[profile_doc_id] = uid_str
                except ValueError as e:
                    logger.error(f"为 person_info_pid '{person_info_pid}' (来自 uid '{uid_str}') 生成 profile_doc_id 失败: {e}")
            # else:
                # logger.debug(f"ProfileManager: 未能为平台用户 '{uid_str}' (平台 '{platform}') 获取 person_info_pid。")
        
        if not profile_doc_ids_to_query:
            logger.debug(f"ProfileManager: 未能为提供的 platform_user_ids 生成任何有效的 profile_document_id。")
            return {}

        # 2. 批量获取 person_names (从 person_info 集合，键为原始平台用户ID)
        try:
            person_names_map: Dict[str, str] = await relationship_manager.get_person_names_batch(platform, user_ids_str)
        except Exception as e:
            logger.error(f"ProfileManager: 调用 get_person_names_batch 时出错: {e}", exc_info=True)
            person_names_map = {}

        # 3. 查询 profile_info 集合获取绰号数据
        sobriquets_result_map: Dict[str, Dict[str, Any]] = {}
        group_key_in_db = f"{platform}-{group_id_str}"

        try:
            profile_cursor = self.profile_collection.find(
                {"_id": {"$in": profile_doc_ids_to_query}},
                {
                    "_id": 1, # profile_document_id
                    f"sobriquets_by_group.{group_key_in_db}.sobriquets": 1,
                    # "person_info_pid_ref": 1 # 可选，用于验证
                }
            )
            
            async for profile_doc in profile_cursor:
                profile_document_id_from_doc = profile_doc.get("_id")
                
                group_data = profile_doc.get("sobriquets_by_group", {}).get(group_key_in_db)
                raw_sobriquets_list = group_data.get("sobriquets", []) if group_data else []

                if not raw_sobriquets_list:
                    continue

                formatted_sobriquets = []
                for item in raw_sobriquets_list:
                    if (isinstance(item, dict) and 
                        isinstance(item.get("name"), str) and 
                        isinstance(item.get("count"), int) and item["count"] > 0):
                        formatted_sobriquets.append({item["name"]: item["count"]})
                
                if not formatted_sobriquets:
                    continue

                original_platform_user_id = profile_doc_id_to_original_uid_map.get(profile_document_id_from_doc)
                if not original_platform_user_id:
                    continue

                person_name = person_names_map.get(original_platform_user_id)
                if not person_name: # 在过渡阶段，我们仍以 person_name 为键返回
                    # logger.warning(f"ProfileManager: 用户 (平台ID: {original_platform_user_id}) 有绰号但未能获取其 person_name。")
                    continue 

                sobriquets_result_map[person_name] = {
                    "user_id": original_platform_user_id,
                    "sobriquets": formatted_sobriquets
                }
        except Exception as e:
            logger.error(f"ProfileManager: 批量获取群组绰号时发生意外错误: {e}", exc_info=True)

        return sobriquets_result_map

profile_manager = ProfileManager()
