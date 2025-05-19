from pymongo.collection import Collection
from pymongo.errors import OperationFailure, DuplicateKeyError
from src.common.logger_manager import get_logger
from typing import Optional, List, Dict, Any

logger = get_logger("sobriquet_db")


class SobriquetDB:
    """
    处理与用户画像（包括绰号）相关的数据库操作 (MongoDB)。
    封装了对 'profile_info' 集合的读写操作。
    """

    def __init__(self, profile_info_collection: Optional[Collection]):
        """
        初始化 SobriquetDB 处理器。

        Args:
            profile_info_collection: MongoDB 'profile_info' 集合对象。
                                    如果为 None，则数据库操作将被禁用。
        """
        if profile_info_collection is None:
            logger.error("未提供 profile_info 集合，SobriquetDB 操作将被禁用。")
            self.profile_info_collection = None
        else:
            self.profile_info_collection = profile_info_collection
            logger.info("SobriquetDB 初始化成功，使用 profile_info 集合。")

    def is_available(self) -> bool:
        """检查数据库集合是否可用。"""
        return self.profile_info_collection is not None

    def ensure_profile_and_platform_user(self, person_id: str, platform: str, user_id_str: str):
        """
        确保 profile_info 中存在 person_id 的文档（作为 _id），并记录其平台和用户ID。
        如果文档或平台信息不存在，则创建它们。
        新的 platforms 结构: "platforms": { "qq": ["id1", "id2"], "wechat": ["id3"] }

        Args:
            person_id: 用户的唯一标识 person_id (将作为 MongoDB 的 _id)。
            platform: 平台名称 (例如 "qq", "wechat")。
            user_id_str: 用户在该平台上的字符串 ID。
        """
        if not self.is_available():
            logger.error(f"数据库集合不可用，无法执行 ensure_profile_and_platform_user for person_id: {person_id}。")
            return

        try:
            # 步骤 1: Upsert person_id 文档，确保基本结构存在
            self.profile_info_collection.update_one(
                {"_id": person_id},
                {
                    "$setOnInsert": {
                        "platforms": {},  # 初始化为对象
                        "sobriquets_by_group": {}
                    }
                },
                upsert=True
            )
            # logger.debug(f"为 person_id '{person_id}' 执行了 Upsert 确保文档和基本结构存在。")

            # 步骤 2: 更新 platforms 对象中的特定平台的用户ID列表
            # 构造更新操作，使用 $addToSet 将 user_id_str 添加到 platform 对应的数组中
            # 如果 platform 键或其数组不存在， $addToSet 会创建它们（需要MongoDB 4.2+对空数组或非数组字段的行为）
            # 为了更安全，可以先检查平台键是否存在，或者依赖 $addToSet 的行为
            
            # 检查平台键是否存在，如果不存在，则初始化为空数组
            # 这个步骤是为了确保 platforms.{platform} 是一个数组，然后才能 $addToSet
            # update_init_platform_array = self.profile_info_collection.update_one(
            #     {"_id": person_id, f"platforms.{platform}": {"$exists": False}},
            #     {"$set": {f"platforms.{platform}": []}}
            # )
            # if update_init_platform_array.modified_count > 0:
            #    logger.debug(f"为 person_id '{person_id}' 初始化了平台 '{platform}' 的用户ID空数组。")

            # 将 user_id_str 添加到数组中，确保唯一性
            update_result = self.profile_info_collection.update_one(
                {"_id": person_id},
                {"$addToSet": {f"platforms.{platform}": user_id_str}}
            )

            if update_result.modified_count > 0:
                logger.debug(f"为 person_id '{person_id}' 的平台 '{platform}' 添加/更新了 user_id '{user_id_str}'。")
            # else:
                # logger.debug(f"person_id '{person_id}' 的平台 '{platform}' 中 user_id '{user_id_str}' 已存在或未执行添加。")

        except OperationFailure as op_err:
            logger.exception(f"数据库操作失败 (OperationFailure) for person_id '{person_id}', platform '{platform}': {op_err}")
            raise # 重新抛出异常，让上层处理
        except Exception as e:
            logger.exception(f"确保用户画像和平台信息时发生意外错误 for person_id '{person_id}', platform '{platform}': {e}")
            raise # 重新抛出异常

    def update_group_sobriquet_count(self, person_id: str, platform: str, group_id_str: str, sobriquet_name: str):
        """
        更新 person_id 文档中特定群组的特定绰号的计数。
        如果绰号或群组条目不存在，则会创建它们。

        Args:
            person_id: 目标文档的 _id。
            platform: 平台名称。
            group_id_str: 目标群组的 ID (字符串)。
            sobriquet_name: 要更新或添加的绰号名称。
        """
        if not self.is_available():
            logger.error(f"数据库集合不可用，无法执行 update_group_sobriquet_count for person_id: {person_id}。")
            return

        group_key = f"{platform}-{group_id_str}"
        sobriquets_array_path = f"sobriquets_by_group.{group_key}.sobriquets"
        
        try:
            # 步骤 1: 尝试增加现有群组中现有绰号的计数
            # 使用 arrayFilters 来定位正确的绰号条目并增加其 count
            result_inc = self.profile_info_collection.update_one(
                {
                    "_id": person_id,
                    f"sobriquets_by_group.{group_key}.sobriquets.name": sobriquet_name 
                    # 确保群组和绰号名称都匹配
                },
                {"$inc": {f"{sobriquets_array_path}.$[elem].count": 1}},
                array_filters=[{"elem.name": sobriquet_name}]
            )

            if result_inc.modified_count > 0:
                # logger.debug(f"成功增加 person_id '{person_id}' 在群组 '{group_key}' 中绰号 '{sobriquet_name}' 的计数。")
                return

            # 步骤 2: 如果上一步未修改 (绰号不存在于数组中，或群组对象/绰号数组不存在)
            # 尝试将新绰号添加到现有群组的 sobriquets 数组中
            # 这个操作要求 sobriquets_by_group.{group_key} 对象存在
            result_push_sobriquet = self.profile_info_collection.update_one(
                {
                    "_id": person_id,
                    f"sobriquets_by_group.{group_key}": {"$exists": True} # 确保群组对象存在
                },
                {"$push": {sobriquets_array_path: {"name": sobriquet_name, "count": 1}}}
            )

            if result_push_sobriquet.modified_count > 0:
                logger.debug(f"成功为 person_id '{person_id}' 在现有群组 '{group_key}' 中添加新绰号 '{sobriquet_name}'。")
                return

            # 步骤 3: 如果上一步也未修改 (可能是 group_key 本身就不存在于 sobriquets_by_group 中)
            # 则创建新的群组条目 (group_key) 并添加绰号
            # $set 会创建 sobriquets_by_group 和 group_key (如果它们不存在)
            result_set_group = self.profile_info_collection.update_one(
                {"_id": person_id},
                {
                    "$set": {
                        f"sobriquets_by_group.{group_key}": {
                            "sobriquets": [{"name": sobriquet_name, "count": 1}]
                        }
                    }
                }
            )

            if result_set_group.modified_count > 0:
                 logger.debug(f"为 person_id '{person_id}' 添加了新的群组条目 '{group_key}' 和绰号 '{sobriquet_name}'。")
            # else:
                # 如果执行到这里且 modified_count 为 0，可能意味着文档已存在且与 $set 内容相同，
                # 或者在并发场景下，其他操作已创建了该结构。
                # logger.warning(f"尝试为 person_id '{person_id}' 添加新群组 '{group_key}' 或绰号 '{sobriquet_name}' 未修改文档。这可能表示数据已存在。Result: {result_set_group.raw_result}")

        except OperationFailure as db_err:
            logger.exception(
                f"数据库操作失败 ({type(db_err).__name__}): person_id '{person_id}', 群组 '{group_key}', 绰号 '{sobriquet_name}'. 错误: {db_err}"
            )
            raise
        except Exception as e:
            logger.exception(
                f"更新群组绰号计数时发生意外错误: person_id '{person_id}', group_key '{group_key}', sobriquet '{sobriquet_name}'. Error: {e}"
            )
            raise
