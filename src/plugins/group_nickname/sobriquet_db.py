from pymongo.collection import Collection
from pymongo.errors import OperationFailure, DuplicateKeyError
from src.common.logger_manager import get_logger
from typing import Optional

logger = get_logger("sobriquet_db")

class SobriquetDB:
    """
    处理与用户画像（profile_info）相关的数据库操作 (MongoDB)。
    在过渡阶段，主要负责绰号数据的写入和 profile_info 文档基本结构的维护。
    文档的 _id 是基于 person_info_pid 生成的 profile_document_id。
    """

    def __init__(self, profile_info_collection: Optional[Collection]):
        if profile_info_collection is None:
            logger.error("未提供 profile_info 集合，SobriquetDB 操作将被禁用。")
            self.profile_info_collection = None
        else:
            self.profile_info_collection = profile_info_collection
            logger.info("SobriquetDB 初始化成功，使用 profile_info 集合。")

    def is_available(self) -> bool:
        return self.profile_info_collection is not None

    def ensure_profile_document_exists(self, 
                                     profile_document_id: str, 
                                     person_info_pid_ref: str, 
                                     platform: str, 
                                     platform_user_id: str):
        """
        确保 profile_info 中存在指定 profile_document_id 的文档。
        如果文档不存在，则使用提供的用户信息创建它，并记录 person_info_pid_ref 及平台账户信息。

        Args:
            profile_document_id (str): profile_info 文档的 _id (基于 person_info_pid 哈希生成)。
            person_info_pid_ref (str): 原始的 person_info 表的 person_id，用于引用。
            platform (str): 平台名称。
            platform_user_id (str): 用户在该平台上的ID。
        """
        if not self.is_available():
            logger.error(f"数据库集合不可用，无法执行 ensure_profile_document_exists for profile_id: {profile_document_id}。")
            return

        try:
            # 步骤 1: Upsert profile_document_id 文档，确保基本结构存在
            update_result = self.profile_info_collection.update_one(
                {"_id": profile_document_id},
                {
                    "$setOnInsert": {
                        "person_info_pid_ref": person_info_pid_ref,
                        "platform_accounts": {}, # 初始化为空对象
                        "sobriquets_by_group": {}  # 初始化为空对象
                    }
                },
                upsert=True
            )
            # if update_result.upserted_id:
            #    logger.debug(f"为 profile_document_id '{profile_document_id}' (person_info_pid_ref: {person_info_pid_ref}) 创建了新的 profile_info 文档。")

            # 步骤 2: 确保平台账户信息被记录或更新
            # 将 platform_user_id 添加到 platform_accounts.{platform} 数组中
            # $addToSet 确保了数组元素的唯一性
            self.profile_info_collection.update_one(
                {"_id": profile_document_id},
                {"$addToSet": {f"platform_accounts.{platform}": platform_user_id}}
            )
            # logger.debug(f"为 profile_document_id '{profile_document_id}' 的平台 '{platform}' 添加/更新了 platform_user_id '{platform_user_id}'。")

        except OperationFailure as op_err:
            logger.exception(f"数据库操作失败 (OperationFailure) for profile_document_id '{profile_document_id}': {op_err}")
            raise
        except Exception as e:
            logger.exception(f"确保 profile_info 文档存在时发生意外错误 for profile_document_id '{profile_document_id}': {e}")
            raise

    def update_group_sobriquet_count(self, profile_document_id: str, platform: str, group_id_str: str, sobriquet_name: str):
        """
        更新 profile_document_id 对应文档中特定群组的特定绰号的计数。
        文档的 _id 是基于 person_info_pid 生成的 profile_document_id。
        """
        if not self.is_available():
            logger.error(f"数据库集合不可用，无法执行 update_group_sobriquet_count for profile_id: {profile_document_id}。")
            return

        group_key = f"{platform}-{group_id_str}"
        sobriquets_array_path = f"sobriquets_by_group.{group_key}.sobriquets"
        
        try:
            # 步骤 1: 尝试增加现有绰号的计数
            result_inc = self.profile_info_collection.update_one(
                {
                    "_id": profile_document_id,
                    f"{sobriquets_array_path}.name": sobriquet_name 
                },
                {"$inc": {f"{sobriquets_array_path}.$[elem].count": 1}},
                array_filters=[{"elem.name": sobriquet_name}]
            )

            if result_inc.modified_count > 0:
                return

            # 步骤 2: 如果绰号不存在，尝试添加到现有群组的绰号数组
            result_push_sobriquet = self.profile_info_collection.update_one(
                {
                    "_id": profile_document_id,
                    f"sobriquets_by_group.{group_key}": {"$exists": True} 
                },
                {"$push": {sobriquets_array_path: {"name": sobriquet_name, "count": 1}}}
            )

            if result_push_sobriquet.modified_count > 0:
                logger.debug(f"为 profile_id '{profile_document_id}' 在现有群组 '{group_key}' 中添加新绰号 '{sobriquet_name}'。")
                return

            # 步骤 3: 如果群组键本身不存在，则创建新的群组条目并添加绰号
            result_set_group = self.profile_info_collection.update_one(
                {"_id": profile_document_id},
                {
                    "$set": {
                        f"sobriquets_by_group.{group_key}": {
                            "sobriquets": [{"name": sobriquet_name, "count": 1}]
                        }
                    }
                }
            )
            if result_set_group.modified_count > 0:
                 logger.debug(f"为 profile_id '{profile_document_id}' 添加了新群组条目 '{group_key}' 和绰号 '{sobriquet_name}'。")

        except OperationFailure as db_err:
            logger.exception(
                f"数据库操作失败 ({type(db_err).__name__}): profile_id '{profile_document_id}', 群组 '{group_key}', 绰号 '{sobriquet_name}'. 错误: {db_err}")
            raise
        except Exception as e:
            logger.exception(
                f"更新群组绰号计数时发生意外错误: profile_id '{profile_document_id}', group_key '{group_key}', sobriquet '{sobriquet_name}'. Error: {e}")
            raise
