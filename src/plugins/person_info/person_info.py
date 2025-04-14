from src.common.logger import get_module_logger
from ...common.database import db
import copy
import hashlib
from typing import Any, Callable, Dict
import datetime
import asyncio
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd


"""
PersonInfoManager 类方法功能摘要：
1. get_person_id - 根据平台和用户ID生成MD5哈希的唯一person_id
2. create_person_info - 创建新个人信息文档（自动合并默认值）
3. update_one_field - 更新单个字段值（若文档不存在则创建）
4. del_one_document - 删除指定person_id的文档
5. get_value - 获取单个字段值（返回实际值或默认值）
6. get_values - 批量获取字段值（任一字段无效则返回空字典）
7. del_all_undefined_field - 清理全集合中未定义的字段
8. get_specific_value_list - 根据指定条件，返回person_id,value字典
9. personal_habit_deduction - 定时推断个人习惯
"""

logger = get_module_logger("person_info")

person_info_default = {
    "person_id": None,
    "platform": None,
    "user_id": None,
    "nickname": None,
    # "age" : 0,
    "relationship_value": 0,
    # "saved" : True,
    # "impression" : None,
    # "gender" : Unkown,
    "konw_time": 0,
    "msg_interval": 3000,
    "msg_interval_list": [],
}  # 个人信息的各项与默认值在此定义，以下处理会自动创建/补全每一项


class PersonInfoManager:
    def __init__(self):
        if "person_info" not in db.list_collection_names():
            db.create_collection("person_info")
            db.person_info.create_index("person_id", unique=True)

    def get_person_id(self, platform: str, user_id: int):
        """获取唯一id"""
        components = [platform, str(user_id)]
        key = "_".join(components)
        return hashlib.md5(key.encode()).hexdigest()

    async def create_person_info(self, person_id: str, data: dict = None):
        """创建一个项"""
        if not person_id:
            logger.debug("创建失败，personid不存在")
            return

        _person_info_default = copy.deepcopy(person_info_default)
        _person_info_default["person_id"] = person_id

        if data:
            for key in _person_info_default:
                if key != "person_id" and key in data:
                    _person_info_default[key] = data[key]

        db.person_info.insert_one(_person_info_default)

    async def update_one_field(self, person_id: str, field_name: str, value, Data: dict = None):
        """更新某一个字段，会补全"""
        if field_name not in person_info_default.keys():
            logger.debug(f"更新'{field_name}'失败，未定义的字段")
            return

        document = db.person_info.find_one({"person_id": person_id})

        if document:
            db.person_info.update_one({"person_id": person_id}, {"$set": {field_name: value}})
        else:
            Data[field_name] = value
            logger.debug(f"更新时{person_id}不存在，已新建")
            await self.create_person_info(person_id, Data)

    async def del_one_document(self, person_id: str):
        """删除指定 person_id 的文档"""
        if not person_id:
            logger.debug("删除失败：person_id 不能为空")
            return

        result = db.person_info.delete_one({"person_id": person_id})
        if result.deleted_count > 0:
            logger.debug(f"删除成功：person_id={person_id}")
        else:
            logger.debug(f"删除失败：未找到 person_id={person_id}")

    async def get_value(self, person_id: str, field_name: str):
        """获取指定person_id文档的字段值，若不存在该字段，则返回该字段的全局默认值"""
        if not person_id:
            logger.debug("get_value获取失败：person_id不能为空")
            return None

        if field_name not in person_info_default:
            logger.debug(f"get_value获取失败：字段'{field_name}'未定义")
            return None

        document = db.person_info.find_one({"person_id": person_id}, {field_name: 1})

        if document and field_name in document:
            return document[field_name]
        else:
            default_value = copy.deepcopy(person_info_default[field_name])
            logger.trace(f"获取{person_id}的{field_name}失败，已返回默认值{default_value}")
            return default_value

    async def get_values(self, person_id: str, field_names: list) -> dict:
        """获取指定person_id文档的多个字段值，若不存在该字段，则返回该字段的全局默认值"""
        if not person_id:
            logger.debug("get_values获取失败：person_id不能为空")
            return {}

        # 检查所有字段是否有效
        for field in field_names:
            if field not in person_info_default:
                logger.debug(f"get_values获取失败：字段'{field}'未定义")
                return {}

        # 构建查询投影（所有字段都有效才会执行到这里）
        projection = {field: 1 for field in field_names}

        document = db.person_info.find_one({"person_id": person_id}, projection)

        result = {}
        for field in field_names:
            result[field] = copy.deepcopy(
                document.get(field, person_info_default[field]) if document else person_info_default[field]
            )

        return result

    async def del_all_undefined_field(self):
        """删除所有项里的未定义字段"""
        # 获取所有已定义的字段名
        defined_fields = set(person_info_default.keys())

        try:
            # 遍历集合中的所有文档
            for document in db.person_info.find({}):
                # 找出文档中未定义的字段
                undefined_fields = set(document.keys()) - defined_fields - {"_id"}

                if undefined_fields:
                    # 构建更新操作，使用$unset删除未定义字段
                    update_result = db.person_info.update_one(
                        {"_id": document["_id"]}, {"$unset": {field: 1 for field in undefined_fields}}
                    )

                    if update_result.modified_count > 0:
                        logger.debug(f"已清理文档 {document['_id']} 的未定义字段: {undefined_fields}")

            return

        except Exception as e:
            logger.error(f"清理未定义字段时出错: {e}")
            return

    async def get_specific_value_list(
        self,
        field_name: str,
        way: Callable[[Any], bool],  # 接受任意类型值
    ) -> Dict[str, Any]:
        """
        获取满足条件的字段值字典

        Args:
            field_name: 目标字段名
            way: 判断函数 (value: Any) -> bool

        Returns:
            {person_id: value} | {}

        Example:
            # 查找所有nickname包含"admin"的用户
            result = manager.specific_value_list(
                "nickname",
                lambda x: "admin" in x.lower()
            )
        """
        if field_name not in person_info_default:
            logger.error(f"字段检查失败：'{field_name}'未定义")
            return {}

        try:
            result = {}
            for doc in db.person_info.find({field_name: {"$exists": True}}, {"person_id": 1, field_name: 1, "_id": 0}):
                try:
                    value = doc[field_name]
                    if way(value):
                        result[doc["person_id"]] = value
                except (KeyError, TypeError, ValueError) as e:
                    logger.debug(f"记录{doc.get('person_id')}处理失败: {str(e)}")
                    continue

            return result

        except Exception as e:
            logger.error(f"数据库查询失败: {str(e)}", exc_info=True)
            return {}

    async def personal_habit_deduction(self):
        """启动个人信息推断，每天根据一定条件推断一次"""
        try:
            while 1:
                await asyncio.sleep(60)
                current_time = datetime.datetime.now()
                logger.info(f"个人信息推断启动: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")

                # "msg_interval"推断
                msg_interval_map = False
                msg_interval_lists = await self.get_specific_value_list(
                    "msg_interval_list", lambda x: isinstance(x, list) and len(x) >= 100
                )
                for person_id, msg_interval_list_ in msg_interval_lists.items():
                    try:
                        time_interval = []
                        for t1, t2 in zip(msg_interval_list_, msg_interval_list_[1:]):
                            delta = t2 - t1
                            if delta > 0:
                                time_interval.append(delta)

                        time_interval = [t for t in time_interval if 500 <= t <= 8000]
                        if len(time_interval) >= 30:
                            time_interval.sort()

                            # 画图(log)
                            msg_interval_map = True
                            log_dir = Path("logs/person_info")
                            log_dir.mkdir(parents=True, exist_ok=True)
                            plt.figure(figsize=(10, 6))
                            time_series = pd.Series(time_interval)
                            plt.hist(time_series, bins=50, density=True, alpha=0.4, color="pink", label="Histogram")
                            time_series.plot(kind="kde", color="mediumpurple", linewidth=1, label="Density")
                            plt.grid(True, alpha=0.2)
                            plt.xlim(0, 8000)
                            plt.title(f"Message Interval Distribution (User: {person_id[:8]}...)")
                            plt.xlabel("Interval (ms)")
                            plt.ylabel("Density")
                            plt.legend(framealpha=0.9, facecolor="white")
                            img_path = log_dir / f"interval_distribution_{person_id[:8]}.png"
                            plt.savefig(img_path)
                            plt.close()
                            # 画图

                            q25, q75 = np.percentile(time_interval, [25, 75])
                            iqr = q75 - q25
                            filtered = [x for x in time_interval if (q25 - 1.5 * iqr) <= x <= (q75 + 1.5 * iqr)]

                            msg_interval = int(round(np.percentile(filtered, 80)))
                            await self.update_one_field(person_id, "msg_interval", msg_interval)
                            logger.trace(f"用户{person_id}的msg_interval已经被更新为{msg_interval}")
                    except Exception as e:
                        logger.trace(f"用户{person_id}消息间隔计算失败: {type(e).__name__}: {str(e)}")
                        continue

                # 其他...

                if msg_interval_map:
                    logger.trace("已保存分布图到: logs/person_info")
                current_time = datetime.datetime.now()
                logger.trace(f"个人信息推断结束: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
                await asyncio.sleep(86400)

        except Exception as e:
            logger.error(f"个人信息推断运行时出错: {str(e)}")
            logger.exception("详细错误信息：")


person_info_manager = PersonInfoManager()
