from typing import Optional
from .personality import Personality
from .identity import Identity


class Individuality:
    """个体特征管理类"""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self.personality: Optional[Personality] = None
        self.identity: Optional[Identity] = None

    @classmethod
    def get_instance(cls) -> "Individuality":
        """获取Individuality单例实例

        Returns:
            Individuality: 单例实例
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def initialize(
        self,
        bot_nickname: str,
        personality_core: str,
        personality_sides: list,
        identity_detail: list,
        height: int,
        weight: int,
        age: int,
        gender: str,
        appearance: str,
    ) -> None:
        """初始化个体特征

        Args:
            bot_nickname: 机器人昵称
            personality_core: 人格核心特点
            personality_sides: 人格侧面描述
            identity_detail: 身份细节描述
            height: 身高（厘米）
            weight: 体重（千克）
            age: 年龄
            gender: 性别
            appearance: 外貌特征
        """
        # 初始化人格
        self.personality = Personality.initialize(
            bot_nickname=bot_nickname, personality_core=personality_core, personality_sides=personality_sides
        )

        # 初始化身份
        self.identity = Identity.initialize(
            identity_detail=identity_detail, height=height, weight=weight, age=age, gender=gender, appearance=appearance
        )

    def to_dict(self) -> dict:
        """将个体特征转换为字典格式"""
        return {
            "personality": self.personality.to_dict() if self.personality else None,
            "identity": self.identity.to_dict() if self.identity else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Individuality":
        """从字典创建个体特征实例"""
        instance = cls.get_instance()
        if data.get("personality"):
            instance.personality = Personality.from_dict(data["personality"])
        if data.get("identity"):
            instance.identity = Identity.from_dict(data["identity"])
        return instance

    def get_prompt(self, type, x_person, level):
        """
        获取个体特征的prompt
        """
        if type == "personality":
            return self.personality.get_prompt(x_person, level)
        elif type == "identity":
            return self.identity.get_prompt(x_person, level)
        else:
            return ""

    def get_traits(self, factor):
        """
        获取个体特征的特质
        """
        if factor == "openness":
            return self.personality.openness
        elif factor == "conscientiousness":
            return self.personality.conscientiousness
        elif factor == "extraversion":
            return self.personality.extraversion
        elif factor == "agreeableness":
            return self.personality.agreeableness
        elif factor == "neuroticism":
            return self.personality.neuroticism
