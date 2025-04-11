import re
import ast
from typing import Dict, Any, Optional, List, Union


class PromptManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._prompts = {}
            cls._instance._counter = 0
        return cls._instance

    def generate_name(self, template: str) -> str:
        """为未命名的prompt生成名称"""
        self._counter += 1
        return f"prompt_{self._counter}"

    def register(self, prompt: "Prompt") -> None:
        """注册一个prompt"""
        if not prompt.name:
            prompt.name = self.generate_name(prompt.template)
        self._prompts[prompt.name] = prompt

    def add_prompt(self, name: str, fstr: str) -> "Prompt":
        prompt = Prompt(fstr, name=name)
        self._prompts[prompt.name] = prompt
        return prompt

    def get_prompt(self, name: str) -> "Prompt":
        if name not in self._prompts:
            raise KeyError(f"Prompt '{name}' not found")
        return self._prompts[name]

    def format_prompt(self, name: str, **kwargs) -> str:
        prompt = self.get_prompt(name)
        return prompt.format(**kwargs)


# 全局单例
global_prompt_manager = PromptManager()


class Prompt(str):
    def __new__(cls, fstr: str, name: Optional[str] = None, args: Union[List[Any], tuple[Any, ...]] = None, **kwargs):
        # 如果传入的是元组，转换为列表
        if isinstance(args, tuple):
            args = list(args)

        # 解析模板
        tree = ast.parse(f"f'''{fstr}'''", mode="eval")
        template_args = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.FormattedValue):
                expr = ast.get_source_segment(fstr, node.value)
                if expr:
                    template_args.add(expr)

        # 如果提供了初始参数，立即格式化
        if kwargs or args:
            formatted = cls._format_template(fstr, args=args, kwargs=kwargs)
            obj = super().__new__(cls, formatted)
        else:
            obj = super().__new__(cls, "")

        obj.template = fstr
        obj.name = name
        obj.args = template_args
        obj._args = args or []
        obj._kwargs = kwargs

        # 自动注册到全局管理器
        global_prompt_manager.register(obj)
        return obj

    @classmethod
    def _format_template(cls, template: str, args: List[Any] = None, kwargs: Dict[str, Any] = None) -> str:
        fmt_str = f"f'''{template}'''"
        tree = ast.parse(fmt_str, mode="eval")
        template_args = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FormattedValue):
                expr = ast.get_source_segment(fmt_str, node.value)
                if expr and expr not in template_args:
                    template_args.append(expr)
        formatted_args = {}
        formatted_kwargs = {}

        # 处理位置参数
        if args:
            for i in range(len(args)):
                arg = args[i]
                if isinstance(arg, Prompt):
                    formatted_args[template_args[i]] = arg.format(**kwargs)
                else:
                    formatted_args[template_args[i]] = arg

        # 处理关键字参数
        if kwargs:
            for key, value in kwargs.items():
                if isinstance(value, Prompt):
                    remaining_kwargs = {k: v for k, v in kwargs.items() if k != key}
                    formatted_kwargs[key] = value.format(**remaining_kwargs)
                else:
                    formatted_kwargs[key] = value

        try:
            # 先用位置参数格式化

            if args:
                template = template.format(**formatted_args)
            # 再用关键字参数格式化
            if kwargs:
                template = template.format(**formatted_kwargs)
            return template
        except (IndexError, KeyError) as e:
            raise ValueError(f"格式化模板失败: {template}, args={formatted_args}, kwargs={formatted_kwargs}") from e

    def format(self, *args, **kwargs) -> "Prompt":
        """支持位置参数和关键字参数的格式化，使用"""
        ret = type(self)(
            self.template, self.name, args=list(args) if args else self._args, **kwargs if kwargs else self._kwargs
        )
        # print(f"prompt build result: {ret}  name: {ret.name} ")
        return ret

    def __str__(self) -> str:
        if self._kwargs or self._args:
            return super().__str__()
        return self.template

    def __repr__(self) -> str:
        return f"Prompt(template='{self.template}', name='{self.name}')"
