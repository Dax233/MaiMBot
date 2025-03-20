from nonebot import get_app
from .api import router
from src.common.logger import get_module_logger

# 获取主应用实例并挂载路由
app = get_app()
app.include_router(router, prefix="/api")

# 打印日志，方便确认API已注册
logger = get_module_logger("cfg_reload")
logger.success("配置重载API已注册，可通过 /api/reload-config 访问")
