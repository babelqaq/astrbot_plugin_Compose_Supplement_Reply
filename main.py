from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.api import logger
import asyncio
from collections import defaultdict
import time
from typing import Dict, List, Optional


@register("Compose Supplement Reply", "babelqaq", "对用户的多条新消息进行整合并回复", "1.0.15")
class PrivateDebounceReply(Star):
    """私聊消息防抖合并插件
    
    功能：
    1. 在私聊中合并用户短时间内发送的多条消息
    2. 合并后统一发送给 LLM 处理
    3. 支持通过指令动态调整防抖参数
    """

    def __init__(self, context: Context):
        super().__init__(context)
        # 缓存数据结构
        self.buffers: Dict[str, List[str]] = defaultdict(list)  # 消息缓冲区
        self.tasks: Dict[str, asyncio.Task] = {}  # 防抖任务
        self.lock: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)  # 会话锁
        self.last_activity: Dict[str, float] = {}  # 最后活动时间
        
        # 可配置参数（支持运行时修改）
        self.wait_time: float = 1.5  # 防抖等待时间（秒）
        self.cleanup_interval: int = 60  # 清理过期会话间隔（秒）
        self.session_timeout: int = 300  # 会话超时时间（秒）
        
        # 内部状态
        self._cleanup_task: Optional[asyncio.Task] = None
        self._config_loaded: bool = False

    async def initialize(self):
        """插件初始化"""
        logger.info("Private Debounce Reply initialized")
        await self._load_config()
        self._cleanup_task = asyncio.create_task(self._cleanup_sessions())

    async def _load_config(self):
        """从 KV 存储加载配置"""
        try:
            config = await self.get_kv_data("debounce_config", {})
            if config:
                self.wait_time = config.get("wait_time", self.wait_time)
                self.cleanup_interval = config.get("cleanup_interval", self.cleanup_interval)
                self.session_timeout = config.get("session_timeout", self.session_timeout)
                logger.info(f"[Config] 加载配置成功: wait_time={self.wait_time}s, "
                           f"cleanup_interval={self.cleanup_interval}s, "
                           f"session_timeout={self.session_timeout}s")
            else:
                await self._save_config()
            self._config_loaded = True
        except Exception as e:
            logger.warning(f"[Config] 加载配置失败，使用默认值: {e}")

    async def _save_config(self):
        """保存配置到 KV 存储"""
        try:
            config = {
                "wait_time": self.wait_time,
                "cleanup_interval": self.cleanup_interval,
                "session_timeout": self.session_timeout
            }
            await self.put_kv_data("debounce_config", config)
            logger.debug("[Config] 配置已保存")
        except Exception as e:
            logger.error(f"[Config] 保存配置失败: {e}")

    async def _cleanup_sessions(self):
        """定期清理过期会话"""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                current_time = time.time()
                expired = [
                    sid for sid, last_time in self.last_activity.items()
                    if current_time - last_time > self.session_timeout
                ]
                
                for session_id in expired:
                    async with self.lock[session_id]:
                        # 清理缓冲区
                        self.buffers.pop(session_id, None)
                        # 取消并清理任务
                        task = self.tasks.pop(session_id, None)
                        if task and not task.done():
                            task.cancel()
                            try:
                                await task
                            except asyncio.CancelledError:
                                pass
                        # 清理其他数据
                        self.last_activity.pop(session_id, None)
                        self.lock.pop(session_id, None)
                        logger.info(f"[Cleanup] 清理过期会话: {session_id}")
            except asyncio.CancelledError:
                logger.debug("[Cleanup] 清理任务已取消")
                break
            except Exception as e:
                logger.error(f"[Cleanup] 清理会话出错: {e}")

    @filter.on_waiting_llm_request()
    async def on_waiting(self, event: AstrMessageEvent):
        """拦截私聊消息进行防抖处理"""
        # 只处理私聊消息
        if not hasattr(event, 'is_private') or not event.is_private:
            return
            
        session_id = event.session_id
        msg = event.message_str.strip()

        # 忽略空消息和命令
        if not msg or msg.startswith("/"):
            return

        # 更新最后活动时间
        self.last_activity[session_id] = time.time()

        async with self.lock[session_id]:
            # 缓存消息
            self.buffers[session_id].append(msg)
            logger.info(f"[Debounce] 会话 {session_id} 缓冲消息: {msg[:30]}...")

            # 取消旧任务
            old_task = self.tasks.get(session_id)
            if old_task and not old_task.done():
                old_task.cancel()
                logger.debug(f"[Debounce] 取消会话 {session_id} 的旧任务")

            # 创建新任务（传递 session_id 和 event 的副本）
            self.tasks[session_id] = asyncio.create_task(
                self._debounce(session_id, event)
            )

            # 阻止当前消息进入 LLM
            event.stop_event()

    async def _debounce(self, session_id: str, event: AstrMessageEvent):
        """防抖核心逻辑"""
        try:
            # 1. 等待防抖时间
            await asyncio.sleep(self.wait_time)

            async with self.lock[session_id]:
                # 2. 获取缓存的消息
                messages = self.buffers.get(session_id, [])
                if not messages:
                    return

                # 3. 检查是否还有新消息（防抖重置检测）
                current_time = time.time()
                last_time = self.last_activity.get(session_id, current_time)
                if current_time - last_time < self.wait_time * 0.8:
                    logger.debug(f"[Debounce] 会话 {session_id} 检测到新消息，重新等待")
                    self.tasks[session_id] = asyncio.create_task(
                        self._debounce(session_id, event)
                    )
                    return

                # 4. 合并消息
                merged_text = self._merge_messages(messages)
                message_count = len(messages)
                
                # 5. 详细日志（便于调试）
                logger.info(f"[Debounce] ========== 合并消息详情 ==========")
                logger.info(f"[Debounce] 会话ID: {session_id}")
                logger.info(f"[Debounce] 消息数: {message_count}")
                for idx, msg in enumerate(messages, 1):
                    logger.info(f"[Debounce]   {idx}. {msg}")
                logger.info(f"[Debounce] 合并后: {merged_text}")
                logger.info(f"[Debounce] =====================================")

                # 6. 清空缓存
                self.buffers[session_id] = []

                # 7. 使用 plain_result 发送消息（最可靠的方式，无需导入 MessageChain）
                await event.send(event.plain_result(merged_text))
                
                logger.info(f"[Debounce] 已发送合并消息到 LLM")

        except asyncio.CancelledError:
            logger.debug(f"[Debounce] 会话 {session_id} 任务已取消")
        except Exception as e:
            logger.error(f"[Debounce] 会话 {session_id} 处理失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # 出错时清理缓存，避免内存泄漏
            async with self.lock[session_id]:
                self.buffers[session_id] = []

    def _merge_messages(self, messages: List[str]) -> str:
        """智能合并多条消息"""
        if not messages:
            return ""
        if len(messages) == 1:
            return messages[0]
        
        merged = []
        for i, msg in enumerate(messages):
            msg = msg.strip()
            if not msg:
                continue
            
            # 智能添加标点：如果前一条消息没有结束标点，自动添加句号
            if i > 0 and merged:
                last_char = merged[-1][-1] if merged[-1] else ""
                if last_char not in "。.!！?？；;：:”\"'’":
                    merged[-1] = merged[-1] + "。"
            
            merged.append(msg)
        
        return "\n".join(merged)

    @filter.command("debounce_config")
    async def config_command(self, event: AstrMessageEvent):
        """配置管理指令
        
        用法：
        /debounce_config              - 查看当前配置
        /debounce_config wait_time 2.0 - 设置等待时间（秒）
        /debounce_config cleanup_interval 120 - 设置清理间隔（秒）
        /debounce_config session_timeout 600 - 设置会话超时（秒）
        """
        args = event.message_str.strip().split()
        
        # 显示当前配置
        if len(args) == 1:
            info = (
                f"📊 **当前防抖配置**\n"
                f"⏱️  等待时间: `{self.wait_time}s`\n"
                f"🧹 清理间隔: `{self.cleanup_interval}s`\n"
                f"⏰ 会话超时: `{self.session_timeout}s`\n\n"
                f"💡 修改方式: `/debounce_config [参数名] [数值]`"
            )
            yield event.plain_result(info)
            return
        
        # 修改配置
        if len(args) == 3:
            key, val = args[1], args[2]
            try:
                value = float(val)
                old_value = None
                
                if key == "wait_time":
                    if value < 0.3:
                        yield event.plain_result("❌ 等待时间不能小于 0.3 秒")
                        return
                    old_value = self.wait_time
                    self.wait_time = value
                elif key == "cleanup_interval":
                    if value < 10:
                        yield event.plain_result("❌ 清理间隔不能小于 10 秒")
                        return
                    old_value = self.cleanup_interval
                    self.cleanup_interval = int(value)
                elif key == "session_timeout":
                    if value < 30:
                        yield event.plain_result("❌ 会话超时不能小于 30 秒")
                        return
                    old_value = self.session_timeout
                    self.session_timeout = int(value)
                else:
                    yield event.plain_result(f"❌ 未知参数: {key}，可选: wait_time, cleanup_interval, session_timeout")
                    return
                
                # 保存配置
                await self._save_config()
                yield event.plain_result(f"✅ `{key}` 已从 `{old_value}` 更新为 `{value}`")
                
            except ValueError:
                yield event.plain_result("❌ 请输入有效的数字")
            return
        
        yield event.plain_result("❌ 用法错误，请查看 `/debounce_config` 帮助")

    async def terminate(self):
        """插件卸载时清理资源"""
        logger.info("Private Debounce Reply 正在终止...")
        
        # 取消清理任务
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # 取消所有防抖任务
        for session_id, task in list(self.tasks.items()):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                logger.debug(f"[Terminate] 取消会话 {session_id} 的任务")
        
        # 保存当前配置
        await self._save_config()
        
        # 清理所有缓存
        self.buffers.clear()
        self.tasks.clear()
        self.lock.clear()
        self.last_activity.clear()
        
        logger.info("Private Debounce Reply 已终止")
