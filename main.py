from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.api import logger
import asyncio
from collections import defaultdict
import time
from typing import Dict, List, Optional


@register("Compose Supplement Reply", "babelqaq", "对用户的多条新消息进行整合并回复", "1.0.16")
class PrivateDebounceReply(Star):
    """私聊消息防抖合并插件"""

    def __init__(self, context: Context):
        super().__init__(context)
        self.buffers: Dict[str, List[str]] = defaultdict(list)
        self.tasks: Dict[str, asyncio.Task] = {}
        self.lock: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self.last_activity: Dict[str, float] = {}
        
        # 可配置参数
        self.wait_time: float = 1.5
        self.cleanup_interval: int = 60
        self.session_timeout: int = 300
        
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
                logger.info(f"[Config] 加载配置: wait_time={self.wait_time}s")
            else:
                await self._save_config()
            self._config_loaded = True
        except Exception as e:
            logger.warning(f"[Config] 加载配置失败: {e}")

    async def _save_config(self):
        """保存配置到 KV 存储"""
        try:
            config = {
                "wait_time": self.wait_time,
                "cleanup_interval": self.cleanup_interval,
                "session_timeout": self.session_timeout
            }
            await self.put_kv_data("debounce_config", config)
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
                        self.buffers.pop(session_id, None)
                        task = self.tasks.pop(session_id, None)
                        if task and not task.done():
                            task.cancel()
                            try:
                                await task
                            except asyncio.CancelledError:
                                pass
                        self.last_activity.pop(session_id, None)
                        self.lock.pop(session_id, None)
                        logger.info(f"[Cleanup] 清理会话: {session_id}")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[Cleanup] 清理出错: {e}")

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

        # 【关键修复】检查是否是合并后重新发送的消息
        # 如果是，则放行让它正常进入 LLM
        if hasattr(event, 'metadata') and event.metadata and event.metadata.get('is_merged'):
            logger.info(f"[Debounce] 检测到合并消息，放行: {msg[:30]}...")
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

            # 创建新任务
            self.tasks[session_id] = asyncio.create_task(
                self._debounce(session_id, event)
            )

            # 阻止当前消息进入 LLM
            event.stop_event()

    async def _debounce(self, session_id: str, event: AstrMessageEvent):
        """防抖核心逻辑"""
        try:
            await asyncio.sleep(self.wait_time)

            async with self.lock[session_id]:
                messages = self.buffers.get(session_id, [])
                if not messages:
                    return

                # 检查是否还有新消息
                current_time = time.time()
                last_time = self.last_activity.get(session_id, current_time)
                if current_time - last_time < self.wait_time * 0.8:
                    logger.debug(f"[Debounce] 会话 {session_id} 有新消息，重新等待")
                    self.tasks[session_id] = asyncio.create_task(
                        self._debounce(session_id, event)
                    )
                    return

                # 合并消息
                merged_text = self._merge_messages(messages)
                message_count = len(messages)
                
                # 后台详细日志
                logger.info(f"[Debounce] ========== 合并消息详情 ==========")
                logger.info(f"[Debounce] 会话ID: {session_id}")
                logger.info(f"[Debounce] 原始消息数: {message_count}")
                for idx, msg in enumerate(messages, 1):
                    logger.info(f"[Debounce]   {idx}. {msg}")
                logger.info(f"[Debounce] 合并后的消息:")
                logger.info(f"[Debounce] {merged_text}")
                logger.info(f"[Debounce] =====================================")

                # 清空缓存
                self.buffers[session_id] = []

                # 【关键修复】创建带标记的 metadata
                if not hasattr(event, 'metadata') or event.metadata is None:
                    event.metadata = {}
                event.metadata['is_merged'] = True
                event.metadata['merged_count'] = message_count
                event.metadata['original_messages'] = messages

                # 发送合并后的消息（带标记）
                await event.send(event.plain_result(merged_text))
                
                logger.info(f"[Debounce] 已发送合并消息（带 merged 标记）")

        except asyncio.CancelledError:
            logger.debug(f"[Debounce] 会话 {session_id} 任务取消")
        except Exception as e:
            logger.error(f"[Debounce] 会话 {session_id} 处理失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
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
            if i > 0 and merged:
                last_char = merged[-1][-1] if merged[-1] else ""
                if last_char not in "。.!！?？；;：:”\"'’":
                    merged[-1] = merged[-1] + "。"
            merged.append(msg)
        return "\n".join(merged)

    @filter.command("debounce_config")
    async def config_command(self, event: AstrMessageEvent):
        """配置管理指令"""
        args = event.message_str.strip().split()
        
        if len(args) == 1:
            info = (
                f"📊 当前防抖配置\n"
                f"⏱️  等待时间: {self.wait_time}s\n"
                f"🧹 清理间隔: {self.cleanup_interval}s\n"
                f"⏰ 会话超时: {self.session_timeout}s"
            )
            yield event.plain_result(info)
            return
        
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
                    yield event.plain_result(f"❌ 未知参数: {key}")
                    return
                
                await self._save_config()
                yield event.plain_result(f"✅ {key} 已从 {old_value} 更新为 {value}")
            except ValueError:
                yield event.plain_result("❌ 请输入有效的数字")
            return
        
        yield event.plain_result("❌ 用法: /debounce_config [key] [value]")

    async def terminate(self):
        """插件卸载时清理资源"""
        logger.info("Private Debounce Reply 正在终止...")
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        for task in self.tasks.values():
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        await self._save_config()
        self.buffers.clear()
        self.tasks.clear()
        self.lock.clear()
        self.last_activity.clear()
        logger.info("Private Debounce Reply 已终止")
