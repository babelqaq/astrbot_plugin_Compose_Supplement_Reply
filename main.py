from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.api import logger
import asyncio
from collections import defaultdict
import time
from typing import Dict, List, Optional


@register("Compose Supplement Reply", "babelqaq", "对用户的多条新消息进行整合并回复", "1.0.18")
class PrivateDebounceReply(Star):
    """私聊消息防抖合并插件 - 优化版"""

    def __init__(self, context: Context):
        super().__init__(context)
        self.buffers: Dict[str, List[str]] = defaultdict(list)
        self.tasks: Dict[str, asyncio.Task] = {}
        self.lock: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self.last_activity: Dict[str, float] = {}
        
        # 【优化】调整默认参数，适应普通人输入速度
        self.wait_time: float = 20.0  # 防抖等待时间
        self.max_wait_time: float = 60.0  # 最大等待时间，防止无限等待
        self.cleanup_interval: int = 60
        self.session_timeout: int = 300
        
        self._cleanup_task: Optional[asyncio.Task] = None

    async def initialize(self):
        """插件初始化"""
        logger.info(f"[PrivateDebounceReply] 插件初始化完成，等待时间: {self.wait_time}秒")
        self._cleanup_task = asyncio.create_task(self._cleanup_sessions())

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
        """拦截消息进行防抖处理"""
        session_id = event.session_id
        msg = event.message_str.strip()

        # 忽略空消息和命令
        if not msg or msg.startswith("/"):
            return

        # 检查是否是合并后重新发送的消息
        if hasattr(event, 'metadata') and event.metadata and event.metadata.get('is_merged'):
            logger.info(f"[Debounce] 检测到合并消息，放行")
            return

        # 更新最后活动时间
        self.last_activity[session_id] = time.time()

        async with self.lock[session_id]:
            # 缓存消息
            self.buffers[session_id].append(msg)
            logger.info(f"[Debounce] 会话 {session_id} 缓冲消息: {msg[:30]}... (缓冲区: {len(self.buffers[session_id])} 条)")

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
            # 【关键】等待防抖时间
            await asyncio.sleep(self.wait_time)

            async with self.lock[session_id]:
                messages = self.buffers.get(session_id, [])
                if not messages:
                    return

                # 【优化】检查是否还有新消息（防抖重置检测）
                current_time = time.time()
                last_time = self.last_activity.get(session_id, current_time)
                time_since_last = current_time - last_time
                
                # 如果距离最后一条消息的时间小于防抖时间的 80%，说明用户还在输入，重新等待
                if time_since_last < self.wait_time * 0.8:
                    logger.debug(f"[Debounce] 会话 {session_id} 检测到新消息({time_since_last:.1f}s 前)，重新等待")
                    self.tasks[session_id] = asyncio.create_task(
                        self._debounce(session_id, event)
                    )
                    return

                # 【优化】如果消息数量较多，可以适当缩短等待时间
                # 但这里保持统一等待，确保用户体验一致
                
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
                logger.info(f"[Debounce] 等待时间: {self.wait_time}s")
                logger.info(f"[Debounce] =====================================")

                # 清空缓存
                self.buffers[session_id] = []

                # 创建带标记的 metadata
                if not hasattr(event, 'metadata') or event.metadata is None:
                    event.metadata = {}
                event.metadata['is_merged'] = True
                event.metadata['merged_count'] = message_count
                event.metadata['original_messages'] = messages

                # 发送合并后的消息
                await event.send(event.plain_result(merged_text))
                
                logger.info(f"[Debounce] 已发送合并消息")

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
            
            # 智能添加标点
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
        /debounce_config wait_time 3.5 - 设置等待时间（秒），建议 2.0-5.0
        """
        args = event.message_str.strip().split()
        
        if len(args) == 1:
            info = (
                f"📊 当前防抖配置\n"
                f"⏱️  等待时间: {self.wait_time} 秒\n"
                f"🧹 清理间隔: {self.cleanup_interval} 秒\n"
                f"⏰ 会话超时: {self.session_timeout} 秒\n\n"
                f"💡 提示: 等待时间建议设为 2.0-5.0 秒\n"
                f"   太短会过早合并，太长会影响体验"
            )
            yield event.plain_result(info)
            return
        
        if len(args) == 3 and args[1] == "wait_time":
            try:
                value = float(args[2])
                if value < 1.0:
                    yield event.plain_result("❌ 等待时间不能小于 1.0 秒")
                    return
                if value > 10.0:
                    yield event.plain_result("❌ 等待时间不能大于 10.0 秒")
                    return
                
                old_value = self.wait_time
                self.wait_time = value
                yield event.plain_result(f"✅ 等待时间已从 {old_value}s 调整为 {value}s")
                
                # 注意：这里没有持久化存储，如需持久化可以添加 KV 存储
            except ValueError:
                yield event.plain_result("❌ 请输入有效的数字")
            return
        
        yield event.plain_result("❌ 用法: /debounce_config wait_time [秒数]")

    async def terminate(self):
        """插件卸载时清理资源"""
        logger.info("PrivateDebounceReply 正在终止...")
        
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
        
        self.buffers.clear()
        self.tasks.clear()
        self.lock.clear()
        self.last_activity.clear()
        logger.info("PrivateDebounceReply 已终止")
