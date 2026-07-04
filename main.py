from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.api import logger
import asyncio
from collections import defaultdict
import time
from typing import Dict, List, Optional


@register("Compose Supplement Reply", "babelqaq", "对用户的多条新消息进行整合并回复", "1.0.17")
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

    async def initialize(self):
        """插件初始化"""
        logger.info("[PrivateDebounceReply] 插件初始化完成")
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
        # 详细日志，确认钩子被触发
        logger.info(f"[Debounce] on_waiting 被触发，session: {event.session_id}, msg: {event.message_str[:30] if event.message_str else '空'}...")
        
        session_id = event.session_id
        msg = event.message_str.strip()

        # 忽略空消息和命令
        if not msg or msg.startswith("/"):
            logger.debug(f"[Debounce] 忽略空消息或命令: {msg}")
            return

        # 【重要】检查是否是合并后重新发送的消息
        if hasattr(event, 'metadata') and event.metadata and event.metadata.get('is_merged'):
            logger.info(f"[Debounce] 检测到合并消息（带标记），放行: {msg[:30]}...")
            return

        # 更新最后活动时间
        self.last_activity[session_id] = time.time()

        async with self.lock[session_id]:
            # 缓存消息
            self.buffers[session_id].append(msg)
            logger.info(f"[Debounce] 会话 {session_id} 缓冲消息: {msg[:30]}... (当前缓冲区: {len(self.buffers[session_id])} 条)")

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
            logger.info(f"[Debounce] 已阻止消息进入 LLM: {msg[:30]}...")

    async def _debounce(self, session_id: str, event: AstrMessageEvent):
        """防抖核心逻辑"""
        try:
            logger.debug(f"[Debounce] 开始防抖等待 {self.wait_time}s, session: {session_id}")
            await asyncio.sleep(self.wait_time)

            async with self.lock[session_id]:
                messages = self.buffers.get(session_id, [])
                if not messages:
                    logger.debug(f"[Debounce] 会话 {session_id} 缓冲区为空，跳过")
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

                # 创建带标记的 metadata
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
