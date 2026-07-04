from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.api import logger
import asyncio
from collections import defaultdict
import time
from typing import Dict, List, Optional


@register("Compose Supplement Reply", "babelqaq", "对用户的多条新消息进行整合并回复", "1.0.21")
class PrivateDebounceReply(Star):
    """私聊消息防抖合并插件"""

    def __init__(self, context: Context, config: dict = None):
        super().__init__(context)
        # 从配置文件加载参数
        self.config = config or {}
        self.wait_time: float = self.config.get("wait_time", 6.0)
        self.min_wait_time: float = 3.0
        self.max_wait_time: float = 30.0
        self.cleanup_interval: int = self.config.get("cleanup_interval", 120)
        self.session_timeout: int = self.config.get("session_timeout", 600)
        
        self.buffers: Dict[str, List[str]] = defaultdict(list)
        self.tasks: Dict[str, asyncio.Task] = {}
        self.lock: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self.last_activity: Dict[str, float] = {}
        self._cleanup_task: Optional[asyncio.Task] = None

    async def initialize(self):
        logger.info(f"[PrivateDebounceReply] 初始化完成，等待时间: {self.wait_time}秒")
        self._cleanup_task = asyncio.create_task(self._cleanup_sessions())

    async def _cleanup_sessions(self):
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
        session_id = event.session_id
        msg = event.message_str.strip()

        if not msg or msg.startswith("/"):
            return

        self.last_activity[session_id] = time.time()

        async with self.lock[session_id]:
            self.buffers[session_id].append(msg)
            logger.info(f"[Debounce] 会话 {session_id} 缓冲消息: {msg[:30]}... (共{len(self.buffers[session_id])}条)")

            old_task = self.tasks.get(session_id)
            if old_task and not old_task.done():
                old_task.cancel()

            self.tasks[session_id] = asyncio.create_task(
                self._debounce(session_id, event)
            )
            event.stop_event()

    async def _debounce(self, session_id: str, event: AstrMessageEvent):
        try:
            await asyncio.sleep(self.wait_time)

            async with self.lock[session_id]:
                messages = self.buffers.get(session_id, [])
                if not messages:
                    return

                current_time = time.time()
                last_time = self.last_activity.get(session_id, current_time)
                if current_time - last_time < self.wait_time * 0.7:
                    self.tasks[session_id] = asyncio.create_task(
                        self._debounce(session_id, event)
                    )
                    return

                merged_text = self._merge_messages(messages)
                message_count = len(messages)
                
                # 简要日志
                logger.info(f"[Debounce] 合并 {message_count} 条消息: {merged_text[:50]}...")
                self.buffers[session_id] = []
                await self._call_llm_and_reply(event, merged_text, message_count)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"[Debounce] 处理失败: {e}")
            async with self.lock[session_id]:
                self.buffers[session_id] = []

    async def _call_llm_and_reply(self, event: AstrMessageEvent, merged_text: str, message_count: int):
        try:
            umo = event.unified_msg_origin
            provider_id = await self.context.get_current_chat_provider_id(umo=umo)
            
            if not provider_id:
                logger.warning("[LLM] 无法获取模型ID，使用降级回复")
                await event.send(event.plain_result(f"（合并了 {message_count} 条消息）{merged_text}"))
                return
            
            llm_resp = await self.context.llm_generate(
                chat_provider_id=provider_id,
                prompt=merged_text,
            )
            
            reply_text = llm_resp.completion_text if llm_resp else "（LLM 未返回有效回复）"
            await event.send(event.plain_result(reply_text))
            
            # 保存对话历史
            try:
                from astrbot.core.agent.message import (
                    AssistantMessageSegment,
                    UserMessageSegment,
                    TextPart,
                )
                conv_mgr = self.context.conversation_manager
                curr_cid = await conv_mgr.get_curr_conversation_id(event.unified_msg_origin)
                if curr_cid:
                    user_msg = UserMessageSegment(content=[TextPart(text=merged_text)])
                    assistant_msg = AssistantMessageSegment(content=[TextPart(text=reply_text)])
                    await conv_mgr.add_message_pair(
                        cid=curr_cid,
                        user_message=user_msg,
                        assistant_message=assistant_msg,
                    )
            except Exception:
                pass
            logger.info("[LLM] 回复已发送")

        except Exception as e:
            logger.error(f"[LLM] 调用失败: {e}")
            await event.send(event.plain_result(f"（合并了 {message_count} 条消息）{merged_text}"))

    def _merge_messages(self, messages: List[str]) -> str:
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
