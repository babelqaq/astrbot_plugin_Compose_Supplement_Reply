from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.api import logger
import asyncio
from collections import defaultdict
import time
from typing import Dict, List, Optional


@register("MergeChat", "babelqaq", "对用户的多条新消息进行整合并回复", "1.0.26")
class PrivateDebounceReply(Star):
    """私聊消息防抖合并插件"""

    def __init__(self, context: Context, config: dict = None):
        super().__init__(context)
        self.config = config or {}
        self.wait_time: float = self.config.get("wait_time", 6.0)
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
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[Cleanup] 清理会话出错: {e}")

    @filter.on_waiting_llm_request()
    async def on_waiting(self, event: AstrMessageEvent):
        session_id = event.session_id
        msg = event.message_str.strip()

        if not msg or msg.startswith("/"):
            return

        self.last_activity[session_id] = time.time()

        async with self.lock[session_id]:
            self.buffers[session_id].append(msg)
            logger.info(f"[Debounce] 会话 {session_id} 缓冲消息: {msg[:30]}...")

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

                merged_text = "\n".join(messages)
                self.buffers[session_id] = []
                await self._call_llm_and_reply(event, merged_text)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"[Debounce] 处理失败: {e}")
            async with self.lock[session_id]:
                self.buffers[session_id] = []

    async def _call_llm_and_reply(self, event: AstrMessageEvent, merged_text: str):
        """调用 LLM，失败时按 fallback_chat_models 列表顺序尝试回退"""
        try:
            primary_provider_id = await self.context.get_current_chat_provider_id(
                event.unified_msg_origin
            )
            fallback_models = await self._get_fallback_models()
            
            # 构建尝试顺序：主模型 + 回退模型（去重）
            providers_to_try = []
            if primary_provider_id:
                providers_to_try.append(primary_provider_id)
            for model_id in fallback_models:
                if model_id != primary_provider_id and model_id not in providers_to_try:
                    providers_to_try.append(model_id)
            
            if not providers_to_try:
                logger.warning("[LLM] 没有可用的模型提供商")
                await event.send(event.plain_result("模型服务暂时不可用，请稍后重试。"))
                return
            
            # 依次尝试
            last_error = None
            for provider_id in providers_to_try:
                try:
                    logger.info(f"[LLM] 尝试模型: {provider_id}")
                    
                    # 记录开始时间
                    start_time = time.time()
                    
                    llm_resp = await self.context.llm_generate(
                        chat_provider_id=provider_id,
                        prompt=merged_text,
                    )
                    
                    # 计算响应时长
                    elapsed_time = time.time() - start_time
                    
                    reply_text = llm_resp.completion_text if llm_resp else "（LLM 未返回有效回复）"
                    
                    # 后台输出：模型名称、响应时长、输出内容
                    logger.info(
                        f"[LLM] 响应成功 | 模型: {provider_id} | 耗时: {elapsed_time:.2f}s | "
                        f"输出: {reply_text[:100]}{'...' if len(reply_text) > 100 else ''}"
                    )
                    
                    await event.send(event.plain_result(reply_text))
                    await self._save_conversation(event, merged_text, reply_text)
                    return
                except Exception as e:
                    last_error = e
                    logger.error(f"[LLM] 模型 {provider_id} 调用失败: {type(e).__name__}: {e}")
                    continue
            
            # 所有模型都失败
            if last_error:
                logger.error(f"[LLM] 所有模型均调用失败，最后错误: {type(last_error).__name__}: {last_error}")
            await event.send(event.plain_result("模型服务暂时不可用，请稍后重试。"))

        except Exception as e:
            logger.error(f"[LLM] 处理失败: {type(e).__name__}: {e}")
            await event.send(event.plain_result("处理请求时发生错误，请稍后重试。"))

    async def _get_fallback_models(self) -> List[str]:
        """从 AstrBot 配置中获取 fallback_chat_models 列表"""
        try:
            config = self.context.get_config() if hasattr(self.context, 'get_config') else None
            if not config:
                return []
            
            # 尝试从 provider_settings 或根级别读取
            fallback = config.get('provider_settings', {}).get('fallback_chat_models', [])
            if not fallback:
                fallback = config.get('fallback_chat_models', [])
            
            return fallback if isinstance(fallback, list) else []
        except Exception as e:
            logger.debug(f"[LLM] 获取 fallback_chat_models 失败: {e}")
            return []

    async def _save_conversation(self, event: AstrMessageEvent, user_msg: str, assistant_msg: str):
        """保存对话历史"""
        try:
            from astrbot.core.agent.message import (
                AssistantMessageSegment,
                UserMessageSegment,
                TextPart,
            )
            conv_mgr = self.context.conversation_manager
            curr_cid = await conv_mgr.get_curr_conversation_id(event.unified_msg_origin)
            if curr_cid:
                user_segment = UserMessageSegment(content=[TextPart(text=user_msg)])
                assistant_segment = AssistantMessageSegment(content=[TextPart(text=assistant_msg)])
                await conv_mgr.add_message_pair(
                    cid=curr_cid,
                    user_message=user_segment,
                    assistant_message=assistant_segment,
                )
        except Exception:
            pass

    async def terminate(self):
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
