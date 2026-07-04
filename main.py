from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.api import logger
import asyncio
from collections import defaultdict
import time
from typing import Dict, List, Optional


@register("Compose Supplement Reply", "babelqaq", "对用户的多条新消息进行整合并回复", "1.0.23")
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

                merged_text = self._merge_messages(messages)
                self.buffers[session_id] = []
                await self._call_llm_and_reply(event, merged_text)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"[Debounce] 处理失败: {e}")
            async with self.lock[session_id]:
                self.buffers[session_id] = []

    async def _call_llm_and_reply(self, event: AstrMessageEvent, merged_text: str):
        """调用 LLM，失败时自动尝试配置文件中的其他模型"""
        try:
            # 1. 获取当前会话默认模型
            primary_provider_id = await self.context.get_current_chat_provider_id(
                event.unified_msg_origin
            )
            
            # 2. 从 AstrBot 配置中获取所有已配置的模型提供商 ID 列表
            all_provider_ids = await self._get_configured_provider_ids()
            
            # 3. 构建尝试顺序：主模型优先，然后去重排列其他模型
            providers_to_try = []
            if primary_provider_id:
                providers_to_try.append(primary_provider_id)
            for pid in all_provider_ids:
                if pid != primary_provider_id and pid not in providers_to_try:
                    providers_to_try.append(pid)
            
            if not providers_to_try:
                logger.warning("[LLM] 没有可用的模型提供商")
                await event.send(event.plain_result(merged_text))
                return
            
            # 4. 依次尝试调用
            last_error = None
            for provider_id in providers_to_try:
                try:
                    logger.info(f"[LLM] 尝试模型: {provider_id}")
                    llm_resp = await self.context.llm_generate(
                        chat_provider_id=provider_id,
                        prompt=merged_text,
                    )
                    reply_text = llm_resp.completion_text if llm_resp else "（LLM 未返回有效回复）"
                    await event.send(event.plain_result(reply_text))
                    await self._save_conversation(event, merged_text, reply_text)
                    return  # 成功则退出
                except Exception as e:
                    last_error = e
                    logger.warning(f"[LLM] 模型 {provider_id} 调用失败: {e}")
                    continue
            
            # 所有模型都失败
            error_msg = str(last_error) if last_error else "未知错误"
            logger.error(f"[LLM] 所有可用模型均调用失败: {error_msg}")
            await event.send(event.plain_result(
                f"⚠️ 模型服务暂时不可用（{error_msg}），请稍后重试。"
            ))

        except Exception as e:
            logger.error(f"[LLM] 处理失败: {e}")
            await event.send(event.plain_result(merged_text))

    async def _get_configured_provider_ids(self) -> List[str]:
        """从 AstrBot 配置中获取所有已配置的模型提供商 ID 列表"""
        try:
            # 获取 AstrBot 核心配置
            config = self.context.get_config() if hasattr(self.context, 'get_config') else None
            if not config:
                logger.debug("[LLM] 无法获取 AstrBot 配置")
                return []
            
            # 根据文档，配置中的 'provider' 列表包含了所有提供商配置
            providers_config = config.get('provider', [])
            if not providers_config:
                logger.debug("[LLM] 配置中未找到 'provider' 列表")
                return []
            
            # 提取每个提供商的 ID
            provider_ids = []
            for p in providers_config:
                if isinstance(p, dict):
                    # 提供商配置可能是字典，包含 'id' 或 'provider_id' 字段
                    pid = p.get('id') or p.get('provider_id')
                    if pid:
                        provider_ids.append(pid)
                elif hasattr(p, 'id'):
                    provider_ids.append(p.id)
            
            logger.debug(f"[LLM] 从配置中获取到 {len(provider_ids)} 个提供商: {provider_ids}")
            return provider_ids
            
        except Exception as e:
            logger.debug(f"[LLM] 获取配置提供商列表失败: {e}")
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
            pass  # 保存历史失败不影响主流程

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
