from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.api import logger
import asyncio
from collections import defaultdict
import time
from typing import Dict, List, Optional


@register("Compose Supplement Reply", "babelqaq", "对用户的多条新消息进行整合并回复", "1.0.20")
class PrivateDebounceReply(Star):
    """私聊消息防抖合并插件 - 直接调用 LLM 版"""

    def __init__(self, context: Context):
        super().__init__(context)
        self.buffers: Dict[str, List[str]] = defaultdict(list)
        self.tasks: Dict[str, asyncio.Task] = {}
        self.lock: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self.last_activity: Dict[str, float] = {}
        
        # 核心参数
        self.wait_time: float = 10.0
        self.min_wait_time: float = 3.0
        self.max_wait_time: float = 30.0
        self.cleanup_interval: int = 120
        self.session_timeout: int = 600
        
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

        # 更新最后活动时间
        self.last_activity[session_id] = time.time()

        async with self.lock[session_id]:
            self.buffers[session_id].append(msg)
            buffer_count = len(self.buffers[session_id])
            logger.info(f"[Debounce] 会话 {session_id} 缓冲消息 #{buffer_count}: {msg[:30]}...")

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
            logger.debug(f"[Debounce] 会话 {session_id} 开始等待 {self.wait_time}s")
            await asyncio.sleep(self.wait_time)

            async with self.lock[session_id]:
                messages = self.buffers.get(session_id, [])
                if not messages:
                    return

                # 检查是否还有新消息
                current_time = time.time()
                last_time = self.last_activity.get(session_id, current_time)
                time_since_last = current_time - last_time
                
                if time_since_last < self.wait_time * 0.7:
                    logger.debug(f"[Debounce] 会话 {session_id} 检测到新消息，重新等待")
                    self.tasks[session_id] = asyncio.create_task(
                        self._debounce(session_id, event)
                    )
                    return

                # 合并消息
                merged_text = self._merge_messages(messages)
                message_count = len(messages)
                
                # 后台详细日志
                logger.info(f"[Debounce] ========================================")
                logger.info(f"[Debounce] 会话ID: {session_id}")
                logger.info(f"[Debounce] 合并消息数: {message_count}")
                logger.info(f"[Debounce] 等待时间: {self.wait_time}s")
                for idx, msg in enumerate(messages, 1):
                    logger.info(f"[Debounce]   {idx}. {msg}")
                logger.info(f"[Debounce] 合并结果: {merged_text}")
                logger.info(f"[Debounce] ========================================")

                # 清空缓存
                self.buffers[session_id] = []

                # 【关键修改】直接调用 LLM 生成回复，而不是重新发送消息
                await self._call_llm_and_reply(event, merged_text, message_count)

        except asyncio.CancelledError:
            logger.debug(f"[Debounce] 会话 {session_id} 任务取消")
        except Exception as e:
            logger.error(f"[Debounce] 会话 {session_id} 处理失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            async with self.lock[session_id]:
                self.buffers[session_id] = []

    async def _call_llm_and_reply(self, event: AstrMessageEvent, merged_text: str, message_count: int):
        """直接调用 LLM 并回复"""
        try:
            # 1. 获取当前会话使用的聊天模型 ID
            umo = event.unified_msg_origin
            provider_id = await self.context.get_current_chat_provider_id(umo=umo)
            
            if not provider_id:
                logger.warning(f"[LLM] 无法获取聊天模型 ID，使用默认回复")
                await event.send(event.plain_result(f"（合并了 {message_count} 条消息）{merged_text}"))
                return
            
            logger.info(f"[LLM] 使用模型: {provider_id}")
            logger.info(f"[LLM] 发送给 LLM: {merged_text[:100]}...")
            
            # 2. 调用 LLM 生成回复
            llm_resp = await self.context.llm_generate(
                chat_provider_id=provider_id,
                prompt=merged_text,
            )
            
            # 3. 获取 LLM 回复文本
            reply_text = llm_resp.completion_text if llm_resp else "（LLM 未返回有效回复）"
            
            logger.info(f"[LLM] LLM 回复: {reply_text[:100]}...")
            
            # 4. 发送回复给用户
            await event.send(event.plain_result(reply_text))
            
            # 5. 【可选】将对话记录添加到会话历史中
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
                    logger.debug(f"[LLM] 已保存对话历史")
            except Exception as e:
                logger.debug(f"[LLM] 保存对话历史失败（非致命）: {e}")
            
            logger.info(f"[LLM] ✅ 已发送 LLM 回复")
            
        except Exception as e:
            logger.error(f"[LLM] 调用 LLM 失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # 降级方案：直接返回合并后的消息
            await event.send(event.plain_result(f"（合并了 {message_count} 条消息）{merged_text}"))

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
                f"📊 **当前防抖配置**\n\n"
                f"⏱️  等待时间: `{self.wait_time}` 秒\n"
                f"⏰ 会话超时: `{self.session_timeout}` 秒 ({self.session_timeout//60} 分钟)\n"
                f"🧹 清理间隔: `{self.cleanup_interval}` 秒\n\n"
                f"💡 修改: `/debounce_config wait_time [秒数]`"
            )
            yield event.plain_result(info)
            return
        
        if len(args) == 3 and args[1] == "wait_time":
            try:
                value = float(args[2])
                if value < self.min_wait_time:
                    yield event.plain_result(f"❌ 等待时间不能小于 {self.min_wait_time} 秒")
                    return
                if value > self.max_wait_time:
                    yield event.plain_result(f"❌ 等待时间不能大于 {self.max_wait_time} 秒")
                    return
                
                old_value = self.wait_time
                self.wait_time = value
                self.session_timeout = max(300, int(value * 60))
                self.cleanup_interval = max(60, int(value * 12))
                
                yield event.plain_result(f"✅ 等待时间: {old_value}s → {value}s")
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
