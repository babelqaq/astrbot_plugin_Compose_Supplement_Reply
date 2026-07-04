from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.api import logger
import asyncio
from collections import defaultdict
import time


@register("Compose Supplement Reply", "babelqaq", "对用户的多条新消息进行整合并回复", "1.0.3")
class PrivateDebounceReply(Star):

    def __init__(self, context: Context):
        super().__init__(context)
        self.buffers = defaultdict(list)  # 缓存消息
        self.tasks = {}  # 防抖任务
        self.lock = defaultdict(asyncio.Lock)  # 会话锁
        self.last_activity = {}  # 记录最后活动时间
        self.wait_time = 1.5  # 防抖等待时间（秒）
        self.max_wait_time = 5.0  # 最大等待时间（秒）
        self.cleanup_interval = 60  # 清理过期会话间隔（秒）
        self.session_timeout = 300  # 会话超时时间（秒）
        self._cleanup_task = None

    async def initialize(self):
        """初始化插件"""
        logger.info("Private Debounce Reply initialized")
        # 启动清理任务
        self._cleanup_task = asyncio.create_task(self._cleanup_sessions())

    async def _cleanup_sessions(self):
        """定期清理过期会话"""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                current_time = time.time()
                expired_sessions = []
                
                for session_id, last_time in self.last_activity.items():
                    if current_time - last_time > self.session_timeout:
                        expired_sessions.append(session_id)
                
                for session_id in expired_sessions:
                    async with self.lock[session_id]:
                        if session_id in self.buffers:
                            del self.buffers[session_id]
                        if session_id in self.tasks:
                            task = self.tasks[session_id]
                            if task and not task.done():
                                task.cancel()
                            del self.tasks[session_id]
                        if session_id in self.last_activity:
                            del self.last_activity[session_id]
                        if session_id in self.lock:
                            del self.lock[session_id]
                        logger.info(f"[Cleanup] 清理过期会话: {session_id}")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[Cleanup] 清理会话出错: {e}")

    # 拦截所有消息
    @filter.on_waiting_llm_request()
    async def on_waiting(self, event: AstrMessageEvent):
        """拦截消息并进行防抖处理"""
        session_id = event.session_id
        msg = event.message_str.strip()

        # 忽略空消息和命令
        if not msg or msg.startswith("/"):
            return

        # 更新最后活动时间
        self.last_activity[session_id] = time.time()

        async with self.lock[session_id]:
            # 追加消息到缓冲区
            self.buffers[session_id].append(msg)
            logger.info(f"[Debounce] 会话 {session_id} 添加消息: {msg[:50]}...")

            # 取消现有的防抖任务
            task = self.tasks.get(session_id)
            if task and not task.done():
                task.cancel()
                logger.debug(f"[Debounce] 取消会话 {session_id} 的旧任务")

            # 创建新的防抖任务
            self.tasks[session_id] = asyncio.create_task(
                self._debounce(session_id, event)
            )

            # 阻止当前消息进入LLM
            event.cancel()

    # debounce核心处理
    async def _debounce(self, session_id, event):
        """防抖逻辑：等待用户停止输入后合并消息"""
        try:
            # 等待防抖时间
            await asyncio.sleep(self.wait_time)

            # 检查是否达到了最大等待时间
            current_time = time.time()
            last_time = self.last_activity.get(session_id, current_time)
            wait_duration = current_time - last_time

            async with self.lock[session_id]:
                # 获取缓冲区消息
                messages = self.buffers.get(session_id, [])
                if not messages:
                    return

                # 检查是否还有新的消息在等待
                # 如果距离最后一条消息的时间小于防抖时间，可能是并发问题，再等待一下
                if wait_duration < self.wait_time * 0.8:
                    logger.debug(f"[Debounce] 会话 {session_id} 有新的消息，延长等待")
                    self.tasks[session_id] = asyncio.create_task(
                        self._debounce(session_id, event)
                    )
                    return

                # 合并消息
                merged = self._merge_messages(messages)
                message_count = len(messages)
                
                # 1. 后台显示合并后的消息
                logger.info(f"[Debounce] ========== 合并消息详情 ==========")
                logger.info(f"[Debounce] 会话ID: {session_id}")
                logger.info(f"[Debounce] 原始消息数: {message_count}")
                logger.info(f"[Debounce] 原始消息列表:")
                for idx, msg in enumerate(messages, 1):
                    logger.info(f"[Debounce]   {idx}. {msg}")
                logger.info(f"[Debounce] 合并后的消息:")
                logger.info(f"[Debounce] {merged}")
                logger.info(f"[Debounce] =====================================")

                # 清空缓存
                self.buffers[session_id] = []

                # 2. 使用 event.send() 方法重新发送合并后的消息
                # 创建新的事件对象，模拟用户发送合并后的消息
                new_event = AstrMessageEvent(
                    session_id=session_id,
                    message_str=merged,
                    # 复制原事件的其他属性
                    platform=event.platform if hasattr(event, 'platform') else None,
                    adapter=event.adapter if hasattr(event, 'adapter') else None,
                )
                
                # 复制更多属性以保持兼容性
                if hasattr(event, 'metadata'):
                    new_event.metadata = event.metadata.copy() if event.metadata else {}
                if hasattr(event, 'self_id'):
                    new_event.self_id = event.self_id
                if hasattr(event, 'user_id'):
                    new_event.user_id = event.user_id
                
                # 添加合并元数据
                if hasattr(new_event, 'metadata'):
                    if not hasattr(new_event, 'metadata') or new_event.metadata is None:
                        new_event.metadata = {}
                    new_event.metadata['merged_count'] = message_count
                    new_event.metadata['merged'] = True
                    new_event.metadata['original_messages'] = messages

                logger.info(f"[Debounce] 发送合并后的消息到LLM: {merged[:100]}...")

                # 发送事件到LLM进行处理
                await self.context.send_event(new_event)

        except asyncio.CancelledError:
            logger.debug(f"[Debounce] 会话 {session_id} 任务被取消")
            raise
        except Exception as e:
            logger.error(f"[Debounce] 处理会话 {session_id} 出错: {e}")
            # 出错时确保缓存被清理，避免消息丢失
            async with self.lock[session_id]:
                if session_id in self.buffers:
                    self.buffers[session_id] = []

    def _merge_messages(self, messages):
        """合并多条消息为一条"""
        if not messages:
            return ""
        
        # 如果只有一条消息，直接返回
        if len(messages) == 1:
            return messages[0]
        
        # 多条消息：用分隔符连接
        # 检测消息是否已经包含标点符号，智能添加分隔
        merged = []
        for i, msg in enumerate(messages):
            msg = msg.strip()
            if not msg:
                continue
            
            # 如果前一条消息没有结束标点，添加句号
            if i > 0 and merged:
                last_char = merged[-1][-1] if merged[-1] else ""
                if last_char not in "。.!！?？；;：:”\"'’":
                    merged[-1] = merged[-1] + "。"
            
            merged.append(msg)
        
        # 用换行符连接，保持消息的独立性
        return "\n".join(merged)

    async def terminate(self):
        """插件终止时清理资源"""
        logger.info("Private Debounce Reply terminating...")
        
        # 取消清理任务
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # 取消所有防抖任务并处理缓存的消息
        for session_id, task in list(self.tasks.items()):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # 清理所有缓存
        self.buffers.clear()
        self.tasks.clear()
        self.lock.clear()
        self.last_activity.clear()
        
        logger.info("Private Debounce Reply terminated")
        
