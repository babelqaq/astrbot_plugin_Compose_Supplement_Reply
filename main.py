from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.api import logger
from astrbot.api.message_components import Plain
from astrbot.core.message.message_event import MessageChain
import asyncio
from collections import defaultdict
import time


@register("Compose Supplement Reply", "babelqaq", "对用户的多条新消息进行整合并回复", "1.0.6")
class PrivateDebounceReply(Star):

    def __init__(self, context: Context):
        super().__init__(context)
        self.buffers = defaultdict(list)  # 缓存消息
        self.tasks = {}  # 防抖任务
        self.lock = defaultdict(asyncio.Lock)  # 会话锁
        self.last_activity = {}  # 记录最后活动时间
        self.wait_time = 1.5  # 防抖等待时间（秒）
        self.cleanup_interval = 60  # 清理过期会话间隔（秒）
        self.session_timeout = 300  # 会话超时时间（秒）
        self._cleanup_task = None

    async def initialize(self):
        """初始化插件"""
        logger.info("Private Debounce Reply initialized")
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
                        # 清理所有相关数据
                        self.buffers.pop(session_id, None)
                        task = self.tasks.pop(session_id, None)
                        if task and not task.done():
                            task.cancel()
                        self.last_activity.pop(session_id, None)
                        self.lock.pop(session_id, None)
                        logger.info(f"[Cleanup] 清理过期会话: {session_id}")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[Cleanup] 清理会话出错: {e}")

    @filter.on_waiting_llm_request()
    async def on_waiting(self, event: AstrMessageEvent):
        """在等待LLM请求时拦截消息并进行防抖处理"""
        session_id = event.session_id
        # 获取消息的纯文本内容
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

            # 使用官方推荐的方法停止当前事件传播，阻止进入LLM
            event.stop_event()

    async def _debounce(self, session_id, event):
        """防抖逻辑：等待用户停止输入后合并消息并重新发送"""
        try:
            await asyncio.sleep(self.wait_time)

            current_time = time.time()
            last_time = self.last_activity.get(session_id, current_time)
            wait_duration = current_time - last_time

            async with self.lock[session_id]:
                messages = self.buffers.get(session_id, [])
                if not messages:
                    return

                # 如果还有新消息，延长等待
                if wait_duration < self.wait_time * 0.8:
                    logger.debug(f"[Debounce] 会话 {session_id} 有新的消息，延长等待")
                    self.tasks[session_id] = asyncio.create_task(
                        self._debounce(session_id, event)
                    )
                    return

                # 合并消息
                merged_text = self._merge_messages(messages)
                message_count = len(messages)
                
                # 后台详细显示合并信息
                logger.info(f"[Debounce] ========== 合并消息详情 ==========")
                logger.info(f"[Debounce] 会话ID: {session_id}")
                logger.info(f"[Debounce] 原始消息数: {message_count}")
                logger.info(f"[Debounce] 原始消息列表:")
                for idx, msg in enumerate(messages, 1):
                    logger.info(f"[Debounce]   {idx}. {msg}")
                logger.info(f"[Debounce] 合并后的消息:")
                logger.info(f"[Debounce] {merged_text}")
                logger.info(f"[Debounce] =====================================")

                # 清空缓存
                self.buffers[session_id] = []

                # 创建 MessageChain 对象
                # 方法1：使用 MessageChain 构造函数
                message_chain = MessageChain()
                message_chain.chain = [Plain(merged_text)]
                
                # 或者方法2：直接创建包含 Plain 的 MessageChain
                # from astrbot.core.message.message_event import MessageChain
                # message_chain = MessageChain([Plain(merged_text)])
                
                # 使用官方推荐的 event.send() 方法发送合并后的消息
                await event.send(message_chain)
                
                logger.info(f"[Debounce] 已发送合并后的消息到LLM")

        except asyncio.CancelledError:
            logger.debug(f"[Debounce] 会话 {session_id} 任务被取消")
        except Exception as e:
            logger.error(f"[Debounce] 处理会话 {session_id} 出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # 出错时清理缓存
            async with self.lock[session_id]:
                if session_id in self.buffers:
                    self.buffers[session_id] = []

    def _merge_messages(self, messages):
        """智能合并多条消息为一条"""
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

    async def terminate(self):
        """插件终止时清理资源"""
        logger.info("Private Debounce Reply terminating...")
        
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # 取消所有任务并清理
        for session_id, task in list(self.tasks.items()):
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
        
        logger.info("Private Debounce Reply terminated")
