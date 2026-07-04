from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.api import logger
import asyncio
from collections import defaultdict


@register("Compose Supplement Reply", "babelqaq", "对用户的多条新消息进行整合并回复", "1.0.2")
class ComposeSupplementReply(Star):

    def __init__(self, context: Context):
        super().__init__(context)

        self.buffers = defaultdict(list)
        self.tasks = {}
        self.locks = defaultdict(asyncio.Lock)
        self.delay = 1.0
        self.pending_sessions = set()

    async def initialize(self):
        logger.info("[Compose Reply] initialized")

    @filter.on_waiting_llm_request()
    async def on_waiting(self, event: AstrMessageEvent):

        session = event.session_id
        msg = event.message_str

        if not msg or msg.startswith("/"):
            return

        async with self.locks[session]:
            # 添加到缓冲区
            self.buffers[session].append(msg)
            logger.info(f"[Compose Reply] 缓冲消息 - 会话: {session[:8]} 内容: {msg[:30]}...")
            
            # 取消旧任务（防抖核心）
            task = self.tasks.get(session)
            if task and not task.done():
                task.cancel()
                logger.debug(f"[Compose Reply] 取消旧任务")

            # 创建新的等待任务
            self.tasks[session] = asyncio.create_task(
                self._wait_and_mark_ready(session)
            )

        # 等待合并完成（阻塞当前请求）
        while session in self.pending_sessions:
            await asyncio.sleep(0.1)

    async def _wait_and_mark_ready(self, session: str):
        """等待延迟后标记会话准备就绪"""
        try:
            await asyncio.sleep(self.delay)
            async with self.locks[session]:
                self.pending_sessions.discard(session)
        except asyncio.CancelledError:
            pass

    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, req):
        """在调用LLM前合并消息"""
        session = event.session_id

        async with self.locks[session]:
            msgs = self.buffers.get(session, [])
            if len(msgs) > 1:
                merged = "\n".join(msgs)
                logger.info(f"[Compose Reply] 合并消息 - 会话: {session[:8]} 消息数: {len(msgs)}")
                logger.info(f"[Compose Reply] 合并内容:\n{merged}")
                # 修改请求文本为合并后的消息
                req.text = merged
            
            # 清空缓冲区
            self.buffers[session] = []

    async def terminate(self):
        for task in self.tasks.values():
            if task and not task.done():
                task.cancel()
        logger.info("[Compose Reply] terminated")
