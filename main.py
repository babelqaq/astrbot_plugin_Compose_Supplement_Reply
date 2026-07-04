from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.api import logger
import asyncio
from collections import defaultdict


@register("Compose Supplement Reply", "babelqaq", "对用户的多条新消息进行整合并回复", "1.0.1")
class StableDebounceSafe(Star):

    def __init__(self, context: Context):
        super().__init__(context)

        self.buffers = defaultdict(list)
        self.tasks = {}
        self.locks = defaultdict(asyncio.Lock)
        self.delay = 1.0

    async def initialize(self):
        logger.info("[Debounce SAFE] init")

    @filter.on_waiting_llm_request()
    async def on_waiting(self, event: AstrMessageEvent):

        session = event.session_id
        msg = event.message_str

        if not msg or msg.startswith("/"):
            return

        async with self.locks[session]:

            # 1. 缓冲消息
            self.buffers[session].append(msg)

            # 2. 取消旧任务（防抖核心）
            task = self.tasks.get(session)
            if task and not task.done():
                task.cancel()

            # 3. 创建新的 flush 任务
            self.tasks[session] = asyncio.create_task(
                self._flush(session, event)
            )

            # 4. 阻断当前 LLM 请求
            event.stop_event()

    async def _flush(self, session: str, event: AstrMessageEvent):

        try:
            await asyncio.sleep(self.delay)

            # 取出并合并消息（必须在锁内保证一致性）
            async with self.locks[session]:
                msgs = self.buffers.get(session, [])
                if not msgs:
                    return

                merged = "\n".join(msgs)
                self.buffers[session].clear()

            # ⭐ 后台日志输出
            logger.info(f"[Debounce SAFE MERGED]\n{merged}")

            # ⭐ 关键：重新进入 LLM 链路
            # 注意：这里不要再 hold lock
            await event.send(merged)

        except asyncio.CancelledError:
            pass

        except Exception as e:
            logger.error(f"[Debounce SAFE] error: {e}")

    async def terminate(self):
        for task in self.tasks.values():
            if task and not task.done():
                task.cancel()
        logger.info("[Debounce SAFE] terminated")
