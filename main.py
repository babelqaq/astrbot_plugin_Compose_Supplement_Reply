from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.api import logger
import asyncio
from collections import defaultdict


@register("Compose Supplement Reply", "babelqaq", "对用户的多条新消息进行整合并回复", "1.0.0")
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

            self.buffers[session].append(msg)

            task = self.tasks.get(session)
            if task and not task.done():
                task.cancel()

            self.tasks[session] = asyncio.create_task(
                self._flush(session, event)
            )

            event.stop_event()

    async def _flush(self, session: str, event: AstrMessageEvent):

        try:
            await asyncio.sleep(self.delay)

            async with self.locks[session]:

                msgs = self.buffers.get(session, [])
                if not msgs:
                    return

                merged = "\n".join(msgs)
                self.buffers[session].clear()

                logger.info(f"[Debounce SAFE MERGED]\n{merged}")

                # ⭐⭐⭐ 修复：plain_result不是异步方法，不需要await
                event.plain_result(merged)

        except asyncio.CancelledError:
            pass

        except Exception as e:
            logger.error(f"[Debounce SAFE] error: {e}")

    async def terminate(self):
        # 取消所有任务
        for task in self.tasks.values():
            if task and not task.done():
                task.cancel()
        logger.info("[Debounce SAFE] terminated")