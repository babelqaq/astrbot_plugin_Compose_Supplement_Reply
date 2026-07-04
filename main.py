from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.api import logger
from astrbot.api.message_components import Plain
from astrbot.api.message_chain import MessageChain
import asyncio
from collections import defaultdict
import time


@register("Compose Supplement Reply", "babelqaq", "对用户的多条新消息进行整合并回复", "1.0.12")
class PrivateDebounceReply(Star):

    def __init__(self, context: Context):
        super().__init__(context)
        self.buffers = defaultdict(list)
        self.tasks = {}
        self.lock = defaultdict(asyncio.Lock)
        self.last_activity = {}
        
        # 配置参数
        self.wait_time = 1.5
        self.cleanup_interval = 60
        self.session_timeout = 300
        self._cleanup_task = None
        self._config_loaded = False

    async def initialize(self):
        """初始化插件"""
        logger.info("Private Debounce Reply initialized")
        await self._load_config()
        self._cleanup_task = asyncio.create_task(self._cleanup_sessions())

    async def _load_config(self):
        """从KV存储加载配置"""
        try:
            config = await self.get_kv_data("debounce_config", {})
            if config:
                self.wait_time = config.get("wait_time", self.wait_time)
                self.cleanup_interval = config.get("cleanup_interval", self.cleanup_interval)
                self.session_timeout = config.get("session_timeout", self.session_timeout)
                logger.info(f"[Config] 加载配置: wait_time={self.wait_time}s")
            else:
                await self._save_config()
            self._config_loaded = True
        except Exception as e:
            logger.warning(f"[Config] 加载配置失败: {e}")

    async def _save_config(self):
        """保存配置到KV存储"""
        try:
            config = {
                "wait_time": self.wait_time,
                "cleanup_interval": self.cleanup_interval,
                "session_timeout": self.session_timeout
            }
            await self.put_kv_data("debounce_config", config)
        except Exception as e:
            logger.error(f"[Config] 保存配置失败: {e}")

    async def _cleanup_sessions(self):
        """定期清理过期会话"""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                current_time = time.time()
                expired = [sid for sid, t in self.last_activity.items() 
                          if current_time - t > self.session_timeout]
                
                for session_id in expired:
                    async with self.lock[session_id]:
                        self.buffers.pop(session_id, None)
                        task = self.tasks.pop(session_id, None)
                        if task and not task.done():
                            task.cancel()
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

        if not msg or msg.startswith("/"):
            return

        self.last_activity[session_id] = time.time()

        async with self.lock[session_id]:
            self.buffers[session_id].append(msg)
            logger.info(f"[Debounce] 会话 {session_id} 缓冲消息: {msg[:30]}...")

            # 取消旧任务
            task = self.tasks.get(session_id)
            if task and not task.done():
                task.cancel()

            # 创建新任务
            self.tasks[session_id] = asyncio.create_task(
                self._debounce(session_id)
            )

            # 阻止当前消息进入LLM
            event.stop_event()

    async def _debounce(self, session_id):
        """防抖核心逻辑：等待、合并、发送"""
        try:
            # 等待防抖时间
            await asyncio.sleep(self.wait_time)

            async with self.lock[session_id]:
                messages = self.buffers.get(session_id, [])
                if not messages:
                    return

                # 检查是否还有新消息
                current_time = time.time()
                last_time = self.last_activity.get(session_id, current_time)
                if current_time - last_time < self.wait_time * 0.8:
                    logger.debug(f"[Debounce] 会话 {session_id} 有新消息，延长等待")
                    self.tasks[session_id] = asyncio.create_task(
                        self._debounce(session_id)
                    )
                    return

                # 合并消息
                merged_text = self._merge_messages(messages)
                message_count = len(messages)
                
                # 后台日志
                logger.info(f"[Debounce] ========== 合并消息详情 ==========")
                logger.info(f"[Debounce] 会话ID: {session_id}")
                logger.info(f"[Debounce] 消息数: {message_count}")
                for idx, msg in enumerate(messages, 1):
                    logger.info(f"[Debounce]   {idx}. {msg}")
                logger.info(f"[Debounce] 合并后: {merged_text}")
                logger.info(f"[Debounce] =====================================")

                # 清空缓存
                self.buffers[session_id] = []

                # 构造并发送消息链
                message_chain = MessageChain()
                message_chain.chain = [Plain(merged_text)]
                
                # 注意：这里不能使用 yield，必须使用 event.send()
                # 但是我们没有 event 对象了！
                # 需要从 context 获取当前会话的 event
                # 这里需要通过 unified_msg_origin 来发送
                
                # 使用 context.send_message 发送主动消息
                # 我们需要构造 unified_msg_origin
                umo = f"Adam_v1:FriendMessage:{session_id}"  # 这里需要根据实际情况调整
                await self.context.send_message(umo, message_chain)
                
                logger.info(f"[Debounce] 已发送合并消息")

        except asyncio.CancelledError:
            logger.debug(f"[Debounce] 会话 {session_id} 任务取消")
        except Exception as e:
            logger.error(f"[Debounce] 会话 {session_id} 出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
            async with self.lock[session_id]:
                self.buffers[session_id] = []

    def _merge_messages(self, messages):
        """智能合并消息"""
        if not messages:
            return ""
        if len(messages) == 1:
            return messages[0]
        
        merged = []
        for i, msg in enumerate(messages):
            msg = msg.strip()
            if not msg:
                continue
            if i > 0 and merged and merged[-1][-1] not in "。.!！?？；;：:”\"'’":
                merged[-1] = merged[-1] + "。"
            merged.append(msg)
        return "\n".join(merged)

    @filter.command("debounce_config")
    async def config_command(self, event: AstrMessageEvent):
        """配置管理指令"""
        args = event.message_str.strip().split()
        
        if len(args) == 1:
            info = (f"📊 当前配置：\n"
                   f"⏱️  等待时间: {self.wait_time}s\n"
                   f"🧹 清理间隔: {self.cleanup_interval}s\n"
                   f"⏰ 会话超时: {self.session_timeout}s")
            yield event.plain_result(info)
            return
        
        if len(args) == 3:
            key, val = args[1], args[2]
            try:
                value = float(val)
                if key == "wait_time" and value >= 0.5:
                    self.wait_time = value
                elif key == "cleanup_interval" and value >= 10:
                    self.cleanup_interval = value
                elif key == "session_timeout" and value >= 30:
                    self.session_timeout = value
                else:
                    yield event.plain_result("❌ 参数无效")
                    return
                await self._save_config()
                yield event.plain_result(f"✅ {key} 已更新为 {value}")
            except ValueError:
                yield event.plain_result("❌ 请输入有效数字")
            return
        
        yield event.plain_result("❌ 用法: /debounce_config [key] [value]")

    async def terminate(self):
        """清理资源"""
        logger.info("Private Debounce Reply terminating...")
        
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
        
        await self._save_config()
        self.buffers.clear()
        self.tasks.clear()
        self.lock.clear()
        self.last_activity.clear()
        logger.info("Private Debounce Reply terminated")
