from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.api import logger
from astrbot.api.message_components import Plain
import asyncio
from collections import defaultdict
import time


@register("Compose Supplement Reply", "babelqaq", "对用户的多条新消息进行整合并回复", "1.0.10")
class PrivateDebounceReply(Star):

    def __init__(self, context: Context):
        super().__init__(context)
        self.buffers = defaultdict(list)  # 缓存消息
        self.tasks = {}  # 防抖任务
        self.lock = defaultdict(asyncio.Lock)  # 会话锁
        self.last_activity = {}  # 记录最后活动时间
        
        # 从存储中加载配置，如果没有则使用默认值
        self.wait_time = 1.5  # 防抖等待时间（秒）
        self.cleanup_interval = 60  # 清理过期会话间隔（秒）
        self.session_timeout = 300  # 会话超时时间（秒）
        self._cleanup_task = None
        self._config_loaded = False

    async def initialize(self):
        """初始化插件"""
        logger.info("Private Debounce Reply initialized")
        # 加载配置
        await self._load_config()
        # 启动清理任务
        self._cleanup_task = asyncio.create_task(self._cleanup_sessions())

    async def _load_config(self):
        """从KV存储加载配置"""
        try:
            # 尝试加载保存的配置
            config = await self.get_kv_data("debounce_config", {})
            if config:
                self.wait_time = config.get("wait_time", self.wait_time)
                self.cleanup_interval = config.get("cleanup_interval", self.cleanup_interval)
                self.session_timeout = config.get("session_timeout", self.session_timeout)
                logger.info(f"[Config] 加载配置: wait_time={self.wait_time}s, "
                           f"cleanup_interval={self.cleanup_interval}s, "
                           f"session_timeout={self.session_timeout}s")
            else:
                # 首次运行，保存默认配置
                await self._save_config()
            self._config_loaded = True
        except Exception as e:
            logger.warning(f"[Config] 加载配置失败，使用默认值: {e}")

    async def _save_config(self):
        """保存配置到KV存储"""
        try:
            config = {
                "wait_time": self.wait_time,
                "cleanup_interval": self.cleanup_interval,
                "session_timeout": self.session_timeout
            }
            await self.put_kv_data("debounce_config", config)
            logger.info("[Config] 配置已保存")
        except Exception as e:
            logger.error(f"[Config] 保存配置失败: {e}")

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

            # 创建新的防抖任务 - 这里传递 event 的副本
            self.tasks[session_id] = asyncio.create_task(
                self._debounce(session_id, event)
            )

            # 停止当前事件传播，阻止进入LLM
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

                # 使用 event.send() 发送消息（不使用 yield）
                # 构建消息链
                message_chain = [Plain(merged_text)]
                
                # 发送消息
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

    # 添加配置管理指令
    @filter.command("debounce_config")
    async def config_command(self, event: AstrMessageEvent):
        """查看或修改防抖配置
        
        用法：
        /debounce_config - 查看当前配置
        /debounce_config wait_time 2.0 - 设置等待时间为2秒
        /debounce_config cleanup_interval 120 - 设置清理间隔为120秒
        /debounce_config session_timeout 600 - 设置会话超时为600秒
        """
        args = event.message_str.strip().split()
        
        if len(args) == 1:
            # 显示当前配置
            config_info = (
                f"📊 当前防抖配置：\n"
                f"⏱️  等待时间: {self.wait_time}s\n"
                f"🧹 清理间隔: {self.cleanup_interval}s\n"
                f"⏰ 会话超时: {self.session_timeout}s"
            )
            yield event.plain_result(config_info)
            return
        
        if len(args) == 3:
            key = args[1]
            try:
                value = float(args[2])
                
                if key == "wait_time" and value >= 0.5:
                    self.wait_time = value
                    await self._save_config()
                    yield event.plain_result(f"✅ 等待时间已更新为 {value}s")
                elif key == "cleanup_interval" and value >= 10:
                    self.cleanup_interval = value
                    await self._save_config()
                    yield event.plain_result(f"✅ 清理间隔已更新为 {value}s")
                elif key == "session_timeout" and value >= 30:
                    self.session_timeout = value
                    await self._save_config()
                    yield event.plain_result(f"✅ 会话超时已更新为 {value}s")
                else:
                    yield event.plain_result("❌ 参数无效，请检查键名和取值范围")
            except ValueError:
                yield event.plain_result("❌ 请提供有效的数字")
            return
        
        yield event.plain_result("❌ 用法错误，请参考 /debounce_config 帮助")

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
        
        # 保存当前配置
        await self._save_config()
        
        self.buffers.clear()
        self.tasks.clear()
        self.lock.clear()
        self.last_activity.clear()
        
        logger.info("Private Debounce Reply terminated")
