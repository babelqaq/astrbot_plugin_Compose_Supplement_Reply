# 多条消息合并回复插件 / MergeChat

## 📖 简介 / Introduction

**中文**  
喜欢发送多条短消息但又不想astrbot每条都回吗？本插件将用户在私聊中短时间内发送的多条消息合并为一条，再交由模型统一处理并生成回复。

**English**  
Want to send multiple short messages but don't want Astrbot to reply to each one individually? This plugin combines several messages sent by a user in quick succession during private chat into a single message, which is then processed and responded to collectively by the model.

### 适用场景 / Use Cases

| 中文 | English |
|------|---------|
| 用户习惯分多条发送短消息（如“你好”、“今天天气怎么样”、“我想出去玩”） | Users habitually send fragmented short messages (e.g., "Hello", "What's the weather today", "I want to go out") |
| 避免 LLM 对每条消息单独回复，造成对话碎片化 | Avoid LLM replying to each message separately, reducing fragmentation |
| 提升对话连贯性，减少重复上下文 | Improve conversation coherence and reduce redundant context |

---

## ✨ 功能特性 / Features

| 功能 | 说明 |
|------|------|
| ✅ **私聊防抖合并** | 在用户停止输入后等待指定时间（默认 6 秒），自动合并该期间内的所有消息 |
| ✅ **直接 LLM 调用** | 合并后直接调用大语言模型生成回复，不重复进入消息管道 |
| ✅ **对话历史保存** | 自动将合并后的消息和 LLM 回复存入会话历史，保持上下文连贯 |
| ✅ **可视化配置** | 支持通过 WebUI 调整防抖时间、清理间隔等参数 |
| ✅ **轻量设置** | 自动清理过期会话 |

---

## 📦 安装 / Installation

### 方法一：通过 AstrBot 插件市场安装（推荐）
在 AstrBot WebUI 的 **插件管理** 页面搜索 `Compose Supplement Reply` 并安装。

### 方法二：手动安装
```bash
cd /path/to/astrbot/data/plugins
git clone https://github.com/babelqaq/astrbot_plugin_Compose_Supplement_Reply.git
```

# 配置与使用示例

## ⚙️ 配置说明

### 配置项详情

插件安装后，可在 AstrBot WebUI 的 **插件配置** 页面调整以下参数：

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `wait_time` | float | 6.0 | 用户停止输入后的等待时间（秒），值越大合并的消息越多 |
| `cleanup_interval` | int | 120 | 清理过期会话的检查间隔（秒） |
| `session_timeout` | int | 600 | 会话无活动后的超时时间（秒） |

## 🚀 使用示例 / Usage Example

```text
用户: 你好
用户: 今天天气怎么样
用户: 我想出去玩
（等待 6 秒）
Bot: 今天天气晴朗，适合出去玩哦！🌞
```
## 其他
该插件输出内容在Astrbot自带的ChatUI里无法显示，但可以在后台看见
在OneBot v11+个人QQ上输出正常，没有问题。其他平台暂未测试，欢迎反馈。


