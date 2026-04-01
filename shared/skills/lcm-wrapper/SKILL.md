---
name: lcm-wrapper
description: Decorator/Hook wrapper for integrating LCM (Lossless Context Management) with nanobot. Provides non-invasive integration via @lcm_hook decorator, @lcm_tool decorator, and LcmHook class. Use when (1) integrating LCM with existing nanobot/agent code, (2) adding conversation memory to agents, (3) enabling automatic context compression, (4) providing retrieval tools to agents.
---

# LCM Wrapper - Nanobot Integration Skill

Decorator/Hook 方式的 LCM 整合包裝器，不改 nanobot 核心即可啟用對話記憶。

## 目錄結構

```
lcm-wrapper/
├── SKILL.md          # 本文件
├── __init__.py       # 導出
└── hook.py           # 主要實現
```

## 快速開始

### 方式 1：Decorator（推薦）

```python
from lcm_wrapper import lcm_hook, LcmHook

# 裝飾你的訊息處理函式
@lcm_hook
async def handle_message(ctx, lcm_hook=None):
    # ctx.messages 自動填充 LCM 組裝的上下文
    response = await llm.chat(ctx.messages)
    
    # 記錄回應
    await lcm_hook.on_message(role="assistant", content=response)
    
    return ctx

# 使用
await handle_message(ctx)
```

### 方式 2：Hook 類

```python
from lcm_wrapper import LcmHook

hook = LcmHook()
await hook.initialize()

# Session 開始
await hook.on_session_start(session_id="sess_001", title="今日對話")

# 訊息寫入
await hook.on_message(role="user", content="你好")
await hook.on_message(role="assistant", content="你好！")

# 獲取組裝的上下文
context = await hook.assemble_context()
# context.messages 可以直接餵給 LLM
```

### 方式 3：快速函式

```python
from lcm_wrapper import quick_start, quick_ingest, quick_assemble

# 一行啟動
hook = await quick_start(session_id="sess_001")

# 寫入訊息
await quick_ingest(conversation_id=hook._current_conversation_id, role="user", content="Hi")

# 組裝上下文
ctx = await quick_assemble(conversation_id=hook._current_conversation_id)
```

## API 參考

### LcmHook 類

#### 初始化

```python
hook = LcmHook(config=LcmHookConfig(
    db_path="~/.nanobot/workspace/lcm.db",  # 可選
    token_budget=128000,                      # 可選，預設 128000
    context_threshold=0.75,                   # 可選
    fresh_tail_count=32,                     # 可選
    auto_compact=True,                       # 可選，預設 True
    auto_compact_threshold=0.90,             # 可選
    system_prompt="你是 PixaClaw..."        # 可選
))
await hook.initialize()
```

#### Session 管理

```python
# 開始新 Session
conversation_id = await hook.on_session_start(
    session_id="unique_session_id",
    title="Optional Title",
    session_key="dedup_key"  # 可選，用於去重
)

# 結束時清理
hook.close()
```

#### 訊息攝入

```python
# 用戶訊息
msg_id = await hook.on_message(
    role="user",
    content="使用者輸入的內容"
)

# AI 回應
msg_id = await hook.on_message(
    role="assistant",
    content="AI 回應內容"
)

# 帶 Tool Call
msg_id = await hook.on_message(
    role="assistant",
    content="讓我搜尋...",
    parts=[
        {
            "type": "tool",
            "tool_name": "search",
            "tool_call_id": "call_001",
            "tool_input": '{"query": "python"}',
            "tool_output": '{"results": [...]}',
            "tool_status": "success"
        }
    ]
)
```

#### 上下文組裝

```python
# 自動 compaction + 組裝
context = await hook.assemble_context(
    system_prompt="你是...",  # 可選，覆蓋預設
    force_compact=False       # 可選，強制壓縮
)

# context 是 HookContext:
# - context.messages: List[Dict] - 可直接餵給 LLM
# - context.token_count: int
# - context.compacted: bool
# - context.metadata: Dict
```

#### 檢索工具

```python
# 查詢摘要/檔案/訊息
result = hook.describe("sum_abc123")
# 返回格式化字串

# 全文搜索
result = hook.grep("python", mode="full_text")
# 返回格式化搜索結果

# 展開摘要
result = hook.expand("sum_abc123", max_tokens=4000)

# 混合搜索（BM25 + decay + recency）
result = hook.hybrid_search("python project", min_tier="working")
```

### Decorator

#### @lcm_hook

```python
from lcm_wrapper import lcm_hook, LcmHook

@lcm_hook
async def my_handler(ctx, lcm_hook=None):
    # 取得注入的 hook
    h = lcm_hook or @lcm_hook.get_hook()
    
    # 組裝上下文
    context = await h.assemble_context()
    ctx.messages = context.messages
    
    # 處理並回傳
    return ctx
```

#### @lcm_tool

```python
from lcm_wrapper import lcm_tool, LcmHook

hook = LcmHook()
await hook.initialize()

@lcm_tool(name="lcm_describe", description="查詢摘要")
def describe(item_id: str) -> str:
    '''查詢摘要或檔案'''
    return hook.describe(item_id)

@lcm_tool(name="lcm_grep", description="搜尋對話")
def grep(query: str) -> str:
    '''搜尋對話歷史'''
    return hook.grep(query)

# 工具自動帶有 _lcm_tool_name 和 _lcm_tool_description 屬性
```

## Nanobot 整合檢查清單

| 步驟 | 檔案 | 狀態 |
|------|------|------|
| 1. 安裝 Skill | 將 lcm-wrapper 加入 nanobot skills | ⬜️ |
| 2. 初始化 | `hook = LcmHook(); await hook.initialize()` | ⬜️ |
| 3. Session 開始 | `await hook.on_session_start(session_id=...)` | ⬜️ |
| 4. 訊息寫入 | `await hook.on_message(role=..., content=...)` | ⬜️ |
| 5. 上下文獲取 | `context = await hook.assemble_context()` | ⬜️ |
| 6. 檢索工具 | `hook.describe()`, `hook.grep()`, etc. | ⬜️ |

## 整合範例

### Telegram Bot 整合

```python
from lcm_wrapper import LcmHook

hook = LcmHook()
await hook.initialize()

async def on_telegram_message(update):
    chat_id = update.message.chat.id
    text = update.message.text
    
    # 確保 session
    await hook.on_session_start(
        session_id=f"telegram_{chat_id}",
        session_key=str(chat_id)
    )
    
    # 記錄用戶訊息
    await hook.on_message(role="user", content=text)
    
    # 組裝上下文
    context = await hook.assemble_context()
    
    # 送給 LLM
    response = await llm.chat(context.messages)
    
    # 記錄回應
    await hook.on_message(role="assistant", content=response)
    
    return response
```

### Web API 整合

```python
from lcm_wrapper import LcmHook, lcm_hook

hook = LcmHook()
await hook.initialize()

@lcm_hook
async def handle_chat(ctx, session_id: str, message: str, lcm_hook=None):
    # 自動處理 LCM
    h = lcm_hook
    
    await h.on_message(role="user", content=message)
    context = await h.assemble_context()
    
    response = await llm.chat(context.messages)
    await h.on_message(role="assistant", content=response)
    
    ctx.messages = context.messages
    ctx.response = response
    return ctx
```

## 配置選項

| 選項 | 預設值 | 說明 |
|------|--------|------|
| `db_path` | `~/.nanobot/workspace/lcm.db` | 資料庫路徑 |
| `token_budget` | `128000` | LLM context window |
| `context_threshold` | `0.75` | 使用 75% budget 開始組裝 |
| `fresh_tail_count` | `32` | 保留最近 32 條訊息 |
| `auto_compact` | `True` | 自動壓縮 |
| `auto_compact_threshold` | `0.90` | 超過 90% threshold 觸發壓縮 |

## 注意事項

1. **單例模式**：`LcmHook.get_instance()` 返回全域實例
2. **非同步**：`initialize()`, `on_session_start()`, `on_message()`, `assemble_context()` 都是 async
3. **自動 compaction**：在 `on_message()` 時自動檢查並執行
4. **Tool 裝飾器**：裝飾的函式帶有 `_lcm_tool_name` 和 `_lcm_tool_description` 屬性
