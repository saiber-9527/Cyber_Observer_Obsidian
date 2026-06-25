# 大模型 API 调用实战

> 各大模型厂商的 API 调用方法，说人话、贴代码。

---

## OpenAI API

```bash
# 安装
pip install openai
```

```python
from openai import OpenAI

client = OpenAI(api_key="sk-xxx")

# Chat Completion（最常用）
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "你是一名网络安全专家"},
        {"role": "user", "content": "解释一下零信任架构"}
    ],
    temperature=0.3,  # 安全场景用低温度
    max_tokens=2000
)

print(response.choices[0].message.content)
```

### Function Calling

```python
tools = [{
    "type": "function",
    "function": {
        "name": "check_ip",
        "description": "查询IP信誉",
        "parameters": {
            "type": "object",
            "properties": {
                "ip": {"type": "string", "description": "IP地址"}
            },
            "required": ["ip"]
        }
    }
}]

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "查一下这个IP: 1.2.3.4"}],
    tools=tools,
    tool_choice="auto"
)

# 判断模型是否要调用工具
if response.choices[0].message.tool_calls:
    # 提取函数名和参数
    func_name = response.choices[0].message.tool_calls[0].function.name
    arguments = response.choices[0].message.tool_calls[0].function.arguments
    print(f"调用 {func_name}, 参数 {arguments}")
```

---

## DeepSeek API

```python
from openai import OpenAI  # DeepSeek 兼容 OpenAI SDK

client = OpenAI(
    api_key="sk-xxx",
    base_url="https://api.deepseek.com"
)

response = client.chat.completions.create(
    model="deepseek-chat",  # DeepSeek V4
    messages=[{"role": "user", "content": "你好"}],
    stream=True
)

for chunk in response:
    print(chunk.choices[0].delta.content or "", end="")
```

---

## 通义千问 API

```python
from openai import OpenAI  # 同样兼容 OpenAI SDK

client = OpenAI(
    api_key="sk-xxx",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

response = client.chat.completions.create(
    model="qwen-max",
    messages=[{"role": "user", "content": "你好"}]
)

print(response.choices[0].message.content)
```

---

## 流式输出

```python
# 适合聊天场景，逐 token 显示
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "讲个安全冷笑话"}],
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

---

## 成本估算

```
GPT-4o:         输入 $2.5/百万 token，输出 $10/百万 token
DeepSeek Chat:  输入 $0.14/百万 token，输出 $0.28/百万 token
Qwen-Max:       ¥3.5/百万 token，输出 ¥12/百万 token

一次典型问答（假设 2K 输入 + 1K 输出）：
  GPT-4o:   约 $0.015
  DeepSeek: 约 $0.0005
  Qwen-Max: 约 ¥0.019
```

---

## ⚠️ 安全注意事项

1. **API Key 不能硬编码** — 使用环境变量 `os.getenv("OPENAI_API_KEY")`
2. **流式输出可能包含敏感信息** — 输出要做安全过滤
3. **日志不能记录完整 prompt 和 response** — 尤其是包含用户个人信息的请求
4. **设置 timeout 防止无限等待**
5. **注意 token 消耗** — 长对话 token 消耗飞快

#AI工具 #API #开发 #实战
