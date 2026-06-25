# AI Agent

> 从"我问你答"到"设定目标，自主执行"。

---

## 什么是 AI Agent

**Agent = 大模型 + 工具 + 记忆 + 规划**

```
用户：帮我查一下这个服务器的安全配置

Agent：
  1. 拆解任务：需要先 SSH 连服务器、执行检查命令、读取结果
  2. 调用工具：SSH 工具 → 执行 `sudo security-check.sh`
  3. 分析结果：发现端口 22 对外开放且未限制来源 IP
  4. 给出建议：建议修改安全组策略
```

---

## Agent 的关键组件

| 组件 | 说明 | 技术方案 |
|------|------|---------|
| **规划（Planning）** | 拆解大任务为子任务 | ReAct、Plan-and-Execute |
| **工具（Tools）** | 能够调用的外部能力 | Function Calling、API、代码执行 |
| **记忆（Memory）** | 对话历史、持久化知识 | Vector Store、SQLite |
| **执行（Execution）** | 实际执行工具调用 | Python 代码执行、Shell 命令 |

---

## 常见 Agent 框架

| 框架 | 特点 |
|------|------|
| **LangChain Agents** | 生态丰富，工具多 |
| **AutoGen（微软）** | 多 Agent 对话协作 |
| **CrewAI** | 角色化多 Agent 团队 |
| **OpenAI Assistants API** | 官方，内置 FaaS 和检索 |
| **Semantic Kernel（微软）** | .NET 友好 |

---

## ReAct 模式

思考（Thought）→ 行动（Action）→ 观察（Observation）→ 循环

```
Thought: 用户想知道一个 IP 是否在威胁情报黑名单中。
Action: 调用威胁情报API查询这个IP
Observation: API返回该IP近期有恶意活动记录
Thought: 需要生成报告下结论
Action: 给出总结和风险评级
```

---

## Function Calling（工具调用）

```python
# 定义工具
tools = [{
    "type": "function",
    "function": {
        "name": "check_ip_reputation",
        "description": "检查IP在威胁情报中的信誉",
        "parameters": {
            "type": "object",
            "properties": {
                "ip": {"type": "string"}
            }
        }
    }
}]

# 模型决定是否调用
response = client.chat.completions.create(
    model="gpt-4",
    tools=tools,
    messages=[{"role": "user", "content": "查一下 1.2.3.4 的可疑程度"}]
)
```

---

## Agent 安全的特殊关注点

| 风险 | 说明 |
|------|------|
| **工具误调用** | Agent 可能错误调用危险操作（如删除数据） |
| **提示注入传播** | 用户恶意 prompt 可能让 Agent 调用不当工具 |
| **递归循环** | Agent 陷入无限循环 |
| **权限过高** | Agent 的 API Key 应该严格限制 |

---

## ⚠️ 实践建议

1. **Agent 不能有危险工具的自由使用权限** — 删除操作要人工确认
2. **设置步骤上限** — 防止 Agent 无限循环（max_iterations）
3. **日志全量记录** — 每个工具调用都要记录，用于审计和调试
4. **从简单 Agent 开始** — 不要一上来就搞多 Agent 协作

#AI工具 #Agent #概念
