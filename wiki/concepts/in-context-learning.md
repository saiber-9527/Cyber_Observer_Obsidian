---
type: concept
title: "上下文学习"
date: 2026-06-01
updated: 2026-06-01
tags: [wiki, wiki/concept]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-06-01
aliases:
  - "上下文学习"
  - "In-Context Learning"
  - "ICL"
---

## Definition

上下文学习（In-Context Learning，ICL）：语言模型从 prompt 中的示例中"学习"任务模式，无需更新任何参数即可适配新任务。GPT-3 提出的核心能力，是提示工程（Prompt Engineering）的理论基础。

## Key Points

- 知识通过 prompt 传递，而非参数更新——这是与微调的根本区别
- 机制存在争议：是真正从示例中学习推理，还是对预训练数据中相似模式的检索？
- 对示例的顺序、格式敏感（Few-shot 的不稳定性来源）
- **Chain-of-Thought（CoT）**：在示例中加入推理链，大幅提升复杂推理能力
- 是现代 LLM 应用（RAG、Agent）的核心基础能力

## My Position

## Contradictions

- ICL 的内部机制至今不完全清楚：梯度下降类比 vs 贝叶斯推理 vs 模式匹配，学界尚无定论

## Sources

- [[gpt-3-2020]]

## Evolution Log

- 2026-06-01（1 source）：初始创建，来自 GPT-3
