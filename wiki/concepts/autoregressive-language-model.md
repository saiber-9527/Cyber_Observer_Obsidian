---
type: concept
title: "自回归语言模型"
date: 2026-06-01
updated: 2026-06-01
tags: [wiki, wiki/concept]
source_count: 6
confidence: medium
domain_volatility: low
last_reviewed: 2026-06-01
aliases:
  - "自回归语言模型"
  - "Autoregressive Language Model"
  - "Causal Language Model"
  - "CLM"
---

## Definition

自回归语言模型（Autoregressive Language Model）：从左到右逐 token 生成序列，每个 token 只能看到其左侧上下文（因果掩码注意力）。训练目标是最大化每个 token 的条件概率：P(x_t | x_1,...,x_{t-1})。GPT 系列的核心架构。

## Key Points

- **因果注意力**：Decoder 中的掩码自注意力，防止当前位置看到未来 token
- **天然适合生成**：自回归采样逐步生成文本，可控性好
- **KV Cache**：推理时缓存历史 K/V 向量，避免重复计算（但内存占用随序列长度增长）
- **规模扩展**：GPT-1(117M) → GPT-2(1.5B) → GPT-3(175B) → GPT-4(未知)，性能持续提升
- **涌现能力**：规模到一定程度出现零样本/少样本能力（GPT-2/GPT-3）

## My Position

## Contradictions

- 与 MLM（双向）的根本对立：自回归适合生成但理解能力弱于双向模型

## Sources

- [[gpt-1-2018]]
- [[gpt-2-2019]]
- [[gpt-3-2020]]
- [[glm-2021]]
- [[qwen-2023]]
- [[gpt-4-2023]]

## Evolution Log

- 2026-06-01（6 sources）：初始创建，整合 GPT 系列和 GLM/Qwen
