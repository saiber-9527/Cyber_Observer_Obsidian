---
type: concept
title: "微调"
date: 2026-06-01
updated: 2026-06-01
tags: [wiki, wiki/concept]
source_count: 3
confidence: low
domain_volatility: medium
last_reviewed: 2026-06-01
aliases:
  - "微调"
  - "Fine-tuning"
  - "SFT"
  - "监督微调"
---

## Definition

微调（Fine-tuning）：在预训练模型基础上，用少量有标注的任务特定数据继续训练，更新全部或部分参数，使模型适配特定下游任务。是"预训练+微调"范式的第二阶段。

## Key Points

- **全参数微调**：更新所有参数，效果最好但计算成本高
- **参数高效微调（PEFT）**：LoRA、Adapter 等方法只更新少量参数（全参数的 0.1%-1%）
- **SFT（监督微调）**：用人工标注的指令-回复对微调，使模型遵循指令
- BERT 微调范式：在预训练模型上加一层任务特定输出层，整体微调
- 与 zero-shot/few-shot 的对比：微调更新参数；zero/few-shot 不更新参数

## My Position

## Contradictions

## Sources

- [[bert-2018]]
- [[gpt-1-2018]]
- [[qwen-2023]]

## Evolution Log

- 2026-06-01（3 sources）：初始创建
