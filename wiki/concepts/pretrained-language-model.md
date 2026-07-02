---
type: concept
title: "预训练语言模型"
date: 2026-06-01
updated: 2026-06-01
tags: [wiki, wiki/concept]
source_count: 8
confidence: medium
domain_volatility: medium
last_reviewed: 2026-06-01
aliases:
  - "预训练语言模型"
  - "Pre-trained Language Model"
  - "PLM"
---

## Definition

预训练语言模型（Pre-trained Language Model，PLM）：先在大规模无标签文本上进行自监督预训练，学习通用语言表示，再在下游任务上微调或直接推理的模型范式。核心思想是"先学语言，再学任务"。

## Key Points

- **预训练目标**：掩码语言模型（MLM，如 BERT）或自回归语言模型（如 GPT）
- **迁移学习**：预训练的通用表示可迁移到多种下游任务，大幅降低标注数据需求
- **规模效应**：参数量越大、数据越多、计算越多，性能越强（缩放定律）
- **两大路线**：编码器路线（BERT 系，适合理解）vs 解码器路线（GPT 系，适合生成）
- **发展脉络**：BERT(2018) → RoBERTa/ALBERT(2019) → GPT-3(2020) → GPT-4/Gemini(2023-2024)

## My Position

## Contradictions

## Sources

- [[bert-2018]]
- [[roberta-2019]]
- [[albert-2019]]
- [[gpt-1-2018]]
- [[gpt-2-2019]]
- [[gpt-3-2020]]
- [[glm-2021]]
- [[qwen-2023]]

## Evolution Log

- 2026-06-01（8 sources）：初始创建，整合 BERT/GPT 系列共 8 篇论文的共同概念
