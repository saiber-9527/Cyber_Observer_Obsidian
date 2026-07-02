---
type: concept
title: "掩码语言模型"
date: 2026-06-01
updated: 2026-06-01
tags: [wiki, wiki/concept]
source_count: 3
confidence: low
domain_volatility: low
last_reviewed: 2026-06-01
aliases:
  - "掩码语言模型"
  - "Masked Language Model"
  - "MLM"
---

## Definition

掩码语言模型（Masked Language Model，MLM）：一种自监督预训练目标，随机遮盖输入序列中的部分 token（通常 15%），训练模型预测被遮盖的词。类似"完形填空"，让模型学习利用双向上下文理解语言。由 BERT 提出并推广。

## Key Points

- BERT 的核心预训练任务之一：随机选 15% token → 80% 替换为 [MASK]，10% 随机词，10% 不变
- 相比自回归语言模型，MLM 允许双向上下文利用，更适合理解任务
- **动态掩码**（RoBERTa 改进）：每次 epoch 重新生成掩码，比静态掩码效果更好
- GLM 的"自回归填空"是 MLM 的变体：遮盖连续 span，自回归生成补全

## My Position

## Contradictions

- RoBERTa 和 ALBERT 均证明 BERT 的 NSP（下一句预测）配合 MLM 反而有害，去掉 NSP 更好

## Sources

- [[bert-2018]]
- [[roberta-2019]]
- [[glm-2021]]

## Evolution Log

- 2026-06-01（3 sources）：初始创建，来自 BERT、RoBERTa、GLM 三篇论文
