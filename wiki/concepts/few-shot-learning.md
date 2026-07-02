---
type: concept
title: "少样本学习"
date: 2026-06-01
updated: 2026-06-01
tags: [wiki, wiki/concept]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-06-01
aliases:
  - "少样本学习"
  - "Few-shot Learning"
  - "Few-shot"
---

## Definition

少样本学习（Few-shot Learning）：在 prompt 中提供少量（通常 1-64 个）示例，让模型在不更新参数的情况下适应新任务。是 GPT-3 提出的核心能力，也是上下文学习（In-Context Learning）的主要形式。

## Key Points

- **k-shot**：提供 k 个示例，k 越大效果越好，但受 context window 长度限制
- 与微调的本质区别：Few-shot 不更新参数，示例只在上下文中存在
- GPT-3 在大量 NLP 任务上，few-shot 性能接近或超越有监督微调的小模型
- **涌现性**：少样本能力随模型规模突然涌现，小模型不具备

## My Position

## Contradictions

## Sources

- [[gpt-3-2020]]

## Evolution Log

- 2026-06-01（1 source）：初始创建，来自 GPT-3
