---
type: concept
title: "零样本学习"
date: 2026-06-01
updated: 2026-06-01
tags: [wiki, wiki/concept]
source_count: 3
confidence: low
domain_volatility: medium
last_reviewed: 2026-06-01
aliases:
  - "零样本学习"
  - "Zero-shot Learning"
  - "Zero-shot"
---

## Definition

零样本学习（Zero-shot Learning）：模型在从未见过该任务训练样本的情况下，仅凭任务描述或类别名称完成推理。在语言模型语境中，指不在 prompt 中提供示例，直接要求模型完成任务。

## Key Points

- GPT-2 首次在大规模 LLM 上系统展示零样本能力
- CLIP 实现图像分类的零样本迁移：类别名 → 文本描述 → 与图像嵌入匹配
- 零样本能力是规模效应的体现：小模型不具备，到一定规模自然涌现
- 与少样本（Few-shot）的区别：零样本不提供任何示例，少样本提供 k 个示例

## My Position

## Contradictions

## Sources

- [[gpt-2-2019]]
- [[clip-2021]]
- [[gpt-3-2020]]

## Evolution Log

- 2026-06-01（3 sources）：初始创建
