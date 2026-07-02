---
type: concept
title: "多头注意力"
date: 2026-06-01
updated: 2026-06-01
tags: [wiki, wiki/concept]
source_count: 1
confidence: low
domain_volatility: low
last_reviewed: 2026-06-01
aliases:
  - "多头注意力"
  - "Multi-Head Attention"
  - "MHA"
---

## Definition

多头注意力（Multi-Head Attention）：将 Query、Key、Value 分别线性投影到 h 个低维子空间，并行执行 h 次注意力计算，最后拼接并再次投影。

`MultiHead(Q,K,V) = Concat(head₁,...,headₕ) Wᴼ`
`headᵢ = Attention(Q·Wᵢᴼ, K·Wᵢᴷ, V·Wᵢᵛ)`

Transformer 原论文中：h=8，dk=dv=dmodel/h=64。

## Key Points

- 允许模型在**不同表示子空间**同时关注**不同位置**的信息
- 单头注意力平均会抑制多维信息，多头解决了这个问题
- h 个头的总计算量与全维度单头注意力相当（降维保持计算量不变）
- 实验表明：head 数量过少或过多均会降低性能（8头最优）

## My Position

## Contradictions

## Sources

- [[attention-is-all-you-need]]

## Evolution Log

- 2026-06-01（1 source）：初始创建，来自 [[attention-is-all-you-need]]（Vaswani et al. 2017）
