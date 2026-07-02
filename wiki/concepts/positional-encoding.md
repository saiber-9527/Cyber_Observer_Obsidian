---
type: concept
title: "位置编码"
date: 2026-06-01
updated: 2026-06-01
tags: [wiki, wiki/concept]
source_count: 1
confidence: low
domain_volatility: low
last_reviewed: 2026-06-01
aliases:
  - "位置编码"
  - "Positional Encoding"
  - "PE"
---

## Definition

位置编码（Positional Encoding）：Transformer 中用于向模型注入序列位置信息的机制。由于注意力机制本身与顺序无关，必须显式提供位置信息。

原论文采用正弦/余弦函数：
- `PE(pos, 2i) = sin(pos / 10000^(2i/dmodel))`
- `PE(pos, 2i+1) = cos(pos / 10000^(2i/dmodel))`

位置编码与词嵌入相加（维度相同，均为 dmodel=512）。

## Key Points

- 正弦式编码的优势：模型可能泛化到训练时未见过的更长序列
- 学习式位置嵌入与正弦式结果几乎相同（实验 E）
- 每个维度对应不同频率的正弦波，波长从 2π 到 10000·2π

## My Position

## Contradictions

## Sources

- [[attention-is-all-you-need]]

## Evolution Log

- 2026-06-01（1 source）：初始创建，来自 [[attention-is-all-you-need]]（Vaswani et al. 2017）
