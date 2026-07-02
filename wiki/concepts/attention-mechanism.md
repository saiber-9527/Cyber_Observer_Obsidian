---
type: concept
title: "注意力机制"
date: 2026-06-01
updated: 2026-06-01
tags: [wiki, wiki/concept]
source_count: 1
confidence: low
domain_volatility: low
last_reviewed: 2026-06-01
aliases:
  - "注意力机制"
  - "Attention Mechanism"
  - "attention"
---

## Definition

注意力机制（Attention Mechanism）：一种允许模型在处理序列时，对输入的不同位置赋予不同权重的机制。核心思想是将查询（Query）与键（Key）计算相似度，再用该相似度对值（Value）加权求和，从而"关注"最相关的信息。

**Scaled Dot-Product Attention 公式**：
`Attention(Q, K, V) = softmax(QKᵀ / √dk) · V`

缩放因子 `1/√dk` 的作用：防止维度较大时点积结果过大导致 softmax 梯度消失。

## Key Points

- 解决了 RNN 长距离依赖问题：任意两个位置之间路径长度为 O(1)（RNN 为 O(n)）
- 可完全并行化，不像 RNN 需要顺序计算
- 分为三种使用场景：编码器自注意力、解码器自注意力（带掩码）、编码器-解码器交叉注意力
- 加性注意力（Additive Attention）和点积注意力在理论复杂度相似，但点积注意力实践中更快

## My Position

## Contradictions

## Sources

- [[attention-is-all-you-need]]

## Evolution Log

- 2026-06-01（1 source）：初始创建，来自 [[attention-is-all-you-need]]（Vaswani et al. 2017）
