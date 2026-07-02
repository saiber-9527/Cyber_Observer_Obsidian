---
type: concept
title: "Transformer架构"
date: 2026-06-01
updated: 2026-06-01
tags: [wiki, wiki/concept]
source_count: 1
confidence: low
domain_volatility: low
last_reviewed: 2026-06-01
aliases:
  - "Transformer架构"
  - "Transformer Architecture"
  - "Transformer"
---

## Definition

Transformer架构（Transformer Architecture）：2017年由 Vaswani 等人提出的序列转换模型，**完全基于注意力机制**，摒弃了此前主流的 RNN 和 CNN，是现代大语言模型（BERT、GPT系列等）的基础架构。

**结构**：编码器（6层）× 解码器（6层），每层包含：
1. Multi-Head Self-Attention 子层
2. Position-wise Feed-Forward Network 子层（dff=2048）
3. 残差连接 + Layer Normalization

## Key Points

- 完全并行化：训练速度远快于 RNN（12小时 vs 数周）
- 编码器将输入映射为连续表示 z，解码器自回归生成输出序列
- Decoder 中使用掩码自注意力，防止当前位置看到未来信息
- base 模型：N=6, dmodel=512, dff=2048, h=8, dk=dv=64，参数量 65M
- big 模型：N=6, dmodel=1024, dff=4096, h=16，参数量 213M
- WMT 2014 EN-DE BLEU 28.4，EN-FR BLEU 41.8（当时最优）

## My Position

## Contradictions

## Sources

- [[attention-is-all-you-need]]

## Evolution Log

- 2026-06-01（1 source）：初始创建，来自 [[attention-is-all-you-need]]（Vaswani et al. 2017）
