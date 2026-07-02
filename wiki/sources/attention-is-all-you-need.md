---
type: source-summary
title: "Attention Is All You Need"
date: 2017-06-12
source_url: "https://arxiv.org/abs/1706.03762"
domain: "arxiv.org"
author: "Ashish Vaswani, Noam Shazeer, Niki Parmar et al. (Google Brain)"
tags: [wiki, wiki/source]
processed: true
raw_file: "raw/articles/ai-papers/01-Attention机制-Transformer开山之作(2017).md"
raw_sha256: "468813a797197766095a06b2d84ea5c3e74f91bd1ce6353605563a55c88e044a"
last_verified: 2026-06-01
possibly_outdated: false
language: en
canonical_source: ""
---

## Summary

2017年 NIPS 论文，Google Brain 提出 Transformer 架构。核心贡献是**完全摒弃 RNN 和 CNN，仅凭注意力机制构建序列转换模型**。在机器翻译任务上以显著更低的训练成本超越所有此前最优模型，并证明可泛化到其他任务（英语成分分析）。

## Key Points

- 提出 **Scaled Dot-Product Attention**：`Attention(Q,K,V) = softmax(QKᵀ/√dk)V`，缩放因子 1/√dk 防止梯度消失
- 提出 **Multi-Head Attention**：h=8 个并行注意力头，dk=dv=64，允许模型在不同表示子空间同时关注不同位置信息
- **位置编码**采用正弦/余弦函数，可能泛化到比训练时更长的序列
- 编码器6层 × 解码器6层，每层含 Self-Attention + Feed-Forward（dff=2048）
- 在 WMT 2014 EN-DE 达到 BLEU 28.4（+2 超越之前所有模型），EN-FR 达到 BLEU 41.8，训练成本仅为竞争模型的 1/4
- 并行化程度远高于 RNN：Self-Attention 复杂度 O(n²·d)，Sequential Operations O(1)

## Concepts Extracted

- [[attention-mechanism]]
- [[multi-head-attention]]
- [[transformer-architecture]]
- [[self-attention]]
- [[positional-encoding]]
- [[encoder-decoder-architecture]]

## Entities Extracted

- [[ashish-vaswani]]
- [[google-brain]]

## Contradictions

无（开山之作，无同类竞品比较中的根本矛盾）

## My Notes
