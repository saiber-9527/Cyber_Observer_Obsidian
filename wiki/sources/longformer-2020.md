---
type: source-summary
title: "Longformer: The Long-Document Transformer"
date: 2020-04-10
source_url: "https://arxiv.org/abs/2004.05150"
domain: arxiv.org
author: "Iz Beltagy, Matthew E. Peters, Arman Cohan (Allen AI)"
tags: [wiki, wiki/source]
processed: true
raw_file: "raw/articles/ai-papers/09-Longformer-长文本处理(2020).md"
raw_sha256: "be9b4e20153b7f8ae4ec94e2030c0667aecdd768ab9c852a0dacf7b3d7dbfe26"
last_verified: 2026-06-01
possibly_outdated: false
language: en
---

## 摘要（中文）

Longformer 将标准 O(n²) 自注意力替换为 O(n) 的**局部窗口注意力 + 任务驱动全局注意力**混合机制，使 Transformer 能高效处理数千 token 的长文档，可作为 BERT 的直接替代品。

## 核心要点

- **局部滑动窗口注意力**：每个 token 只关注左右 w/2 个邻居，复杂度 O(n·w)
- **全局注意力**：对特殊 token（[CLS]、问题 token）启用全局注意力，使其能关注所有位置
- **空洞注意力**：通过跳跃采样增大感受野，不增加计算量
- 支持最长 4096 tokens（标准 BERT 仅 512），在长文档任务上显著优于 BERT
- 注意力复杂度从 O(n²) 降至 O(n)

## 提取概念

- [[linear-attention]]
- [[attention-mechanism]]

## 我的笔记
