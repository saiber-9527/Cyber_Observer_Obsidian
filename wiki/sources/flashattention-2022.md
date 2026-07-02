---
type: source-summary
title: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
date: 2022-06-23
source_url: "https://arxiv.org/abs/2205.14135"
domain: arxiv.org
author: "Tri Dao, Daniel Y. Fu, Stefano Ermon et al. (Stanford)"
tags: [wiki, wiki/source]
processed: true
raw_file: "raw/articles/ai-papers/08-FlashAttention-IO优化注意力(2022).md"
raw_sha256: "6c49023111c1adc6cf7a9cf93ea4f234e04cca4bad1912e86ae019d2cfbeedb7"
last_verified: 2026-06-01
possibly_outdated: false
language: en
---

## 摘要（中文）

FlashAttention 针对 Transformer 注意力的 **IO 瓶颈**（GPU 内存带宽，而非计算量）进行优化。通过**分块计算（Tiling）**避免大型注意力矩阵写入 HBM，实现精确注意力计算同时速度提升 2-4 倍、显存减少 5-20 倍。已成为现代 LLM 训练标配。

## 核心要点

- **核心洞见**：注意力计算瓶颈在 HBM（High Bandwidth Memory）读写次数，而非 FLOPS
- **分块计算（Tiling）**：Q/K/V 分成小块，每次只载入 SRAM（快速片上内存），大幅减少 HBM 访问
- **重计算（Recomputation）**：反向传播时不存储注意力矩阵（节省显存），而是重新计算
- **精确注意力**：数值完全等价于标准注意力（非近似）
- BERT 训练加速 15%，GPT-2 加速 3 倍，支持 8K-64K tokens 长序列
- 已被 PyTorch 2.0+ 原生集成

## 提取概念

- [[io-aware-attention]]
- [[attention-mechanism]]

## 我的笔记
