---
type: source-summary
title: "RoBERTa: A Robustly Optimized BERT Pretraining Approach"
date: 2019-07-26
source_url: "https://arxiv.org/abs/1907.11692"
domain: arxiv.org
author: "Yinhan Liu, Myle Ott et al. (Facebook AI)"
tags: [wiki, wiki/source]
processed: true
raw_file: "raw/articles/ai-papers/03-RoBERTa-BERT优化版(2019).md"
raw_sha256: "e074857670638a8fe065935c2c76564ba394da908b3d8b0e7608b613d72fdf17"
last_verified: 2026-06-01
possibly_outdated: false
language: en
---

## 摘要（中文）

Facebook AI 对 BERT 进行系统性消融实验，发现 BERT 存在**严重训练不足**问题。RoBERTa 不改变模型架构，仅通过优化训练方案在多个基准上全面超越 BERT，证明"训练策略"对预训练模型性能影响极大。核心结论：更大 batch、更多数据、更长训练、去掉 NSP，就能大幅提升性能。

## 核心要点

- **去除 NSP 任务**：实验证明下一句预测反而有害，去掉后性能提升
- **更大 batch size**：从 256 增至 8192
- **更多数据**：从 16GB 扩展到 160GB（新增 CC-News、OpenWebText、Stories）
- **更长训练步数**：训练步数增加 10 倍
- **动态掩码**：每次 epoch 重新生成掩码，而非静态掩码
- 结论：BERT 能力边界远未被充分探索，工程调优与数据量同样重要

## 提取概念

- [[pretrained-language-model]]
- [[masked-language-model]]

## 矛盾

- 直接推翻 BERT 的 NSP 设计；后续 ALBERT 也印证了 NSP 有害

## 我的笔记
