---
type: source-summary
title: "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity"
date: 2021-01-11
source_url: "https://arxiv.org/abs/2101.03961"
domain: arxiv.org
author: "William Fedus, Barret Zoph, Noam Shazeer (Google)"
tags: [wiki, wiki/source]
processed: true
raw_file: "raw/articles/ai-papers/11-SwitchTransformer-MoE工业化(2021).md"
raw_sha256: "3667fc9972b4cd8dbee6c31646d476249986b7fafa8a3b680de8745325ae21ab"
last_verified: 2026-06-01
possibly_outdated: false
language: en
---

## 摘要（中文）

Switch Transformer 将 MoE 工业化，关键简化是**每次只路由到 1 个专家（k=1）**，大幅简化训练稳定性。最大版本达到 **1.6T 参数**（激活参数约 30B），以 7 倍于 T5-Base 的速度达到相同困惑度。

## 核心要点

- **k=1 路由**：比 Top-2 更简单稳定，实验证明性能损失可忽略
- **容量因子**：控制每个专家能接收的最大 token 数，防止溢出
- **辅助负载均衡损失**：鼓励均匀分配，防止专家坍塌
- **bf16 混合精度**：解决 MoE 训练不稳定问题
- 1.6T 参数验证了 MoE 的可扩展性，为后续 GPT-4、Mixtral 等奠基

## 提取概念

- [[mixture-of-experts]]
- [[sparse-activation]]
- [[scaling-law]]

## 我的笔记
