---
type: source-summary
title: "Retentive Network: A Successor to Transformer for Large Language Models"
date: 2023-07-17
source_url: "https://arxiv.org/abs/2307.08621"
domain: arxiv.org
author: "Yutao Sun, Li Dong et al. (Microsoft Research & Tsinghua)"
tags: [wiki, wiki/source]
processed: true
raw_file: "raw/articles/ai-papers/16-RetNet-保持机制(2023).md"
raw_sha256: "71f922f6cbaf1e12a057f18ef31640393cb4dee065ae641f6e76c953f8ca599b"
last_verified: 2026-06-01
possibly_outdated: false
language: en
---

## 摘要（中文）

RetNet 提出**保持机制（Retention）**，理论统一 Transformer（并行训练）与 RNN（高效推理）的优点：训练时用并行形式，推理时切换为 O(1) 每步的循环形式。微软亚洲研究院与清华联合提出。

## 核心要点

- **三种计算范式**：并行（训练加速）、循环（O(1) 推理）、分块循环（平衡两者）
- **保持机制**：在注意力公式中加入指数衰减因子 γ，使远距离位置影响指数级衰减
- **多尺度保持（MSR）**：不同头使用不同衰减率，类似多头注意力
- 推理内存 O(1) vs Transformer 的 O(n)，2048 token 解码速度快 8.4 倍
- 语言建模性能与 Transformer 相当（略低于同规模 Transformer）

## 提取概念

- [[linear-attention]]
- [[state-space-model]]

## 我的笔记
