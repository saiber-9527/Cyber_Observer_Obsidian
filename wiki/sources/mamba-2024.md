---
type: source-summary
title: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
date: 2023-12-01
source_url: "https://arxiv.org/abs/2312.00752"
domain: arxiv.org
author: "Albert Gu, Tri Dao (CMU & Princeton)"
tags: [wiki, wiki/source]
processed: true
raw_file: "raw/articles/ai-papers/15-Mamba-状态空间模型(2024).md"
raw_sha256: "f11730ab69082f15f505a3d2588561454ee8489fdc61f537b2cf2b7dd9e00f73"
last_verified: 2026-06-01
possibly_outdated: false
language: en
---

## 摘要（中文）

Mamba 提出**选择性状态空间模型（Selective SSM）**：系统参数随输入动态变化，使模型能对相关信息"记住"、对无关信息"遗忘"。在语言任务上首次与 Transformer 性能持平，且推理速度快 5 倍（无 KV Cache 瓶颈），O(1) 推理内存。

## 核心要点

- **SSM 基础**：状态空间模型可视为线性 RNN，矩阵 A/B/C 描述隐状态转移，复杂度 O(n)
- **选择性机制（核心创新）**：参数 B、C、∆ 由输入决定（非固定值），赋予模型"内容感知"能力
- **硬件感知实现**：类似 FlashAttention，在 SRAM 内完成计算，避免 HBM 读写瓶颈
- **无注意力机制**：推理时隐状态大小恒定，不随序列长度增长
- 在语言建模（Pile）上以相同参数量超越 Transformer，推理吞吐量为 Transformer 的 5 倍

## 提取概念

- [[state-space-model]]
- [[linear-attention]]

## 矛盾

- 在需要精确 recall 的任务（如复制序列）上，Mamba 仍弱于 Transformer 的精确注意力

## 我的笔记
