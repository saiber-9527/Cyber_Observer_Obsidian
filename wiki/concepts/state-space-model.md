---
type: concept
title: "状态空间模型"
date: 2026-06-01
updated: 2026-06-01
tags: [wiki, wiki/concept]
source_count: 2
confidence: low
domain_volatility: high
last_reviewed: 2026-06-01
aliases:
  - "状态空间模型"
  - "State Space Model"
  - "SSM"
  - "选择性状态空间"
---

## Definition

状态空间模型（State Space Model，SSM）：用连续时间状态方程（h'(t) = Ah(t) + Bx(t)，y(t) = Ch(t)）描述序列动态的模型，可视为线性 RNN 的推广。离散化后可高效并行训练，推理时以递归形式以 O(1) 每步处理序列，天然适合长序列。

## Key Points

- **连续时间**：SSM 源自控制理论，用微分方程描述系统状态演化
- **离散化**：通过 ZOH 或双线性变换离散化，得到可并行训练的形式
- **线性 RNN 视角**：SSM 推理时与 RNN 等价，O(1) 推理内存
- **Mamba 的选择性 SSM**：参数 B/C/∆ 依赖输入（input-dependent），赋予内容感知能力
- **RetNet 的保持机制**：SSM + 指数衰减，支持并行/循环/分块三种计算范式

## My Position

## Contradictions

## Sources

- [[mamba-2024]]
- [[retnet-2023]]

## Evolution Log

- 2026-06-01（2 sources）：初始创建，来自 Mamba 和 RetNet
