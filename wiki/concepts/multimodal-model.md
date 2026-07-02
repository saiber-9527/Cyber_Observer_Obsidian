---
type: concept
title: "多模态模型"
date: 2026-06-01
updated: 2026-06-01
tags: [wiki, wiki/concept]
source_count: 4
confidence: low
domain_volatility: high
last_reviewed: 2026-06-01
aliases:
  - "多模态模型"
  - "Multimodal Model"
  - "多模态大模型"
  - "MLLM"
---

## Definition

多模态模型（Multimodal Model）：能同时处理和理解多种输入模态（如文本、图像、音频、视频）的 AI 模型。多模态理解指不同模态之间的语义对齐和联合推理。

## Key Points

- **早期多模态**：CLIP（图文对比对齐）、LDM（文本条件图像生成）
- **多模态 LLM**：GPT-4V、Gemini、LLaVA 等——将图像/视频理解集成到语言模型
- **原生多模态 vs 拼接多模态**：Gemini 从训练开始就联合处理多模态；GPT-4V 是将图像编码器嫁接到语言模型
- **模态对齐**：不同模态的表示需要投影到同一语义空间（CLIP 的核心贡献）
- 发展趋势：从单模态 LLM → 图文多模态 → 全模态（音频/视频/代码）

## My Position

## Contradictions

## Sources

- [[clip-2021]]
- [[latent-diffusion-2022]]
- [[gpt-4-2023]]
- [[gemini-2024]]

## Evolution Log

- 2026-06-01（4 sources）：初始创建
