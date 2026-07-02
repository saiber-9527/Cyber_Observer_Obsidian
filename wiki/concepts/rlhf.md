---
type: concept
title: "人类反馈强化学习"
date: 2026-06-01
updated: 2026-06-01
tags: [wiki, wiki/concept]
source_count: 2
confidence: low
domain_volatility: high
last_reviewed: 2026-06-01
aliases:
  - "人类反馈强化学习"
  - "Reinforcement Learning from Human Feedback"
  - "RLHF"
---

## Definition

人类反馈强化学习（RLHF）：通过人类标注员对模型输出进行偏好标注（哪个回答更好），训练奖励模型（Reward Model），再用强化学习（通常是 PPO）优化语言模型使其输出符合人类偏好。是 ChatGPT、GPT-4、Qwen-Chat 等对话模型的核心对齐技术。

## Key Points

- **三阶段流程**：①SFT 监督微调 → ②训练奖励模型（RM）→ ③PPO 强化学习优化
- **偏好数据**：人类标注员对同一 prompt 的多个回答排序，构建比较数据集
- **奖励模型**：预测人类对回答的偏好分数，替代真实人类反馈
- **PPO 优化**：用 RM 分数作为奖励信号，优化语言模型策略
- **替代方案**：DPO（Direct Preference Optimization）直接优化偏好，无需显式 RM

## My Position

## Contradictions

- RLHF 可能导致奖励欺骗（Reward Hacking）：模型学会迎合 RM 而非真正提高质量

## Sources

- [[gpt-4-2023]]
- [[qwen-2023]]

## Evolution Log

- 2026-06-01（2 sources）：初始创建
