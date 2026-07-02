---
type: source-summary
title: "ALBERT: A Lite BERT for Self-supervised Learning of Language Representations"
date: 2019-09-26
source_url: "https://arxiv.org/abs/1909.11942"
domain: arxiv.org
author: "Zhenzhong Lan, Mingda Chen et al. (Google Research)"
tags: [wiki, wiki/source]
processed: true
raw_file: "raw/articles/ai-papers/04-ALBERT-轻量化BERT(2019).md"
raw_sha256: "e3bc4ffd29821f5504c86c4b7722dd059b4d304bd76fbbc916090cb7bcdddc7f"
last_verified: 2026-06-01
possibly_outdated: false
language: en
---

## 摘要（中文）

ALBERT 通过两项参数压缩技术将 BERT 参数量大幅压缩，同时以更少参数取得更高分数，解决大模型显存和训练时间瓶颈。ALBERT-xxlarge 仅 235M 参数却超越 BERT-LARGE（334M）。

## 核心要点

- **因式分解词嵌入**：词表嵌入维度（E=128）与隐层维度（H=768）解耦，大幅减少嵌入层参数
- **跨层参数共享**：所有 Transformer 层共享参数（注意力头权重 + FFN），参数量骤降但计算量不变
- **句子顺序预测（SOP）** 替代 NSP：判断两句话顺序是否被交换，比 NSP 更难、效果更好
- ALBERT-xxlarge 在 GLUE/RACE/SQuAD 上全面超越 BERT-LARGE

## 提取概念

- [[parameter-sharing]]
- [[pretrained-language-model]]

## 矛盾

- 参数共享减少参数量，但训练/推理的计算量并未减少（每层仍需计算，只是权重相同）

## 我的笔记
