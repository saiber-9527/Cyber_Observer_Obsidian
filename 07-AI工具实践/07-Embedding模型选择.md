# Embedding 模型选择

> 把文本变成向量——RAG、语义搜索、聚类的基础。

---

## 什么是 Embedding

```
"苹果是一种水果"  →  Embedding 模型  →  [0.12, -0.34, 0.67, ..., 0.89]  （768维向量）
"苹果是一家公司"  →  Embedding 模型  →  [0.89, 0.12, -0.45, ..., -0.23]  （768维向量）

这两个向量的距离（余弦相似度）比较大，因为语境不同。
```

---

## 主流 Embedding 模型对比

| 模型 | 厂商 | 维度 | 最大输入 | 语言 | 特点 |
|------|------|------|---------|------|------|
| **text-embedding-3-small** | OpenAI | 512/1536 | 8191 tokens | 多语言 | 最强通用，付费 |
| **text-embedding-3-large** | OpenAI | 256/3072 | 8191 tokens | 多语言 | 精度最高，付费 |
| **bge-large-zh-v1.5** | BAAI | 1024 | 512 tokens | 中英 | 开源，中文最好之一 |
| **bge-m3** | BAAI | 1024 | 8192 tokens | 多语言 | M3：多语言+多粒度 |
| **m3e-large** | 莫妮卡 | 1024 | 512 tokens | 中文为主 | 开源中文 |
| **gte-Qwen2-1.5B** | Alibaba | 1536 | 8192 tokens | 多语言 | 基于 Qwen 的嵌入 |
| **gte-large-zh** | Alibaba | 1024 | 512 tokens | 中文 | 阿里出品 |
| **jina-embeddings-v3** | Jina AI | 1024 | 8192 tokens | 多语言 | 开源，长文本 |
| **Cohere Embed v3** | Cohere | 1024/384 | 512 tokens | 多语言 | 企业级，付费 |

---

## 选型指南

```
中文知识库 RAG：
  ├── 小规模 → bge-large-zh / m3e-large
  ├── 大规模 → bge-m3 / gte-Qwen2
  └── 极致精度 → text-embedding-3-large（付费）

英文 RAG：
  ├── 高性价比 → text-embedding-3-small
  └── 极致精度 → text-embedding-3-large

多语言（中英混合）：
  ├── 开源 → bge-m3
  └── 付费 → text-embedding-3-large

长文本（>512 tokens）：
  ├── 开源 → jina-embeddings-v3 / bge-m3
  └── 付费 → text-embedding-3-large
```

---

## Embedding 质量评估

### 评测维度

| 维度 | 说明 | 评估方法 |
|------|------|---------|
| **检索准确率** | Top-k 结果中有多少真正相关 | MTEB、BEIR 基准 |
| **聚类效果** | 同主题文档是否聚集在一起 | Silhouette 分数 |
| **跨语言对齐** | 不同语言的相似文本是否靠近 | 双语评估 |
| **分类效果** | 向量表示是否能区分不同类别 | 分类器测试 |

### MTEB 中文基准得分参考

```
Model                      Retrieval    Classification  Reranking  Average
bge-m3                       62.5         75.2         65.2       66.8
bge-large-zh-v1.5            60.8         74.1         63.8       65.3
gte-Qwen2-1.5B               63.8         74.9         66.1       67.1
m3e-large                    58.2         71.5         62.3       63.4
text-embedding-3-small       59.7         72.8         61.9       64.0
```

---

## Embedding 使用实战

```python
from sentence_transformers import SentenceTransformer
import numpy as np

# 加载模型（本地运行）
model = SentenceTransformer('BAAI/bge-large-zh-v1.5')

# 文档分块
documents = [
    "AES是一种对称加密算法，密钥长度128/256位",
    "RSA是非对称加密算法，密钥长度2048/4096位",
    "哈希函数SHA-256输出256位的摘要"
]

# 计算向量
embeddings = model.encode(documents, normalize_embeddings=True)
print(f"向量维度: {embeddings.shape}")  # (3, 1024)

# 查询
query = "对称加密有哪些"
query_embedding = model.encode(query, normalize_embeddings=True)

# 余弦相似度（已经归一化，所以直接用点积）
scores = np.dot(embeddings, query_embedding)
best_idx = np.argmax(scores)
print(f"最相关文档: {documents[best_idx]} (相似度: {scores[best_idx]:.3f})")
```

---

## ⚠️ 常见陷阱

1. **向量维度越高不一定越好** — 向量维度高 + 数据量小 = 过拟合
2. **中文场景必须用中文 Embedding** — 用英文 Embedding 处理中文，效果很差
3. **Embedding 模型不在同一维度的不能直接比较** — 换了模型需要重新计算全部向量
4. **长文本超过模型输入限制会被截断** — 注意分块策略和模型的最大输入长度
5. **Embedding 质量决定了 RAG 的上限** — 换更好的 Embedding 模型比花更多精力调 RAG 流程更有效

#AI工具 #Embedding #向量 #RAG #选型
