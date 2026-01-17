# EAGLE 数据准备与模型训练全流程文档

> 面向入门深度学习开发者的完整教程

## 📚 文档导航

本教程系列涵盖了EAGLE（Extrapolation Algorithm for Greater Language-model Efficiency）项目的数据准备和模型训练全流程，适合有一定深度学习基础的开发者学习使用。

### 文档结构

| 文档 | 描述 | 预计字数 |
|------|------|----------|
| [01. 入门指南](docs/01-getting-started.md) | 投机解码理论、EAGLE架构概述、环境配置 | ~3000字 |
| [02. 数据准备详解](docs/02-data-preparation.md) | Chat/FIM数据格式、预处理流程、数据增强 | ~4000字 |
| [03. 模型架构深度解析](docs/03-model-architecture.md) | EAGLE-1/3架构对比、核心组件详解 | ~5000字 |
| [04. 训练流程详解](docs/04-training-pipeline.md) | EAGLE-1/3训练脚本逐行解析 | ~5000字 |
| [05. 实战教程](docs/05-practical-tutorial.md) | 端到端训练示例、完整工作流 | ~4000字 |
| [06. 高级主题](docs/06-advanced-topics.md) | 性能优化、调试指南、自定义扩展 | ~3000字 |
| [07. API参考](docs/07-api-reference.md) | 完整函数签名、参数说明 | ~2000字 |

---

## 🎯 快速开始

### 前置知识

阅读本教程前，建议您具备以下基础知识：

- **深度学习框架**：PyTorch基础操作
- **Transformers库**：Hugging Face Transformers使用经验
- **LLM基础**：了解Transformer架构、自回归生成
- **训练概念**：熟悉损失函数、优化器、学习率调度等

### 环境要求

```bash
# Python版本
Python >= 3.8

# 主要依赖
torch >= 2.0.0
transformers >= 4.53.1
deepspeed  # 仅EAGLE-3需要

# 可选
wandb  # 训练监控
```

### 5分钟快速上手

如果您想快速了解EAGLE训练流程，建议按以下顺序阅读：

1. **入门指南** → 理解投机解码原理
2. **数据准备详解** → 准备训练数据
3. **实战教程** → 跟随完整示例训练模型
4. **高级主题** → 解决实际问题

---

## 📖 EAGLE版本对比

| 特性 | EAGLE-1 | EAGLE-2 | EAGLE-3 |
|------|---------|---------|---------|
| **架构** | 单层特征预测 | 动态草稿树 | 多层次特征融合 |
| **训练** | 单GPU | 复用EAGLE-1 | DeepSpeed分布式 |
| **数据格式** | 预处理.pt文件 | 同EAGLE-1 | Chat/FIM原始数据 |
| **加速比** | 3x (13B) | 4x (13B) | 5.6x (13B) |
| **推荐场景** | 小规模实验 | 推理优化 | 大规模训练 |
| **文档重点** | ✅ | 推理优化 | ✅ |

**本教程主要讲解EAGLE-1和EAGLE-3的训练流程。**

---

## 🔑 核心概念速览

### 投机解码 (Speculative Decoding)

投机解码是一种通过"猜测"未来token来加速LLM推理的技术：

1. **草稿模型** (Draft Model)：快速生成多个候选token
2. **目标模型** (Target Model)：并行验证所有候选token
3. **接受/拒绝**：保留正确的token，回滚错误的token

### EAGLE的创新点

传统方法训练独立的草稿模型，而EAGLE：

- **特征外推**：直接利用基础模型的中间层特征
- **无需独立训练**：共享基础模型的embedding和大部分参数
- **保证一致性**：理论上保持与原始模型相同的输出分布

### 训练目标

EAGLE的训练目标非常简单：

1. **特征重建**：预测基础模型的下一层hidden states
2. **token预测**：匹配基础模型的token概率分布
3. **多位置预测**：一次性预测未来多个位置

---

## 📂 项目结构概览

```
EAGLE-Code-Completion/
├── eagle/
│   ├── train/                 # EAGLE-1训练脚本
│   │   └── main.py            # 单GPU训练主脚本
│   ├── traineagle3/           # EAGLE-3训练脚本
│   │   ├── main.py            # DeepSpeed分布式训练
│   │   ├── cnets.py           # EAGLE-3模型定义
│   │   └── configs.py         # 模型配置
│   ├── model/                 # 模型定义
│   │   ├── ea_model.py        # 统一推理接口
│   │   ├── cnets1.py          # EAGLE-1模型
│   │   └── configs.py         # 配置类
│   └── data/
│       └── fim/               # FIM格式示例数据
├── docs/                      # 本教程文档
└── README_TRAINING.md         # 本文档
```

---

## 💡 学习路径建议

### 路径A：理论+实践型（推荐）

适合希望深入理解原理的开发者

```
01. 入门指南 → 03. 模型架构深度解析
     ↓
02. 数据准备详解 → 04. 训练流程详解
     ↓
05. 实战教程 → 06. 高级主题
```

### 路径B：快速上手型

适合只想快速跑通代码的开发者

```
01. 入门指南（理论部分可跳过）
     ↓
02. 数据准备详解（仅看数据格式）
     ↓
05. 实战教程（跟随完整示例）
     ↓
06. 高级主题（遇到问题时查阅）
```

### 路径C：深入研究型

适合希望扩展或改进EAGLE的开发者

```
按顺序阅读所有文档 + 阅读源代码
     ↓
参考07. API参考进行二次开发
     ↓
参考06. 高级主题进行优化
```

---

## 🔗 相关资源

### 论文

- [EAGLE (ICML 2024)](https://arxiv.org/pdf/2401.15077.pdf)
- [EAGLE-2 (EMNLP 2024)](https://arxiv.org/pdf/2406.16858)
- [EAGLE-3 (NeurIPS 2025)](https://arxiv.org/pdf/2503.01840)

### 官方仓库

- [GitHub: SafeAILab/EAGLE](https://github.com/SafeAILab/EAGLE)

### 社区资源

- [SpecForge](https://github.com/sgl-project/SpecForge) - SGLang集成的EAGLE-3训练工具
- [vLLM EAGLE支持](https://github.com/vllm-project/vllm/pull/6830)

---

## ❓ 常见问题

### Q1: 我应该选择EAGLE-1还是EAGLE-3？

**A:**
- **单GPU实验**：选择EAGLE-1
- **多GPU训练**：选择EAGLE-3
- **生产部署**：推荐EAGLE-3（性能最佳）

### Q2: 训练需要多少资源？

**A:**
- **EAGLE-1**：单张RTX 3090（24GB）可训练7B模型
- **EAGLE-3**：至少2张RTX 3090，推荐8张进行分布式训练
- **训练时间**：7B模型约1-2天（取决于数据量）

### Q3: 我的数据格式不对怎么办？

**A:** 参考[02. 数据准备详解](docs/02-data-preparation.md)中的数据转换指南

### Q4: 训练出错怎么排查？

**A:** 参考[06. 高级主题](docs/06-advanced-topics.md)中的调试指南

---

## 📝 更新日志

- **2025-01-17**：初始版本发布，涵盖EAGLE-1和EAGLE-3完整流程

---

## 🤝 贡献

欢迎对本教程提出改进建议！您可以：

- 报告文档错误
- 提出内容改进建议
- 补充实际使用案例

---

**开始学习**：[01. 入门指南](docs/01-getting-started.md) →
