# 01. 入门指南

> 理解投机解码原理、EAGLE架构与环境配置

## 目录

- [1.1 投机解码理论](#11-投机解码理论)
- [1.2 EAGLE架构概述](#12-eagle架构概述)
- [1.3 环境配置](#13-环境配置)
- [1.4 快速开始](#14-快速开始)

---

## 1.1 投机解码理论

### 1.1.1 为什么需要投机解码？

大型语言模型（LLM）的推理速度受限于自回归生成的特性：

```
传统自回归生成：
token_1 → token_2 → token_3 → token_4 → ...
  ↓         ↓         ↓         ↓
 串行处理，每个token都需要完整的前向传播
```

**核心瓶颈**：生成512个token需要运行512次前向传播。

### 1.1.2 投机解码的基本思想

投机解码通过"小模型猜测、大模型验证"的思路加速：

```
投机解码流程：
1. 草稿模型快速生成 k 个候选token
2. 目标模型并行验证这 k 个token
3. 接受正确的token，拒绝错误的token
4. 从第一个错误token重新开始
```

**数学原理**：

假设：
- 草稿模型生成速度：s₁ tokens/s
- 目标模型验证速度：s₂ tokens/s
- 验证k个token的时间 ≈ 生成1个token的时间
- 接受率：α（正确token的比例）

**加速比** ≈ (1 + α(k-1)) / (1/α + (k-1))

当α=0.8, k=10时，理论上可获得约3-4倍加速。

### 1.1.3 传统投机解码的问题

传统方法（如Medusa）面临以下挑战：

1. **训练独立的草稿模型**：需要大量额外训练资源
2. **性能下降**：草稿模型可能改变输出分布
3. **架构限制**：通常只能用于特定的模型系列

### 1.1.4 EAGLE的创新点

EAGLE (Extrapolation Algorithm for Greater Language-model Efficiency) 采取了不同的思路：

**核心洞察**：基础模型的中间层已经包含预测下一层的所有信息

```
传统方法：
基础模型(hidden_states) → 独立训练的草稿模型 → token

EAGLE方法：
基础模型(hidden_states) → 轻量级外推层 → hidden_states' → token
```

**三大创新**：

1. **特征外推**：直接预测下一层的hidden states，而非token
2. **参数共享**：复用基础模型的embedding和大部分参数
3. **理论保证**：可证明与原始模型的输出分布一致

---

## 1.2 EAGLE架构概述

### 1.2.1 EAGLE-1 基础架构

```
输入文本
    ↓
[基础模型] (frozen)
    ↓
hidden_states (倒数第二层)
    ↓
[EAGLE层] (trainable)
    ├─ 特征投影
    ├─ 注意力层
    └─ MLP层
    ↓
predicted_hidden_states
    ↓
[LM Head] (frozen)
    ↓
候选tokens
```

**关键特性**：
- 仅训练EAGLE层（约0.2-0.4B参数）
- 基础模型和LM Head保持冻结
- 支持树采样(tree-based sampling)

### 1.2.2 EAGLE-3 增强架构

EAGLE-3引入了多层次语义特征融合：

```
输入文本
    ↓
[基础模型] (frozen)
    ├─ 浅层特征 (low-level semantics)
    ├─ 中层特征 (mid-level semantics)
    └─ 深层特征 (high-level semantics)
    ↓
[特征融合层]
    ↓
[EAGLE层] (trainable)
    ├─ 双输入处理 (input_emb + hidden_states)
    ├─ 草稿词汇表映射
    └─ 多位置预测头
    ↓
多位置预测 (7个位置)
```

**关键改进**：
1. **双输入融合**：同时使用token embedding和hidden states
2. **多位置预测**：一次性预测未来7个位置的hidden states
3. **位置加权损失**：更近的位置权重更高 (0.8^i)
4. **训练时测试**：模拟推理时的树采样过程

### 1.2.3 版本对比

| 特性 | EAGLE-1 | EAGLE-3 |
|------|---------|---------|
| **训练脚本** | `eagle/train/main.py` | `eagle/traineagle3/main.py` |
| **模型定义** | `eagle/model/cnets1.py` | `eagle/traineagle3/cnets.py` |
| **分布式训练** | ❌ 单GPU | ✅ DeepSpeed |
| **数据格式** | 预处理.pt文件 | 原始Chat/FIM数据 |
| **损失函数** | vloss + ploss (固定权重) | 位置加权ploss |
| **加速比** | 3x | 5.6x |
| **推荐场景** | 快速实验 | 生产部署 |

---

## 1.3 环境配置

### 1.3.1 硬件要求

#### EAGLE-1 训练

```
最低配置：
- GPU: RTX 3090 (24GB)
- 内存: 32GB
- 存储: 100GB

推荐配置：
- GPU: RTX 3090 / A100 (40GB)
- 内存: 64GB
- 存储: 200GB SSD
```

#### EAGLE-3 训练

```
最低配置：
- GPU: 2x RTX 3090 (48GB总显存)
- 内存: 64GB
- 存储: 200GB

推荐配置：
- GPU: 8x RTX 3090 / A100
- 内存: 128GB
- 存储: 500GB NVMe SSD
```

### 1.3.2 软件依赖

#### 基础依赖

```bash
# 创建虚拟环境
python -m venv eagle_env
source eagle_env/bin/activate  # Linux/Mac
# 或 eagle_env\Scripts\activate  # Windows

# Python版本
Python >= 3.8

# 核心依赖
pip install torch>=2.0.0
pip install transformers>=4.53.1
pip install safetensors
pip install datasets
pip install accelerate
```

#### EAGLE-1 特定依赖

```bash
pip install fschat  # FastChat for conversation templates
pip install tqdm
pip install numpy
```

#### EAGLE-3 特定依赖

```bash
pip install deepspeed
pip install fschat
pip install tqdm
pip install numpy
```

#### 可选依赖

```bash
# 训练监控
pip install wandb

# ROCm支持 (AMD GPU)
# 查看 requirements-rocm.txt
```

### 1.3.3 安装验证

创建验证脚本 `verify_installation.py`：

```python
import torch
import transformers
import deepspeed  # EAGLE-3需要

print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"CUDA版本: {torch.version.cuda}")
print(f"GPU数量: {torch.cuda.device_count()}")
print(f"Transformers版本: {transformers.__version__}")
print(f"DeepSpeed版本: {deepspeed.__version__}")

# 测试GPU
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  显存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
```

运行验证：

```bash
python verify_installation.py
```

预期输出：

```
PyTorch版本: 2.0.1
CUDA可用: True
CUDA版本: 11.8
GPU数量: 2
GPU 0: NVIDIA GeForce RTX 3090
  显存: 24.0 GB
GPU 1: NVIDIA GeForce RTX 3090
  显存: 24.0 GB
```

---

## 1.4 快速开始

### 1.4.1 克隆仓库

```bash
git clone https://github.com/SafeAILab/EAGLE.git
cd EAGLE
```

### 1.4.2 准备基础模型

```bash
# 使用Hugging Face模型
# 例如：Llama-3.1-8B-Instruct

# 方法1: 使用transformers自动下载
export BASE_MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"

# 方法2: 从本地路径加载
export BASE_MODEL_PATH="/path/to/your/model"
```

### 1.4.3 准备训练数据

**Chat格式示例** (`train_data.jsonl`)：

```jsonl
{"id": "1", "conversations": [{"from": "human", "value": "你好"}, {"from": "gpt", "value": "你好！有什么可以帮助你的吗？"}]}
{"id": "2", "conversations": [{"from": "human", "value": "解释什么是深度学习"}, {"from": "gpt", "value": "深度学习是机器学习的一个分支..."}]}
```

**FIM格式示例** (`fim_data.jsonl`)：

```jsonl
{"prefix": "def hello():\n    ", "middle": "print('hello')", "suffix": "\n    return"}
{"prefix": "def add(a, b):\n    ", "middle": "return a + b", "suffix": "\n\nresult = add(1, 2)"}
```

### 1.4.4 EAGLE-1 快速训练

```bash
cd eagle/train

python main.py \
    --basepath /path/to/base/model \
    --tmpdir /path/to/processed/data \
    --cpdir /path/to/checkpoints \
    --lr 3e-5 \
    --bs 4 \
    --num_epochs 20
```

**关键参数说明**：

- `--basepath`: 基础模型路径
- `--tmpdir`: 预处理数据目录（.pt文件）
- `--cpdir`: 检查点保存目录
- `--lr`: 学习率 (默认: 3e-5)
- `--bs`: 批大小 (默认: 4)
- `--num_epochs`: 训练轮数 (通过config配置，默认20)

### 1.4.5 EAGLE-3 快速训练

#### Step 1: 创建DeepSpeed配置 (`ds_config.json`)

```json
{
    "train_micro_batch_size_per_gpu": 1,
    "gradient_accumulation_steps": 2,
    "fp16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 2
    },
    "gradient_clipping": 0.5,
    "steps_per_print": 100
}
```

#### Step 2: 运行训练

```bash
cd eagle/traineagle3

# 单GPU（测试用）
python main.py \
    --basepath /path/to/base/model \
    --trainpath /path/to/train.jsonl \
    --testpath /path/to/test.jsonl \
    --savedir /path/to/checkpoints \
    --data-format chat \
    --deepspeed_config ds_config.json

# 多GPU（推荐）
deepspeed --num_gpus=2 main.py \
    --basepath /path/to/base/model \
    --trainpath /path/to/train.jsonl \
    --testpath /path/to/test.jsonl \
    --savedir /path/to/checkpoints \
    --data-format fim \
    --deepspeed_config ds_config.json
```

**关键参数说明**：

- `--trainpath`: 训练数据路径（JSONL格式）
- `--testpath`: 测试数据路径
- `--savedir`: 模型保存目录
- `--data-format`: 数据格式（`chat` 或 `fim`）
- `--deepspeed_config`: DeepSpeed配置文件路径

### 1.4.6 监控训练进度

#### 使用WandB

```bash
# 登录WandB
wandb login

# 训练时会自动上传指标
# 在脚本中查看：
# eagle/train/main.py (line 70)
# eagle/traineagle3/main.py (line 340)
```

#### 本地监控

训练脚本会输出类似以下信息：

```
Now training epoch 0
Data format: chat
Train Epoch [1/20], position 0, Acc: 45.23
Train Epoch [1/20], position 0, pLoss: 2.34
...
Test Epoch [1/20], position 0, Acc: 43.12
```

**关键指标**：

- `Acc`: 准确率（每个位置）
- `pLoss`: 困惑度损失（位置加权）
- `train/lr`: 当前学习率

### 1.4.7 检查点管理

#### EAGLE-1 检查点

```
checkpoints/
├── state_0.pt    # epoch 0
├── state_5.pt    # epoch 5
├── state_10.pt   # epoch 10
├── state_15.pt   # epoch 15
└── state_20.pt   # epoch 20 (最终)
```

保存频率：通过`train_config["save_freq"]`控制（默认5个epoch）

#### EAGLE-3 检查点

```
checkpoints/
├── state_0/
│   ├── zero_to_fp32.py
│   └── model_weights.pt
├── state_10/
│   └── ...  # 完整检查点（每10个epoch）
└── state_20/
    └── ...  # 最终检查点
```

保存策略：
- 每个epoch保存16位模型
- 每10个epoch保存完整DeepSpeed检查点

---

## 1.5 下一步

现在您已经了解了EAGLE的基本原理和环境配置，建议继续学习：

**→ [02. 数据准备详解](02-data-preparation.md)**

深入了解：
- Chat格式和FIM格式的区别
- 数据预处理流程
- Loss mask的生成策略
- 自定义数据适配

---

## 1.6 常见问题

### Q1: 为什么EAGLE-3需要DeepSpeed？

**A:** EAGLE-3使用了更复杂的模型和更大的batch size，DeepSpeed的ZeRO优化可以：
- 分片模型参数到多个GPU
- 减少单GPU显存占用
- 支持更大的模型训练

### Q2: 我可以在CPU上训练吗？

**A:** 理论上可以，但极度不推荐：
- 训练速度会慢100-1000倍
- 可能需要数月才能完成
- 建议至少使用GPU

### Q3: 如何选择学习率？

**A:** 推荐配置：
- EAGLE-1: 3e-5 (默认)
- EAGLE-3: 通过DeepSpeed配置（通常5e-5）
- 如果loss震荡，降低学习率
- 如果loss不下降，提高学习率

### Q4: 训练需要多长时间？

**A:** 取决于多个因素：

| 模型大小 | GPU配置 | 数据量 | 预计时间 |
|----------|---------|--------|----------|
| 7B | 1x RTX 3090 | 100K样本 | 1-2天 |
| 13B | 2x RTX 3090 | 100K样本 | 2-3天 |
| 70B | 8x A100 | 100K样本 | 5-7天 |

---

**下一篇**：[02. 数据准备详解](02-data-preparation.md) →
