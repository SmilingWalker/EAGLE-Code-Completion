# 03. 模型架构深度解析

> EAGLE-1 vs EAGLE-3架构对比与核心组件详解

## 目录

- [3.1 架构对比](#31-架构对比)
- [3.2 EAGLE-1模型架构](#32-eagle-1模型架构)
- [3.3 EAGLE-3模型架构](#33-eagle-3模型架构)
- [3.4 前向传播流程](#34-前向传播流程)
- [3.5 核心组件详解](#35-核心组件详解)

---

## 3.1 架构对比

| 组件 | EAGLE-1 | EAGLE-3 |
|------|---------|---------|
| **输入** | hidden_states (倒数第二层) | 多层hidden_states + input_emb |
| **特征融合** | 单层投影 | fc: hidden_size×3 → hidden_size |
| **处理层** | LlamaDecoderLayer | LlamaDecoderLayeremb |
| **输出** | 单位置预测 | 7个位置预测 |
| **草稿词汇** | 完整词汇表 | 压缩词汇表 (top-N tokens) |
| **训练** | 单GPU | DeepSpeed分布式 |

---

## 3.2 EAGLE-1模型架构

### 3.2.1 模型初始化

代码：`eagle/model/cnets1.py:472-550`

```python
class Model(nn.Module):
    def __init__(self, config, load_emb=True, path=None):
        """
        EAGLE-1模型初始化

        关键参数：
        - hidden_size: 隐藏层维度
        - num_hidden_layers: 层数
        - num_attention_heads: 注意力头数
        """
        super().__init__()

        # [嵌入层] - 从基础模型加载，冻结
        self.embed_tokens = nn.Embedding(...)  # 冻结

        # [EAGLE层] - 唯一训练的组件
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(config) for _ in range(1)
        ])

        # [归一化层]
        self.norm = LlamaRMSNorm(...)

        # [LM Head] - 从基础模型加载，冻结
        self.lm_head = nn.Linear(...)
```

### 3.2.2 前向传播

简化版流程：

```python
def forward(self, hidden_states, input_ids, attention_mask):
    """
    输入：
    - hidden_states: [batch, seq_len, hidden_size] 基础模型倒数第二层
    - input_ids: [batch, seq_len] 输入token ids
    - attention_mask: [batch, seq_len] 注意力掩码

    输出：
    - predicted_hidden_states: [batch, seq_len, hidden_size]
    """

    # [步骤1] 输入投影（可选）
    # hidden_states = self.fc(hidden_states)

    # [步骤2] 通过EAGLE层
    for layer in self.layers:
        hidden_states = layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids
        )

    # [步骤3] 归一化
    hidden_states = self.norm(hidden_states)

    # [步骤4] 预测token
    logits = self.lm_head(hidden_states)

    return logits
```

---

## 3.3 EAGLE-3模型架构

### 3.3.1 模型初始化

代码：`eagle/traineagle3/cnets.py:480-535`

```python
class Model(nn.Module):
    def __init__(self, config, ds_config, training_config, load_head=False, load_emb=True, path=None):
        super().__init__()

        # [关键配置]
        self.length = 7  # 预测7个位置
        self.draft_vocab_size = config.draft_vocab_size  # 草稿词汇表大小

        # [目标模型] - 冻结的基础模型
        self.target_model = LlamaForCausalLM.from_pretrained(path)
        for param in self.target_model.parameters():
            param.requires_grad = False  # 冻结

        # [特征融合层]
        self.fc = nn.Linear(self.hidden_size * 3, self.hidden_size, bias=False)
        # 输入：前3层hidden_states拼接
        # 输出：融合后的hidden_states

        # [嵌入层] - 冻结
        self.embed_tokens = nn.Embedding(...)
        for param in self.embed_tokens.parameters():
            param.requires_grad = False

        # [EAGLE处理层]
        self.midlayer = LlamaDecoderLayeremb(config)

        # [归一化层]
        self.norm = LlamaRMSNorm(...)

        # [草稿LM Head] - 训练
        self.lm_head = nn.Linear(config.hidden_size, config.draft_vocab_size)

        # [草稿词汇表映射]
        self.d2t = ...  # draft到target的映射
        self.t2d = ...  # target到draft的映射
```

### 3.3.2 草稿词汇表构建

代码：`eagle/traineagle3/cnets.py:536-688`

```python
def scandata(self, datapath, tokenizerpath):
    """
    扫描数据集构建草稿词汇表

    策略：选择频率最高的N个token作为草稿词汇表

    Args:
        datapath: 训练数据路径
        tokenizerpath: tokenizer路径
    """

    if not os.path.exists("cache.pt"):
        # [步骤1] 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizerpath)

        # [步骤2] 加载数据集
        dataset = load_dataset('json', data_files=datapath)

        # [步骤3] 统计token频率
        # 使用多进程并行处理
        token_dict = Counter()
        for sample in dataset:
            for token_id in sample['input_ids']:
                token_dict[token_id] += 1

        # [步骤4] 选择top-N tokens
        N = self.draft_vocab_size  # 例如：4096
        top_N = token_dict.most_common(N)

        # [步骤5] 构建映射
        used_tokens = [token for token, _ in top_N]
        used_tokens.sort()

        # draft到target: draft_id → target_id的偏移量
        d2t = [used_tokens[i] - i for i in range(len(used_tokens))]

        # target到draft: 判断token是否在草稿词汇表中
        t2d = [i in used_tokens for i in range(self.vocab_size)]

        # [步骤6] 保存缓存
        cache = {"d2t": d2t, "t2d": t2d}
        torch.save(cache, "cache.pt")
    else:
        cache = torch.load("cache.pt")
        d2t = cache["d2t"]
        t2d = cache["t2d"]

    # 注册为buffer（不参与训练）
    self.register_buffer("d2t", torch.tensor(d2t))
    self.register_buffer("t2d", torch.tensor(t2d))
```

**草稿词汇表优势**：
- 减少输出维度（32000 → 4096）
- 加速训练和推理
- 覆盖高频token（通常占90%以上）

---

## 3.4 前向传播流程

### 3.4.1 完整流程图

```
输入文本
    ↓
[目标模型] frozen
    ├─ Layer 0 → hidden_states_0 (浅层特征)
    ├─ Layer 1 → hidden_states_1 (中层特征)
    └─ Layer 2 → hidden_states_2 (深层特征)
    ↓
[特征拼接] cat((hs0, hs1, hs2), dim=-1)
    ↓
[特征融合] fc: hidden_size*3 → hidden_size
    ↓
[多位置预测循环] (for idx in range(7))
    ├─ [嵌入] embed_tokens(input_ids)
    ├─ [EAGLE层] midlayer(emb, hidden_states)
    ├─ [归一化] norm(hidden_states)
    ├─ [预测] lm_head(hidden_states)
    ├─ [计算loss] 位置加权
    ├─ [更新] input_ids = padding(input_ids)
    └─ 重复7次
    ↓
输出: 7个位置的loss和accuracy
```

### 3.4.2 代码详解

代码：`eagle/traineagle3/cnets.py:733-868`

```python
def forward(self, input_ids, attention_mask, loss_mask):
    """
    EAGLE-3前向传播

    Args:
        input_ids: [batch, seq_len]
        attention_mask: [batch, seq_len]
        loss_mask: [batch, seq_len]

    Returns:
        plosses: 7个位置的perplexity loss
        vlosses: 7个位置的value loss（未使用）
        acces: 7个位置的accuracy
    """

    # [步骤1] 数据准备 - 提取多层hidden_states
    hidden_states, target, loss_mask, input_ids = self.dataprepare(
        input_ids, attention_mask, loss_mask
    )

    # dataprepare内部 (line 714-731):
    # - 通过target_model提取前3层hidden_states
    # - 拼接: [hs0, hs1, hs2] → hidden_size * 3
    # - 准备target logits（用于计算loss）

    # [步骤2] 特征融合
    hidden_states = self.fc(hidden_states)  # [batch, seq_len, hidden_size*3] → [batch, seq_len, hidden_size]

    # [步骤3] 准备位置ids和attention mask
    position_ids = torch.arange(seq_length, device=device)
    attention_mask = self._prepare_decoder_attention_mask(...)

    # [步骤4] 多位置预测循环
    plosses = []
    acces = []
    cache_hidden = [[], []]  # KV cache

    for idx in range(self.length):  # self.length = 7
        last = idx == self.length - 1  # 是否是最后一个位置

        # [4.1] 获取embeddings
        inputs_embeds = self.embed_tokens(input_ids)  # [batch, seq_len, hidden_size]

        # [4.2] 通过EAGLE层（支持梯度检查点）
        if self.gradient_checkpointing and self.training:
            layer_outputs, cache_hidden = torch.utils.checkpoint.checkpoint(
                self.midlayer,
                inputs_embeds,
                hidden_states,
                cache_hidden,
                attention_mask,
                position_ids,
            )
        else:
            layer_outputs, cache_hidden = self.midlayer(
                input_emb=inputs_embeds,
                hidden_states=hidden_states,
                cache_hidden=cache_hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )

        # [4.3] 归一化
        hidden_states_out = self.norm(layer_outputs[0])

        # [4.4] 预测logits
        logits = self.lm_head(hidden_states_out)  # [batch, seq_len, draft_vocab_size]

        # [4.5] 准备计算loss
        with torch.no_grad():
            # 获取target模型预测
            target_head = target  # [batch, seq_len, vocab_size]
            target_max_token = target_head.argmax(-1)  # [batch, seq_len]

            # 应用草稿词汇表mask
            target_mask = self.t2d[target_max_token]  # [batch, seq_len]
            position_mask = target_mask * loss_mask.squeeze(-1)  # [batch, seq_len]

            # 计算target概率分布
            target_head_draft = target_head[..., self.t2d]  # 映射到草稿词汇表
            target_p = nn.Softmax(dim=2)(target_head_draft)  # [batch, seq_len, draft_vocab_size]

        # [4.6] 计算perplexity loss
        out_logp = nn.LogSoftmax(dim=2)(logits)  # [batch, seq_len, draft_vocab_size]
        plogp = target_p * out_logp  # [batch, seq_len, draft_vocab_size]

        # 位置加权：只在mask=1的位置计算loss
        loss = -torch.sum(position_mask.unsqueeze(-1) * plogp, dim=2).mean()
        plosses.append(loss)

        # [4.7] 计算accuracy
        with torch.no_grad():
            acc = ((logits.argmax(-1) == target_p.argmax(-1)) * position_mask).sum()
            acc = acc / (loss_mask.sum() + 1e-6)
            acces.append(acc)

        # [4.8] 为下一个位置更新输入
        if not last:
            input_ids = padding(input_ids, left=False)  # 向右移位
            target = padding(target, left=False)
            loss_mask = padding(loss_mask, left=False)

        # 更新hidden_states供下一个位置使用
        hidden_states = hidden_states_out

    return plosses, vlosses, acces
```

### 3.4.3 位置加权策略

训练脚本中的应用：`eagle/traineagle3/main.py:391-393`

```python
# 位置权重：0.8^i
ploss_weight = [0.8 ** i for i in range(len(plosses))]
# [0.8^0=1.0, 0.8^1=0.8, 0.8^2=0.64, ...]

# 加权求和
ploss = sum([ploss_weight[i] * plosses[i] for i in range(len(plosses))])
```

**为什么使用位置加权？**

1. **更近的位置更重要**：第1个位置比第7个位置更容易预测准确
2. **平衡训练**：避免模型只关注后面的位置
3. **经验调优**：0.8的衰减率是实验得出的最佳值

---

## 3.5 核心组件详解

### 3.5.1 特征融合层

```python
self.fc = nn.Linear(self.hidden_size * 3, self.hidden_size, bias=False)
```

**作用**：将前3层hidden_states融合

**数学表示**：

```
h_fused = W @ [h_0; h_1; h_2] + b

其中：
- h_0: 第0层输出 (浅层语义)
- h_1: 第1层输出 (中层语义)
- h_2: 第2层输出 (深层语义)
- W: [hidden_size, hidden_size*3] 权重矩阵
- [;]: 拼接操作
```

**为何选择前3层？**

- **Layer 0**: 局部语法特征
- **Layer 1**: 中层语义特征
- **Layer 2**: 高层抽象特征
- 组合起来提供多尺度信息

### 3.5.2 EAGLE处理层

```python
self.midlayer = LlamaDecoderLayeremb(config)
```

继承自标准的`LlamaDecoderLayer`，但修改了输入：

**标准LlamaDecoderLayer**：
```python
def forward(self, hidden_states, ...):
    # 仅接受hidden_states
```

**LlamaDecoderLayeremb** (EAGLE-3)：
```python
def forward(self, input_emb, hidden_states, cache_hidden, ...):
    # 接受两个输入：
    # - input_emb: token embeddings [batch, seq_len, hidden_size]
    # - hidden_states: 融合后的特征 [batch, seq_len, hidden_size]
```

**内部处理**：
1. 拼接两个输入：`[input_emb; hidden_states]`
2. 通过自注意力层
3. 通过MLP层
4. 返回更新后的hidden_states + KV cache

### 3.5.3 KV Cache

```python
cache_hidden = [[], []]  # [key_cache, value_cache]
```

**作用**：缓存之前计算的Key和Value，避免重复计算

**流程**：

```
位置1:
  - 计算K1, V1
  - 存入cache

位置2:
  - 从cache读取K1, V1
  - 计算K2, V2
  - 拼接: [K1, K2], [V1, V2]

位置3:
  - 从cache读取K1, K2, V1, V2
  - 计算K3, V3
  - 拼接: [K1, K2, K3], [V1, V2, V3]
```

**优势**：
- 加速推理：避免重复计算
- 支持长序列：内存效率高

---

## 3.6 架构演进总结

| 版本 | 发布时间 | 核心创新 | 加速比 |
|------|----------|----------|--------|
| **EAGLE-1** | 2024.01 | 单层特征外推 | 3x |
| **EAGLE-2** | 2024.06 | 动态草稿树 | 4x |
| **EAGLE-3** | 2025.01 | 多层特征融合 + 多位置预测 | 5.6x |

**趋势**：
1. 更多语义层次（1层 → 3层）
2. 更多预测位置（1个 → 7个）
3. 更高效的训练（单GPU → DeepSpeed）
4. 更好的性能（3x → 5.6x）

---

**下一篇**：[04. 训练流程详解](04-training-pipeline.md) →
