# 06. 高级主题

> 性能优化、调试指南与自定义扩展

## 目录

- [6.1 性能优化](#61-性能优化)
- [6.2 调试指南](#62-调试指南)
- [6.3 常见问题排查](#63-常见问题排查)
- [6.4 自定义扩展](#64-自定义扩展)

---

## 6.1 性能优化

### 6.1.1 数据加载优化

**问题**：数据加载成为瓶颈

**解决方案**：

```python
# 方案1: 增加num_workers
train_loader = DataLoader(
    traindataset,
    batch_size=4,
    num_workers=8,  # 增加到8个worker
    pin_memory=True  # 启用pin_memory加速GPU传输
)

# 方案2: 预处理数据并缓存
# 在训练前运行预处理脚本
python preprocess_data.py \
    --input raw_data.jsonl \
    --output cached_data/ \
    --num_workers 16

# 方案3: 使用内存映射（大数据集）
from datasets import load_dataset

dataset = load_dataset(
    'json',
    data_files='large_data.jsonl',
    keep_in_memory=False  # 使用内存映射
)
```

**性能对比**：

| 配置 | 数据加载时间 | GPU利用率 |
|------|--------------|-----------|
| num_workers=2 | 3.2s/batch | 60% |
| num_workers=8 | 0.8s/batch | 95% |
| 预处理缓存 | 0.3s/batch | 98% |

### 6.1.2 训练速度优化

**梯度检查点**：

```python
# 在config中启用
train_config = {
    "gradient_checkpoint": True  # 牺牲计算换内存
}
```

**混合精度训练**：

```json
// DeepSpeed配置
{
    "fp16": {
        "enabled": true,
        "initial_scale_power": 16,
        "loss_scale_window": 1000,
        "hysteresis": 2,
        "min_loss_scale": 1
    }
}
```

**注意**：Qwen2等模型建议使用BF16

```python
# 修改训练脚本
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.bfloat16  # 使用bfloat16
)
```

**DeepSpeed ZeRO优化**：

```json
// ZeRO-2: 梯度分片
{
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": true,
        "overlap_comm": true,
        "reduce_scatter": true,
        "contiguous_gradients": true
    }
}

// ZeRO-3: 参数分片（更省显存，但稍慢）
{
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": true
    }
}
```

### 6.1.3 推理优化

**KV Cache优化**：

```python
# 启用静态KV cache（避免动态分配）
class OptimizedEaModel(EaModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache_max_len = 2048
        self.register_buffer(
            "cached_kv",
            torch.zeros(
                2,  # key和value
                self.config.num_hidden_layers,
                1,  # batch size (动态)
                self.config.num_attention_heads,
                self.cache_max_len,
                self.config.hidden_size // self.config.num_attention_heads
            )
        )
```

**批处理优化**：

```python
# 批量推理
def batch_inference(model, tokenizer, prompts, batch_size=8):
    """批量推理提高吞吐量"""
    results = []

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        inputs = tokenizer(
            batch_prompts,
            padding=True,
            return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            outputs = model.eagenerate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=128
            )

        batch_results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        results.extend(batch_results)

    return results
```

---

## 6.2 调试指南

### 6.2.1 调试数据加载

**问题**：DataLoader报错

**调试步骤**：

```python
# 测试单个样本
dataset = build_dataset_rank(tokenizer, "train.jsonl", data_format='chat')
sample = dataset[0]

print("Input IDs shape:", sample['input_ids'].shape)
print("Loss mask sum:", sample['loss_mask'].sum())
print("Attention mask sum:", sample['attention_mask'].sum())

# 可视化
def visualize_sample(sample, tokenizer):
    tokens = tokenizer.convert_ids_to_tokens(sample['input_ids'].squeeze())
    masks = sample['loss_mask'].squeeze().tolist()

    for token, mask in zip(tokens, masks):
        symbol = "✓" if mask == 1 else "✗"
        print(f"{symbol} {token}")

visualize_sample(sample, tokenizer)
```

**常见错误**：

```python
# 错误1: IndexError
# 原因: Loss mask全为0
# 解决: 检查数据格式和loss mask生成逻辑

# 错误2: RuntimeError: CUDA out of memory
# 原因: batch size过大
# 解决: 减小batch size或启用gradient_checkpointing

# 错误3: KeyError: 'conversations'
# 原因: 数据格式不匹配
# 解决: 检查JSON结构
```

### 6.2.2 调试训练循环

**梯度检查**：

```python
# 在训练循环中添加梯度检查
for batch_idx, data in enumerate(train_loader):
    # 前向传播
    plosses, vlosses, acces = model_engine(...)

    # 检查梯度
    loss = sum(plosses)
    loss.backward()

    # 检查梯度是否存在
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            if grad_norm > 100:  # 梯度爆炸
                print(f"⚠️ Large gradient: {name}, norm={grad_norm}")
            if torch.isnan(param.grad).any():  # 梯度为NaN
                print(f"⚠️ NaN gradient: {name}")

    model_engine.step()
```

**Loss监控**：

```python
# 检测loss异常
def monitor_loss(loss_history):
    """检测loss异常"""
    # 检测NaN
    if np.isnan(loss_history[-1]):
        print("⚠️ Loss is NaN!")
        print("Possible causes:")
        print("1. Learning rate too high")
        print("2. Division by zero")
        print("3. Numerical overflow")
        return False

    # 检测爆炸
    if loss_history[-1] > 1000:
        print("⚠️ Loss exploding!")
        print("Possible causes:")
        print("1. Gradient explosion")
        print("2. Incorrect loss scaling")
        return False

    # 检测不下降
    if len(loss_history) > 100:
        recent_mean = np.mean(loss_history[-100:])
        old_mean = np.mean(loss_history[-200:-100])
        if recent_mean >= old_mean:
            print("⚠️ Loss not decreasing!")
            print("Possible causes:")
            print("1. Learning rate too low")
            print("2. Data issues")
            return False

    return True

# 使用
loss_history = []
for epoch in range(num_epochs):
    for batch in train_loader:
        loss = train_step(batch)
        loss_history.append(loss.item())

        if not monitor_loss(loss_history):
            break  # 停止训练
```

### 6.2.3 调试模型输出

**输出质量检查**：

```python
def test_generation_quality(model, tokenizer, test_cases):
    """测试生成质量"""
    for i, (prompt, expected) in enumerate(test_cases):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # 生成
        outputs = model.eagenerate(
            inputs.input_ids,
            max_new_tokens=50,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 计算相似度
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([generated, expected])
        similarity = (vectors[0] * vectors[1].T).toarray()[0, 0]

        print(f"\n=== Test {i+1} ===")
        print(f"Prompt: {prompt}")
        print(f"Expected: {expected}")
        print(f"Generated: {generated}")
        print(f"Similarity: {similarity:.2f}")

        # 检查重复
        if len(set(generated.split())) < len(generated.split()) * 0.5:
            print("⚠️ Warning: High repetition!")

test_generation_quality(model, tokenizer, test_cases)
```

---

## 6.3 常见问题排查

### 6.3.1 数据相关

**Q: Loss mask全为0？**

```python
# 诊断
def diagnose_loss_mask(dataset, tokenizer):
    sample = dataset[0]
    print("Loss mask sum:", sample['loss_mask'].sum().item())
    print("Loss mask shape:", sample['loss_mask'].shape)

    # 找到mask=1的位置
    mask_indices = (sample['loss_mask'].squeeze() == 1).nonzero().squeeze()
    print("Mask=1 positions:", mask_indices.tolist()[:10])  # 前10个

    # 对应的tokens
    tokens = tokenizer.convert_ids_to_tokens(sample['input_ids'].squeeze())
    for idx in mask_indices[:10]:
        print(f"  Position {idx}: {tokens[idx]}")

diagnose_loss_mask(traindataset, tokenizer)
```

**Q: 数据加载很慢？**

```python
# 解决方案1: 预处理
# 将原始数据预处理为.pt文件（EAGLE-1）

# 解决方案2: 使用缓存
from datasets import load_dataset

dataset = load_dataset(
    'json',
    data_files='data.jsonl',
    cache_dir='./cache'  # 指定缓存目录
)

# 解决方案3: 减少预处理
dataset = dataset.map(
    preprocess_function,
    batched=True,
    num_proc=4,  # 减少进程数
    load_from_cache_file=True  # 使用缓存
)
```

### 6.3.2 训练相关

**Q: Loss为NaN？**

```python
# 原因排查
checklist = {
    "Learning rate too high?": lambda: train_config['lr'] > 1e-3,
    "Gradient clipping disabled?": lambda: train_config.get('grad_clip', 0) == 0,
    "Mixed precision issue?": lambda: ds_config.get('fp16', {}).get('enabled', False)
}

for issue, check in checklist.items():
    if check():
        print(f"⚠️ {issue}")

# 解决方案
# 1. 降低学习率
train_config['lr'] = 1e-5

# 2. 启用梯度裁剪
train_config['grad_clip'] = 0.5

# 3. 使用FP32或BF16
ds_config['fp16']['enabled'] = False
ds_config['bf16'] = {'enabled': True}
```

**Q: CUDA Out of Memory？**

```python
# 诊断
def check_gpu_memory():
    import torch
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"Memory Allocated: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
    print(f"Memory Cached: {torch.cuda.memory_reserved() / 1024**3:.1f} GB")

check_gpu_memory()

# 解决方案梯度递减尝试
solutions = [
    "1. 减小batch size",
    "2. 启用gradient checkpointing",
    "3. 使用DeepSpeed ZeRO-3",
    "4. 减小max_len",
    "5. 使用模型并行"
]

for solution in solutions:
    print(solution)
```

### 6.3.3 性能相关

**Q: 训练速度慢？**

```python
# 性能分析
import torch.autograd.profiler as profiler

with profiler.profile(profile_memory=True, record_shapes=True) as prof:
    for batch_idx, data in enumerate(train_loader):
        if batch_idx >= 10:  # 只分析前10个batch
            break
        plosses, vlosses, acces = model_engine(**data)
        loss = sum(plosses)
        model_engine.backward(loss)
        model_engine.step()

# 打印结果
print(prof.key_averages().table(sort_by="cuda_time_total"))

# 查找瓶颈
# - 高cuda_time_total: 考虑模型优化或混合精度
# - 高cpu_time: 考虑数据加载优化
# - 高memory_usage: 考虑激活检查点或ZeRO
```

---

## 6.4 自定义扩展

### 6.4.1 添加新数据格式

```python
# 示例: 添加指令微调格式

def build_instruction_dataset_rank(tokenizer, datapath):
    """构建指令格式数据集"""
    ds = load_dataset('json', data_files=datapath)
    ds = ds['train']

    def preprocess_function(examples):
        new_examples = {
            "input_ids": [],
            "attention_mask": [],
            "loss_mask": []
        }

        for i in range(len(examples['instruction'])):
            # 格式: ### Instruction:\n{instruction}\n\n### Response:\n{response}
            instruction = examples['instruction'][i]
            response = examples['response'][i]

            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids[0]

            # 只在response部分计算loss
            instruction_len = len(tokenizer(f"### Instruction:\n{instruction}\n\n### Response:\n").input_ids)
            loss_mask = torch.zeros_like(input_ids, dtype=torch.float)
            loss_mask[instruction_len:] = 1.0

            attention_mask = torch.ones_like(loss_mask)

            new_examples["input_ids"].append(input_ids[None, :])
            new_examples["loss_mask"].append(loss_mask[None, :])
            new_examples["attention_mask"].append(attention_mask[None, :])

        return new_examples

    ds = ds.map(preprocess_function, batched=True, num_proc=8)
    ds.set_format(type="torch")
    return ds

# 在main.py中添加
def build_dataset_rank(tokenizer, datapath, data_format='chat'):
    if data_format == 'fim':
        return build_fim_dataset_rank(tokenizer, datapath)
    elif data_format == 'instruction':
        return build_instruction_dataset_rank(tokenizer, datapath)
    else:
        return build_chat_dataset_rank(tokenizer, datapath)
```

### 6.4.2 自定义损失函数

```python
# 示例: 添加Focal Loss

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, target, loss_mask):
        """
        Args:
            logits: [batch, seq_len, vocab_size]
            target: [batch, seq_len, vocab_size] (概率分布)
            loss_mask: [batch, seq_len]
        """
        # 计算交叉熵
        log_prob = nn.functional.log_softmax(logits, dim=-1)
        nll_loss = -log_prob * target
        nll_loss = nll_loss.sum(dim=-1)  # [batch, seq_len]

        # 计算focal weight
        prob = nn.functional.softmax(logits, dim=-1)
        prob_target = (prob * target).sum(dim=-1)  # [batch, seq_len]
        focal_weight = self.alpha * (1 - prob_target) ** self.gamma

        # 应用mask和focal weight
        loss = (focal_weight * nll_loss * loss_mask).sum() / (loss_mask.sum() + 1e-5)

        return loss

# 在模型中使用
class Model(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # ...
        self.focal_loss = FocalLoss(alpha=1.0, gamma=2.0)

    def forward(self, ...):
        # ...
        for idx in range(self.length):
            # ...前向传播...

            # 使用focal loss替代交叉熵
            loss = self.focal_loss(logits, target_p, position_mask)
            plosses.append(loss)
```

### 6.4.3 多任务学习

```python
# 示例: 同时训练Chat和FIM

class MultiTaskDataLoader:
    """多任务数据加载器"""
    def __init__(self, chat_dataset, fim_dataset, batch_size, task_ratio=0.5):
        self.chat_loader = DataLoader(chat_dataset, batch_size=batch_size//2)
        self.fim_loader = DataLoader(fim_dataset, batch_size=batch_size//2)
        self.task_ratio = task_ratio

    def __iter__(self):
        for chat_batch, fim_batch in zip(self.chat_loader, self.fim_loader):
            # 添加任务标识
            chat_batch['task_id'] = 0  # Chat任务
            fim_batch['task_id'] = 1   # FIM任务

            # 合并batch
            combined_batch = {}
            for key in chat_batch.keys():
                if key == 'task_id':
                    combined_batch[key] = torch.cat([
                        torch.full((chat_batch['input_ids'].shape[0],), 0),
                        torch.full((fim_batch['input_ids'].shape[0],), 1)
                    ])
                else:
                    combined_batch[key] = torch.cat([chat_batch[key], fim_batch[key]], dim=0)

            yield combined_batch

# 使用
chat_dataset = build_chat_dataset_rank(tokenizer, "chat_data.jsonl")
fim_dataset = build_fim_dataset_rank(tokenizer, "fim_data.jsonl")

train_loader = MultiTaskDataLoader(chat_dataset, fim_dataset, batch_size=8)

for batch in train_loader:
    task_id = batch['task_id']  # [0, 0, 0, 0, 1, 1, 1, 1]
    # 根据task_id使用不同的loss计算逻辑
    ...
```

---

**下一篇**：[07. API参考](07-api-reference.md) →
