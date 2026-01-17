# 04. 训练流程详解

> EAGLE-1和EAGLE-3训练脚本逐行解析

## 目录

- [4.1 EAGLE-1训练流程](#41-eagle-1训练流程)
- [4.2 EAGLE-3训练流程](#42-eagle-3训练流程)
- [4.3 损失函数详解](#43-损失函数详解)
- [4.4 优化器与调度](#44-优化器与调度)
- [4.5 检查点管理](#45-检查点管理)

---

## 4.1 EAGLE-1训练流程

### 4.1.1 初始化阶段

代码：`eagle/train/main.py:1-98`

```python
# [步骤1] 参数解析 (line 3-11)
parser = argparse.ArgumentParser()
parser.add_argument('--basepath', type=str, default='...')  # 基础模型路径
parser.add_argument('--lr', type=float, default=3e-5)       # 学习率
parser.add_argument('--bs', type=int, default=4)            # 批大小
parser.add_argument('--tmpdir', type=str, default='0')      # 数据目录
parser.add_argument('--cpdir', type=str, default='0')       # 检查点目录
args = parser.parse_args()

# [步骤2] 训练配置 (line 13-41)
train_config = {
    "lr": args.lr,                              # 学习率
    "bs": args.bs,                              # 批大小
    "gradient_accumulation_steps": 1,           # 梯度累积步数
    "num_epochs": 20,                           # 训练轮数
    "num_warmup_steps": 2000,                   # Warmup步数
    "total_steps": 800000,                      # 总步数
    "p_w": 0.1,                                 # Perplexity loss权重
    "v_w": 1.0,                                 # Value loss权重
    "data_noise": True,                         # 是否添加噪声
    "noise": "uniform",                         # 噪声类型
    "std": 0.2,                                 # 噪声标准差
    "grad_clip": 0.5,                           # 梯度裁剪
    "save_freq": 5,                             # 保存频率
}

# [步骤3] 加载LM Head (line 72-97)
head = torch.nn.Linear(baseconfig.hidden_size, baseconfig.vocab_size)
# 从基础模型加载权重
head.weight.data = load_lm_head_weight(args.basepath)
head.eval()  # 设置为评估模式
for param in head.parameters():
    param.requires_grad = False  # 冻结

# [步骤4] 初始化模型 (line 321-322)
model = Model(config, load_emb=True, path=args.basepath)
model = model.to(device)

# [步骤5] 初始化优化器和调度器 (line 328-338)
criterion = nn.SmoothL1Loss(reduction="none")
optimizer = optim.AdamW(model.parameters(), lr=3e-5, betas=(0.9, 0.95))
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=2000,
    num_training_steps=800000
)
```

### 4.1.2 训练循环

代码：`eagle/train/main.py:341-401`

```python
for epoch in range(num_epochs + 1):  # line 341
    model.train()  # line 347

    for batch_idx, data in enumerate(train_loader):  # line 348
        # [1] 数据移到设备 (line 351)
        data = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in data.items()}

        # [2] 梯度管理 (line 353-357)
        should_step = (batch_idx + 1) % gradient_accumulation_steps == 0
        if should_step:
            optimizer.zero_grad()

        # [3] 前向传播 (line 359)
        predict = model(
            data["hidden_states"],      # [batch, seq_len, hidden_size]
            input_ids=data["input_ids"],  # [batch, seq_len]
            attention_mask=data["attention_mask"]  # [batch, seq_len]
        )

        # [4] 计算目标 (line 360-364)
        with torch.no_grad():
            target_head = head(data["target"])  # 目标模型的logits
            target_p = nn.Softmax(dim=2)(target_head)  # 概率分布
            target_p = target_p.detach()  # 分离梯度
        loss_mask = data["loss_mask"][:, :, None]  # [batch, seq_len, 1]

        # [5] 计算损失 (line 365-366)
        vloss, ploss, out_head = compute_loss(
            data["target"], target_p, predict, loss_mask
        )
        loss = v_w * vloss + p_w * ploss  # 组合损失
        loss = loss / gradient_accumulation_steps  # 梯度累积

        # [6] 反向传播 (line 370)
        loss.backward()

        # [7] 优化器步进 (line 372-376)
        if should_step:
            torch.nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            optimizer.step()
            if is_warmup:
                scheduler.step()

        # [8] 计算准确率 (line 378-389)
        with torch.no_grad():
            _, predicted = torch.max(out_head, 2)
            _, target = torch.max(target_head, 2)
            correct = ((predicted == target) * loss_mask.squeeze()).sum()
            total = loss_mask.sum()

        # [9] 日志记录 (line 391-395)
        if total != 0:
            logdict = {
                "train/lr": optimizer.param_groups[0]["lr"],
                "train/vloss": vloss.item(),
                "train/ploss": ploss.item(),
                "train/loss": loss.item(),
                "train/acc": correct.item() / total.item()
            }
            wandb.log(logdict)
```

### 4.1.3 评估与保存

代码：`eagle/train/main.py:415-483`

```python
# [评估] 每5个epoch评估一次 (line 415)
if (epoch + 1) % train_config["save_freq"] == 0:
    model.eval()

    # 测试循环 (line 424-453)
    for batch_idx, data in enumerate(tqdm(test_loader)):
        with torch.no_grad():
            # 前向传播
            predict = model(...)
            # 计算loss和accuracy

    # [保存检查点] (line 475-483)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': epoch_loss,
    }
    if is_warmup:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    torch.save(checkpoint, f"{args.cpdir}/state_{epoch}.pt")
```

---

## 4.2 EAGLE-3训练流程

### 4.2.1 初始化阶段

代码：`eagle/traineagle3/main.py:1-45`

```python
# [步骤1] 参数解析 (line 4-16)
parser = argparse.ArgumentParser()
parser.add_argument('--basepath', type=str)        # 基础模型路径
parser.add_argument('--trainpath', type=str)       # 训练数据路径
parser.add_argument('--testpath', type=str)        # 测试数据路径
parser.add_argument('--savedir', type=str)         # 保存目录
parser.add_argument('--data-format', type=str,     # 数据格式
                    choices=['chat', 'fim'])
parser = deepspeed.add_config_arguments(parser)   # 添加DeepSpeed参数
args = parser.parse_args()

# [步骤2] DeepSpeed配置 (line 20-31)
deepspeed_config = args.deepspeed_config
with open(deepspeed_config) as f:
    ds_config = json.load(f)
train_config = {
    "bs": ds_config["train_micro_batch_size_per_gpu"],  # 每GPU批大小
    "num_epochs": 40,
    "max_len": 2048,
    "gradient_checkpoint": True,
    "data_format": args.data_format
}

# [步骤3] 数据加载 (line 316-318)
tokenizer = AutoTokenizer.from_pretrained(args.basepath)
traindataset = build_dataset_rank(tokenizer, args.trainpath, args.data_format)
testdataset = build_dataset_rank(tokenizer, args.testpath, args.data_format)

# [步骤4] 模型初始化 (line 320-322)
config = EConfig.from_pretrained("config.json")
model = Model(config, ds_config, train_config, path=args.basepath,
              load_emb=True, load_head=True)
model.scandata(args.trainpath, args.basepath)  # 构建草稿词汇表

# [步骤5] DeepSpeed初始化 (line 328-335)
model_engine, optimizer, _, _ = deepspeed.initialize(
    args=args,
    model=model,
    model_parameters=model.parameters()
)

# [步骤6] 分布式采样器 (line 344-351)
sampler = DistributedSampler(
    traindataset,
    num_replicas=world_size,      # 总GPU数
    rank=global_rank,             # 当前GPU rank
    shuffle=True                  # 训练集打乱
)
train_loader = DataLoader(
    traindataset,
    batch_size=train_config["bs"],
    sampler=sampler,
    num_workers=4,
    collate_fn=DataCollatorWithPadding()
)
```

### 4.2.2 训练循环

代码：`eagle/traineagle3/main.py:373-423`

```python
for epoch in range(start_epoch, num_epochs):  # line 373
    # [1] 设置epoch（确保每个epoch数据shuffle不同） (line 374)
    train_sampler.set_epoch(epoch + 1)
    model.train()

    # 初始化epoch指标
    epoch_acces = [[] for _ in range(model.length)]  # 7个位置
    epoch_plosses = [[] for _ in range(model.length)]

    # [2] 训练循环 (line 382-406)
    for batch_idx, data in enumerate(tqdm(train_loader)):
        model.zero_grad()  # 清空梯度 (line 384)

        # 前向传播 (line 386-389)
        plosses, vlosses, acces = model_engine(
            input_ids=data["input_ids"].to(rank),
            attention_mask=data["attention_mask"].to(rank),
            loss_mask=data["loss_mask"],
        )
        # 返回：
        # - plosses: [7] 每个位置的perplexity loss
        # - vlosses: [7] 每个位置的value loss（未使用）
        # - acces: [7] 每个位置的accuracy

        # [3] 计算加权损失 (line 391-393)
        ploss_weight = [0.8 ** i for i in range(len(plosses))]
        # [1.0, 0.8, 0.64, 0.512, 0.410, 0.328, 0.262]
        ploss = sum([ploss_weight[i] * plosses[i]
                     for i in range(len(plosses))])
        loss = ploss  # EAGLE-3只使用perplexity loss

        # [4] 反向传播和优化 (line 394-396)
        model_engine.backward(loss)  # DeepSpeed自动处理梯度同步
        model_engine.step()

        # [5] 日志记录 (line 398-406)
        if global_rank == 0:  # 只在rank 0记录
            logdict = {"train/lr": optimizer.param_groups[0]["lr"]}
            for i in range(len(plosses)):
                logdict[f"train/ploss_{i}"] = plosses[i].item()
                logdict[f"train/acc_{i}"] = acces[i]
            wandb.log(logdict)

        # 累积epoch指标
        epoch_acces = [epoch_acces[i] + [acces[i]]
                       for i in range(len(acces))]
        epoch_plosses = [epoch_plosses[i] + [plosses[i].item()]
                         for i in range(len(plosses))]

    # [6] Epoch级别的指标聚合 (line 408-422)
    for i in range(len(epoch_acces)):
        # 多GPU平均 (line 409-411)
        acc_i = torch.tensor(epoch_acces[i]).cuda().mean()
        deepspeed.comm.all_reduce(acc_i, op=deepspeed.comm.ReduceOp.AVG)
        acc_i = acc_i.item()

        if global_rank == 0:
            wandb.log({f"train/epochacc_{i}": acc_i})
            print(f"Train Epoch [{epoch + 1}/{num_epochs}], "
                  f"position {i}, Acc: {acc_i:.2f}")

    for i in range(len(epoch_plosses)):
        loss_i = torch.tensor(epoch_plosses[i]).cuda().mean()
        deepspeed.comm.all_reduce(loss_i, op=deepspeed.comm.ReduceOp.AVG)
        loss_i = loss_i.item()

        if global_rank == 0:
            wandb.log({f"train/epochploss_{i}": loss_i})
            print(f"Train Epoch [{epoch + 1}/{num_epochs}], "
                  f"position {i}, pLoss: {loss_i:.2f}")
```

### 4.2.3 评估与保存

代码：`eagle/traineagle3/main.py:427-456`

```python
# [评估循环] (line 427-450)
for batch_idx, data in enumerate(tqdm(test_loader)):
    with torch.no_grad():
        plosses, vlosses, acces = model_engine(...)
        # 累积指标

# 多GPU聚合
for i in range(len(epoch_acces)):
    acc_i = torch.tensor(epoch_acces[i]).cuda().mean()
    deepspeed.comm.all_reduce(acc_i, op=deepspeed.comm.ReduceOp.AVG)

# [保存检查点] (line 454-456)
# 每个epoch保存16位模型
model_engine.save_16bit_model(
    f"{args.savedir}/state_{epoch}",
    exclude_frozen_parameters=True  # 不保存冻结的参数
)

# 每10个epoch保存完整检查点
if epoch % 10 == 0:
    deepspeed.DeepSpeedEngine.save_checkpoint(
        model_engine,
        save_dir=f"{args.savedir}/state_{epoch}"
    )
```

---

## 4.3 损失函数详解

### 4.3.1 EAGLE-1损失函数

代码：`eagle/train/main.py:231-238`

```python
def compute_loss(target, target_p, predict, loss_mask):
    """
    计算EAGLE-1的组合损失

    Args:
        target: [batch, seq_len, hidden_size] 目标hidden_states
        target_p: [batch, seq_len, vocab_size] 目标概率分布
        predict: [batch, seq_len, hidden_size] 预测hidden_states
        loss_mask: [batch, seq_len, 1] loss掩码

    Returns:
        vloss: Value loss（特征重建损失）
        ploss: Perplexity loss（概率匹配损失）
        out_head: [batch, seq_len, vocab_size] 预测的logits
    """
    # [1] 通过LM Head获取预测logits (line 232)
    out_head = head(predict)  # [batch, seq_len, vocab_size]

    # [2] 计算Perplexity Loss (line 233-235)
    out_logp = nn.LogSoftmax(dim=2)(out_head)  # log(p(x))
    plogp = target_p * out_logp  # p_target * log(p_predict)
    ploss = -torch.sum(loss_mask * plogp, 2) / (loss_mask.sum() + 1e-5)
    # 交叉熵：H(p_target, p_predict) = -Σ p_target * log(p_predict)

    # [3] 计算Value Loss (line 236-237)
    vloss = criterion(predict, target)  # SmoothL1Loss
    vloss = torch.sum(torch.mean(loss_mask * vloss, 2)) / (loss_mask.sum() + 1e-5)
    # Smooth L1 Loss: |x| if |x|>1 else 0.5*x²

    return vloss, ploss, out_head
```

**损失权重**：

```python
loss = v_w * vloss + p_w * ploss
# loss = 1.0 * vloss + 0.1 * ploss
```

**设计理念**：
- `v_w = 1.0`: 特征重建更重要，确保hidden_states准确
- `p_w = 0.1`: 概率匹配辅助，帮助token预测

### 4.3.2 EAGLE-3损失函数

代码：`eagle/traineagle3/cnets.py:846-859`

```python
# 在forward循环中，对每个位置计算loss
for idx in range(self.length):  # 7个位置
    # ...前向传播...

    # [1] 计算Perplexity Loss (line 851-856)
    logits = self.lm_head(hidden_states_out)  # [batch, seq_len, draft_vocab_size]
    out_logp = nn.LogSoftmax(dim=2)(logits)
    plogp = target_p * out_logp  # [batch, seq_len, draft_vocab_size]

    # 应用position_mask（只在特定位置计算loss）
    loss = -torch.sum(position_mask.unsqueeze(-1) * plogp, dim=2).mean()
    plosses.append(loss)

    # [2] 计算Accuracy (line 857-859)
    acc = ((logits.argmax(-1) == target_p.argmax(-1)) * position_mask).sum()
    acc = acc / (loss_mask.sum() + 1e-6)
    acces.append(acc)
```

**位置加权**（在训练脚本中）：

```python
ploss_weight = [0.8 ** i for i in range(len(plosses))]
ploss = sum([ploss_weight[i] * plosses[i] for i in range(len(plosses))])
```

**为何EAGLE-3只用Perplexity Loss？**
- Value loss需要目标hidden_states，增加计算开销
- Perplexity loss已经足够训练token预测
- 简化训练流程，提高效率

---

## 4.4 优化器与调度

### 4.4.1 优化器配置

**EAGLE-1**：
```python
optimizer = optim.AdamW(
    model.parameters(),
    lr=3e-5,         # 学习率
    betas=(0.9, 0.95),  # Adam动量参数
    eps=1e-8,
    weight_decay=0.01
)
```

**EAGLE-3**：
```python
# 通过DeepSpeed配置
{
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 5e-5,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "weight_decay": 0.01
        }
    }
}
```

### 4.4.2 学习率调度

**EAGLE-1**：线性warmup + 线性衰减

```python
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=2000,    # 前2000步线性增长
    num_training_steps=800000  # 总步数800K
)
```

**学习率变化**：

```
lr_max = 3e-5

Warmup阶段 (0-2000步):
  lr = lr_max * (step / 2000)

衰减阶段 (2000-800000步):
  lr = lr_max * (1 - (step - 2000) / (800000 - 2000))
```

**EAGLE-3**：DeepSpeed WarmupDecayLR

```json
{
    "scheduler": {
        "type": "WarmupDecayLR",
        "params": {
            "total_num_steps": 800000,
            "warmup_min_lr": 0,
            "warmup_max_lr": 5e-5,
            "warmup_num_steps": 12000
        }
    }
}
```

### 4.4.3 梯度管理

**梯度裁剪**：

```python
# EAGLE-1
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)

# EAGLE-3 (DeepSpeed配置)
{
    "gradient_clipping": 0.5
}
```

**梯度累积**：

```python
# EAGLE-1
should_step = (batch_idx + 1) % gradient_accumulation_steps == 0
if should_step:
    optimizer.step()
    optimizer.zero_grad()

# EAGLE-3 (DeepSpeed配置)
{
    "gradient_accumulation_steps": 2
}
```

---

## 4.5 检查点管理

### 4.5.1 EAGLE-1检查点

**保存**：

```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': epoch_loss,
    'scheduler_state_dict': scheduler.state_dict()  # 可选
}
torch.save(checkpoint, f"{cpdir}/state_{epoch}.pt")
```

**加载**：

```python
checkpoint = torch.load(f"{cpdir}/state_{epoch}.pt")
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

### 4.5.2 EAGLE-3检查点

**16位模型**（每个epoch）：

```python
model_engine.save_16bit_model(
    f"{savedir}/state_{epoch}",
    exclude_frozen_parameters=True  # 不保存frozen参数
)
```

**完整检查点**（每10个epoch）：

```python
deepspeed.DeepSpeedEngine.save_checkpoint(
    model_engine,
    save_dir=f"{savedir}/state_{epoch}"
)
```

**自动恢复**：

```python
def find_max_state_with_file(directory):
    """找到最新的检查点"""
    max_epoch = -1
    for subdir in os.listdir(directory):
        match = re.match(r"state_(\d+)", subdir)
        if match and os.path.exists(f"{directory}/{subdir}/zero_to_fp32.py"):
            max_epoch = max(max_epoch, int(match.group(1)))
    return f"{directory}/state_{max_epoch}", max_epoch + 1

checkpoint_path, start_epoch = find_max_state_with_file(args.savedir)
if checkpoint_path:
    model_engine.load_checkpoint(checkpoint_path)
```

---

## 4.6 训练最佳实践

### 4.6.1 超参数选择

| 参数 | 7B模型 | 13B模型 | 70B模型 |
|------|--------|---------|---------|
| 学习率 | 3e-5 | 3e-5 | 2e-5 |
| 批大小 | 4-8 | 2-4 | 1-2 |
| 梯度累积 | 1 | 2 | 4 |
| Warmup | 2K | 5K | 10K |
| 总步数 | 80K | 160K | 320K |

### 4.6.2 训练监控

**关键指标**：

```python
# WandB仪表盘
{
    "train/lr": 学习率
    "train/ploss_0": 第0位置的loss
    "train/acc_0": 第0位置的accuracy
    "train/epochacc_0": Epoch平均accuracy
    "test/epochacc_0": 测试集accuracy
}
```

**健康检查**：

```python
# Loss应该：
# 1. 稳定下降
# 2. 没有剧烈波动
# 3. train和test差距不大

# Accuracy应该：
# 1. 稳定上升
# 2. 位置0 > 位置1 > ... > 位置6
# 3. 最终达到60-80%（取决于数据）
```

### 4.6.3 常见问题

**Q: Loss不下降？**

**A:** 检查：
1. 学习率是否过低/过高
2. 数据是否正确加载
3. Loss mask是否正确
4. 模型是否正确初始化

**Q: 训练速度慢？**

**A:** 优化：
1. 增加`num_workers`
2. 启用`gradient_checkpointing`
3. 使用混合精度训练
4. 增大batch size

**Q: 显存不足？**

**A:** 解决：
1. 减小`train_micro_batch_size_per_gpu`
2. 增加`gradient_accumulation_steps`
3. 启用DeepSpeed ZeRO-2/3
4. 启用`gradient_checkpointing`

---

**下一篇**：[05. 实战教程](05-practical-tutorial.md) →
