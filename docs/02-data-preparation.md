# 02. 数据准备详解

> Chat格式、FIM格式与预处理流程逐行解析

## 目录

- [2.1 数据格式概述](#21-数据格式概述)
- [2.2 Chat格式详解](#22-chat格式详解)
- [2.3 FIM格式详解](#23-fim格式详解)
- [2.4 数据预处理流程](#24-数据预处理流程)
- [2.5 Loss Mask生成策略](#25-loss-mask生成策略)
- [2.6 数据增强技术](#26-数据增强技术)
- [2.7 自定义数据适配](#27-自定义数据适配)

---

## 2.1 数据格式概述

EAGLE-3支持两种主要的数据格式：

| 格式 | 用途 | 训练脚本 | 特殊tokens |
|------|------|----------|------------|
| **Chat** | 对话训练 | `--data-format chat` | 系统提示词模板 |
| **FIM** | 代码补全 | `--data-format fim` | `<fim_prefix>`, `<fim_middle>`, `<fim_suffix>` |

### 2.1.1 格式选择指南

**使用Chat格式**：
- 通用对话模型
- 指令跟随任务
- 问答系统
- 多轮对话

**使用FIM格式**：
- 代码补全
- 文本填充
- 中间内容生成

---

## 2.2 Chat格式详解

### 2.2.1 数据结构

Chat格式采用类ShareGPT结构：

```json
{
  "id": "unique_id_123",
  "conversations": [
    {"from": "human", "value": "你好，请介绍一下你自己"},
    {"from": "gpt", "value": "我是AI助手，可以帮你解答问题..."},
    {"from": "human", "value": "你能做什么？"},
    {"from": "gpt", "value": "我可以帮助你..."}
  ]
}
```

**字段说明**：

- `id`: 唯一标识符（字符串）
- `conversations`: 对话列表（数组）
  - `from`: 角色标识（`"human"` 或 `"gpt"`）
  - `value`: 消息内容（字符串）

### 2.2.2 Chat格式预处理

代码位置：`eagle/traineagle3/main.py:145-261`

#### 逐行解析

```python
def build_chat_dataset_rank(tokenizer, datapath):
    """
    构建Chat格式数据集

    Args:
        tokenizer: Hugging Face tokenizer
        datapath: 数据文件路径（.jsonl）

    Returns:
        处理后的数据集，包含input_ids, attention_mask, loss_mask
    """
    # [149-153] 加载数据集
    ds = load_dataset('json', data_files=datapath)
    ds = ds['train']
    ds = ds.shuffle(seed=42)  # 固定随机种子，保证可复现
    ds1 = ds
    original_columns1 = ds1.column_names  # 保存原始列名，后续删除
    num_proc = 8  # 并行处理的进程数

    # [156-161] 定义预处理函数
    def preprocess_function(examples):
        new_examples = {
            "attention_mask": [],
            "input_ids": [],
            "loss_mask": []
        }

        # [162] 遍历所有样本
        for i in range(len(examples['id'])):

            # [163-166] 添加系统提示词
            messages = [
                {"role": "system",
                 "content": "You are a helpful, respectful and honest assistant..."}
            ]
            # 这个系统提示词定义了助手的角色和安全约束

            # [167-168] 定义角色映射
            convroles = ["user", "assistant"]  # LLaMA-3模板使用的角色名
            roles = {"human": "user", "gpt": "assistant"}  # 数据中的角色名映射

            # [169-174] 获取对话并验证
            source = examples['conversations'][i]
            if not source:
                continue  # 跳过空对话
            if roles[source[0]["from"]] != "user":
                # 如果第一条不是用户消息，跳过第一条
                source = source[1:]

            # [175-182] 构建消息列表
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                assert role == convroles[j % 2], f"{i}"  # 确保角色交替出现
                messages.append(
                    {"role": role, "content": sentence["value"]}
                )

            # [183-187] 应用chat模板
            conversation = tokenizer.apply_chat_template(
                messages,
                tokenize=False,  # 返回字符串而非token ids
                add_generation_prompt=False,
            )

            # [189-190] 设置pad token
            if not tokenizer.pad_token_id:
                tokenizer.pad_token_id = tokenizer.unk_token_id

            # [192-196] Tokenization
            input_ids = tokenizer(
                conversation,
                return_tensors="pt",
                add_special_tokens=False,
            ).input_ids[0]

            # [197-199] 过滤过长序列
            if len(input_ids) > train_config["max_len"]:
                continue  # 跳过超过max_len的样本

            # [200] 初始化loss mask（全部为1，表示所有位置都计算loss）
            loss_mask = torch.ones_like(input_ids)

            # [203-241] 计算loss mask
            # 关键逻辑：只在assistant回复上计算loss，忽略user指令

            sep = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            sep2 = "<|eot_id|><|start_header_id|>user<|end_header_id|>"

            # [205-211] 分割对话轮次
            total_len = len(input_ids)
            turns = conversation.split(sep2)

            # [210] 重建第一轮（包含system prompt）
            turns[1] = turns[0] + sep2 + turns[1]
            turns = turns[1:]  # 去掉冗余的第一项

            # [213-214] 初始位置
            cur_len = 1
            loss_mask[:cur_len] = 0  # 第一个token不计算loss

            # [215-240] 遍历每一轮对话
            for i, turn in enumerate(turns):
                if turn == "":
                    break

                # [218-219] 计算当前轮次的token长度
                turn_len = len(tokenizer(turn).input_ids)

                # [220-232] 分离user和assistant部分
                parts = turn.split(sep)
                if len(parts) != 2:
                    break  # 格式错误，跳过
                parts[0] += sep  # 重新拼接sep

                # [225] 计算指令长度
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

                # [227-231] Mask掉user指令
                if i == 0:
                    # 第一轮：mask掉system prompt + user message
                    loss_mask[cur_len: cur_len + instruction_len - 2] = 0
                else:
                    # 后续轮次：mask掉user message
                    loss_mask[cur_len - 3: cur_len + instruction_len + 1] = 0

                # [232-234] 更新当前位置
                cur_len += turn_len
                if i != 0:
                    cur_len += 3  # 考虑特殊token

            # [241] Mask掉最后一个token
            loss_mask[cur_len:] = 0
            attention_mask = torch.ones_like(loss_mask)

            # [244-247] 添加到结果
            new_examples["input_ids"].append(input_ids[None, :])
            new_examples["loss_mask"].append(loss_mask[None, :])
            new_examples["attention_mask"].append(attention_mask[None, :])

        return new_examples

    # [251-257] 应用预处理
    ds1 = ds1.map(
        preprocess_function,
        batched=True,  # 批处理模式
        num_proc=num_proc,  # 多进程并行
        remove_columns=original_columns1,  # 删除原始列
        load_from_cache_file=False  # 不使用缓存
    )

    ds1.set_format(type="torch")
    return ds1
```

### 2.2.3 Loss Mask可视化

```
原始对话：
User: 你好
Assistant: 你好！
User: 你能做什么？
Assistant: 我可以帮助你...

Token序列：
[<bos>] [User] [你] [好] [<sep>] [Assistant] [你] [好] [！] [User] [你] [能] [做] [什么] [？] [<sep>] [Assistant] [我] [可] [以] [帮] [助] [你] [...] [<eos>]

Loss Mask：
[0] [0] [0] [0] [0] [0] [1] [1] [1] [0] [0] [0] [0] [0] [0] [0] [0] [0] [0] [1] [1] [1] [1] [1] [1] [1] [1] [0] ...
 ↑                    ↑   ↑ ↑ ↑                    ↑ ↑ ↑
 忽略User           忽略sep  Assistant回复    忽略User      Assistant回复
```

**关键点**：只有assistant的回复部分loss_mask=1，其他位置都是0。

---

## 2.3 FIM格式详解

### 2.3.1 数据结构

FIM（Fill-In-Middle）格式用于代码补全任务：

```json
{
  "prefix": "def hello():\n    ",
  "middle": "print('hello world')",
  "suffix": "\n    return"
}
```

**字段说明**：

- `prefix`: 代码前缀（函数签名、注释等）
- `middle`: 需要生成的中间部分（函数体）
- `suffix`: 代码后缀（后续代码）

### 2.3.2 FIM格式预处理

代码位置：`eagle/traineagle3/main.py:58-142`

#### 逐行解析

```python
def build_fim_dataset_rank(tokenizer, datapath):
    """
    构建FIM格式数据集

    Args:
        tokenizer: Hugging Face tokenizer
        datapath: 数据文件路径（.jsonl）

    Returns:
        处理后的数据集，包含input_ids, attention_mask, loss_mask
    """
    # [71-76] 加载数据集
    ds = load_dataset('json', data_files=datapath)
    ds = ds['train']
    ds = ds.shuffle(seed=42)
    ds1 = ds
    original_columns1 = ds1.column_names
    num_proc = 8

    # [78-81] 定义FIM特殊tokens
    FIM_PREFIX = "<fim_prefix>"
    FIM_MIDDLE = "<fim_middle>"
    FIM_SUFFIX = "<fim_suffix>"
    # 注意：这些不是特殊token，而是作为普通文本处理

    # [83-131] 定义预处理函数
    def preprocess_fim_function(examples):
        new_examples = {
            "attention_mask": [],
            "input_ids": [],
            "loss_mask": []
        }

        # [90] 遍历所有样本
        for i in range(len(examples.get('prefix', []))):
            # [91-93] 提取prefix, middle, suffix
            prefix = examples['prefix'][i]
            middle = examples['middle'][i]
            suffix = examples['suffix'][i]

            # [95-96] 构建FIM prompt
            fim_prompt = f"{FIM_PREFIX}{prefix}{FIM_MIDDLE}{middle}{FIM_SUFFIX}{suffix}"
            # 结果示例：
            # "<fim_prefix>def hello():\n    <fim_middle>print('hello')<fim_suffix>\n    return"

            # [98-103] Tokenize完整prompt
            input_ids = tokenizer(
                fim_prompt,
                return_tensors="pt",
                add_special_tokens=False,
            ).input_ids[0]

            # [105-108] Tokenize各部分（用于计算loss mask）
            prefix_ids = tokenizer(prefix, add_special_tokens=False).input_ids
            middle_ids = tokenizer(middle, add_special_tokens=False).input_ids
            suffix_ids = tokenizer(suffix, add_special_tokens=False).input_ids

            # [110-112] 过滤过长序列
            if len(input_ids) > train_config["max_len"]:
                continue

            # [114-123] 计算loss mask
            # 关键：只在middle部分计算loss

            loss_mask = torch.zeros_like(input_ids, dtype=torch.float)

            # [117-120] 计算middle部分的起止位置
            # middle部分在完整input_ids中的位置：
            # | prefix | <fim_prefix> | <fim_middle> | middle | <fim_suffix> | suffix |
            #  len(prefix_ids) + len(FIM_PREFIX) + len(FIM_MIDDLE) -> middle开始

            middle_start = (len(prefix_ids) +
                          len(tokenizer(FIM_PREFIX).input_ids) +
                          len(tokenizer(FIM_MIDDLE).input_ids))
            middle_end = middle_start + len(middle_ids)

            # [122-123] 只在middle部分设置loss_mask=1
            loss_mask[middle_start:middle_end] = 1.0

            # [125] 创建attention mask
            attention_mask = torch.ones_like(loss_mask)

            # [127-129] 添加到结果
            new_examples["input_ids"].append(input_ids[None, :])
            new_examples["loss_mask"].append(loss_mask[None, :])
            new_examples["attention_mask"].append(attention_mask[None, :])

        return new_examples

    # [133-139] 应用预处理
    ds1 = ds1.map(
        preprocess_fim_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns1,
        load_from_cache_file=False
    )

    ds1.set_format(type="torch")
    return ds1
```

### 2.3.3 FIM Loss Mask可视化

```
原始代码：
def hello():          <- prefix
    print('hello')     <- middle (需要生成)
    return             <- suffix

Token序列：
[<fim_prefix>] [def] [hello] [(] [:)] [\n] [<fim_middle>] [print] [(] ['hello'] [)] [<fim_suffix>] [\n] [return]

Loss Mask：
[0] [0] [0] [0] [0] [0] [0] [0] [1] [1] [1] [1] [1] [0] [0] [0] [0]
 ↑                              ↑           ↑ ↑ ↑ ↑ ↑              ↑
 prefix部分               middle部分    只在这些位置计算loss     suffix部分
```

---

## 2.4 数据预处理流程

### 2.4.1 EAGLE-1 vs EAGLE-3

| 特性 | EAGLE-1 | EAGLE-3 |
|------|---------|---------|
| **数据格式** | 预处理.pt文件 | 原始JSONL文件 |
| **预处理时机** | 训练前 | 训练时 |
| **包含内容** | hidden_states + input_ids | 仅input_ids |
| **数据增强** | ✅ 支持噪声 | ❌ 不支持 |

### 2.4.2 EAGLE-1数据预处理

**准备步骤**（在训练前完成）：

```python
# 需要预先运行基础模型提取hidden states
# 这个脚本不在当前仓库中，通常需要单独编写

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载模型
model = AutoModelForCausalLM.from_pretrained("path/to/base/model")
tokenizer = AutoTokenizer.from_pretrained("path/to/base/model")

# 提取hidden states
def extract_hidden_states(text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-2]  # 倒数第二层
    return {
        "input_ids": inputs.input_ids,
        "hidden_state": hidden_states,
        "loss_mask": ...  # 根据格式生成
    }

# 保存为.pt文件
torch.save(extract_hidden_states("your text"), "data.pt")
```

**EAGLE-1数据加载**：

代码位置：`eagle/train/main.py:134-175`

```python
class CustomDataset(Dataset):
    def __init__(self, datapath, transform=None):
        self.data = datapath  # .pt文件列表
        self.transform = transform  # 数据增强（可选）

    def __getitem__(self, index):
        # [144] 加载预处理数据
        data = torch.load(self.data[index])
        new_data = {}

        # [146-148] 截断到max_len
        hidden_state = data['hidden_state'][:train_config["max_len"]][None, :]
        input_ids = data['input_ids'][:train_config["max_len"]][None, :]
        loss_mask = data["loss_mask"][:train_config["max_len"]][None, :]

        # [157-159] 构建目标（下一位置的hidden_states）
        input_ids_target = input_ids[:, 1:]  # 向前移一位
        zeropadding = torch.tensor([[0]])
        input_ids_target = torch.cat((input_ids_target, zeropadding), dim=1)

        # [161-163] 目标hidden_states（下一层）
        target = hidden_state[:, 1:, :]
        zeropadding = torch.zeros(1, 1, target.shape[2])
        target = torch.cat((target, zeropadding), dim=1)

        # [165-169] 组装数据
        new_data["attention_mask"] = attention_mask
        new_data["loss_mask"] = loss_mask
        new_data["target"] = target
        new_data["hidden_state_big"] = hidden_state
        new_data["input_ids"] = input_ids_target

        # [172-173] 应用数据增强（如果启用）
        if self.transform:
            new_data = self.transform(new_data)

        return new_data
```

### 2.4.3 EAGLE-3数据预处理

EAGLE-3在训练时实时处理数据：

```python
# main.py:317-318
tokenizer = AutoTokenizer.from_pretrained(args.basepath)
traindataset = build_dataset_rank(tokenizer, args.trainpath, args.data_format)
testdataset = build_dataset_rank(tokenizer, args.testpath, args.data_format)

# build_dataset_rank根据data_format选择处理函数
def build_dataset_rank(tokenizer, datapath, data_format='chat'):
    if data_format == 'fim':
        return build_fim_dataset_rank(tokenizer, datapath)
    else:
        return build_chat_dataset_rank(tokenizer, datapath)
```

---

## 2.5 Loss Mask生成策略

### 2.5.1 为什么需要Loss Mask？

训练时我们只想在特定位置计算loss：

**Chat格式**：
- ❌ 不计算：user指令（模型不应学会生成指令）
- ✅ 计算：assistant回复（学习生成回复）

**FIM格式**：
- ❌ 不计算：prefix和suffix（上下文）
- ✅ 计算：middle（需要生成的内容）

### 2.5.2 Loss Mask的作用机制

在训练循环中应用：

```python
# eagle/traineagle3/main.py:386-393
plosses, vlosses, acces = model_engine(
    input_ids=data["input_ids"].to(rank),
    attention_mask=data["attention_mask"].to(rank),
    loss_mask=data["loss_mask"],  # 传入loss_mask
)

# 在模型内部（cnets.py）：
# loss = torch.sum(loss_mask * plogp) / (loss_mask.sum() + 1e-5)
# 只有loss_mask=1的位置才会计算loss
```

### 2.5.3 Loss Mask调试

**验证Loss Mask正确性**：

```python
def visualize_loss_mask(sample, tokenizer):
    """可视化loss mask"""
    input_ids = sample['input_ids'][0]
    loss_mask = sample['loss_mask'][0]

    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    for token, mask in zip(tokens, loss_mask):
        symbol = "✓" if mask == 1 else "✗"
        print(f"{symbol} {token}")

# 使用示例
sample = traindataset[0]
visualize_loss_mask(sample, tokenizer)
```

预期输出（Chat格式）：

```
✓ <s>
✗ User
✗ :
✗ 你
✗ 好
✗ <
✓ Assistant
✓ :
✓ 你
✓ 好
✓ ！
...
```

---

## 2.6 数据增强技术

### 2.6.1 EAGLE-1数据增强

EAGLE-1支持在hidden_states上添加噪声：

代码位置：`eagle/train/main.py:109-131`

#### 高斯噪声 (Gaussian Noise)

```python
class AddGaussianNoise:
    """添加高斯噪声到hidden states"""

    def __init__(self, mean=0.0, std=0.0):
        self.mean = mean    # 噪声均值
        self.std = std      # 噪声标准差

    def __call__(self, data):
        tensor = data["hidden_state_big"]
        # [116] 生成高斯噪声
        noise = torch.randn(tensor.size()) * self.std + self.mean
        # [117] 添加噪声
        noisy_tensor = tensor + noise
        data["hidden_state_big"] = noisy_tensor
        return data
```

**数学表示**：

```
h_noisy = h + ε, where ε ~ N(mean, std²)
```

#### 均匀噪声 (Uniform Noise)

```python
class AddUniformNoise:
    """添加均匀噪声到hidden states"""

    def __init__(self, std=0.0):
        self.std = std

    def __call__(self, data):
        tensor = data["hidden_state_big"]
        # [128] 生成均匀噪声：范围 [-std/2, std/2]
        # 乘以512/seq_len是为了标准化不同序列长度
        noise = (torch.rand_like(tensor) - 0.5) * self.std * 512 / tensor.shape[1]
        # [129] 添加噪声
        noisy_tensor = tensor + noise
        data["hidden_state_big"] = noisy_tensor
        return data
```

**数学表示**：

```
h_noisy = h + ε, where ε ~ U(-std/2, std/2)
```

### 2.6.2 使用数据增强

```python
# eagle/train/main.py:296-302
if train_config["data_noise"]:
    if train_config["noise"] == "uniform":
        aug = AddUniformNoise(std=train_config["std"])  # 默认std=0.2
    else:
        aug = AddGaussianNoise(mean=train_config["mean"], std=train_config["std"])  # 默认mean=0.0, std=0.2
else:
    aug = None

# 应用到数据集
traindataset = CustomDataset(traindatapath, transform=aug)
```

### 2.6.3 数据增强的作用

1. **防止过拟合**：增加训练数据的多样性
2. **提高泛化**：使模型对特征扰动更鲁棒
3. **模拟推理差异**：推理时的hidden states可能与训练时不同

**何时使用**：

- 数据量较少时（< 100K样本）
- 观察到过拟合时（train loss远低于test loss）
- 需要提高模型鲁棒性时

---

## 2.7 自定义数据适配

### 2.7.1 准备自定义数据

#### Step 1: 确定数据格式

**对话数据 → Chat格式**：

```python
import json

def convert_to_chat_format(source_data):
    """将自定义对话数据转换为Chat格式"""
    output = []
    for item in source_data:
        output.append({
            "id": str(item['id']),
            "conversations": [
                {"from": "human", "value": item['question']},
                {"from": "gpt", "value": item['answer']}
            ]
        })
    return output

# 保存为JSONL
with open('custom_data.jsonl', 'w') as f:
    for item in converted_data:
        f.write(json.dumps(item) + '\n')
```

**代码数据 → FIM格式**：

```python
def convert_to_fim_format(source_code):
    """将代码转换为FIM格式"""
    # 方法1: 随机分割
    import random
    lines = source_code.split('\n')
    split_point = random.randint(1, len(lines)-1)

    prefix = '\n'.join(lines[:split_point])
    middle = ""  # 需要生成
    suffix = '\n'.join(lines[split_point:])

    # 方法2: 基于函数签名
    # 提取函数签名作为prefix，函数体作为middle
    # ...更复杂的逻辑

    return {"prefix": prefix, "middle": middle, "suffix": suffix}
```

#### Step 2: 数据质量检查

```python
def validate_chat_data(jsonl_file):
    """验证Chat格式数据"""
    errors = []
    with open(jsonl_file) as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line)
                # 检查必需字段
                if 'id' not in data:
                    errors.append(f"Line {i}: missing 'id'")
                if 'conversations' not in data:
                    errors.append(f"Line {i}: missing 'conversations'")
                # 检查对话格式
                for j, conv in enumerate(data['conversations']):
                    if 'from' not in conv or 'value' not in conv:
                        errors.append(f"Line {i}, conv {j}: missing fields")
                    if conv['from'] not in ['human', 'gpt']:
                        errors.append(f"Line {i}, conv {j}: invalid role")
            except json.JSONDecodeError:
                errors.append(f"Line {i}: invalid JSON")

    return errors

def validate_fim_data(jsonl_file):
    """验证FIM格式数据"""
    errors = []
    with open(jsonl_file) as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line)
                for field in ['prefix', 'middle', 'suffix']:
                    if field not in data:
                        errors.append(f"Line {i}: missing '{field}'")
            except json.JSONDecodeError:
                errors.append(f"Line {i}: invalid JSON")

    return errors
```

#### Step 3: 数据集划分

```python
def split_dataset(jsonl_file, train_ratio=0.95):
    """划分训练集和测试集"""
    with open(jsonl_file) as f:
        lines = f.readlines()

    import random
    random.shuffle(lines)

    split_point = int(len(lines) * train_ratio)
    train_lines = lines[:split_point]
    test_lines = lines[split_point:]

    with open('train.jsonl', 'w') as f:
        f.writelines(train_lines)

    with open('test.jsonl', 'w') as f:
        f.writelines(test_lines)

    print(f"Train: {len(train_lines)}, Test: {len(test_lines)}")
```

### 2.7.2 训练自定义模型

#### EAGLE-1训练

```bash
# 准备预处理数据（需要额外脚本）
python preprocess_data.py \
    --input custom_data.jsonl \
    --output processed_data/ \
    --base-model /path/to/base/model

# 训练
cd eagle/train
python main.py \
    --basepath /path/to/base/model \
    --tmpdir processed_data/ \
    --cpdir checkpoints/ \
    --lr 3e-5 \
    --bs 4
```

#### EAGLE-3训练

```bash
cd eagle/traineagle3

# Chat格式
python main.py \
    --basepath /path/to/base/model \
    --trainpath train.jsonl \
    --testpath test.jsonl \
    --savedir checkpoints/ \
    --data-format chat \
    --deepspeed_config ds_config.json

# FIM格式
python main.py \
    --basepath /path/to/base/model \
    --trainpath train.jsonl \
    --testpath test.jsonl \
    --savedir checkpoints/ \
    --data-format fim \
    --deepspeed_config ds_config.json
```

### 2.7.3 验证数据加载

```python
# test_data_loading.py
from transformers import AutoTokenizer
from eagle.traineagle3.main import build_dataset_rank

tokenizer = AutoTokenizer.from_pretrained("/path/to/base/model")

# 测试Chat格式
dataset = build_dataset_rank(tokenizer, "train.jsonl", data_format='chat')
print(f"Dataset size: {len(dataset)}")
sample = dataset[0]
print(f"Input IDs shape: {sample['input_ids'].shape}")
print(f"Loss mask sum: {sample['loss_mask'].sum()}")  # 应该 > 0

# 测试FIM格式
dataset = build_dataset_rank(tokenizer, "train.jsonl", data_format='fim')
sample = dataset[0]
print(f"Loss mask sum: {sample['loss_mask'].sum()}")  # 应该 > 0
```

---

## 2.8 最佳实践

### 2.8.1 数据准备清单

- [ ] 确定数据格式（Chat vs FIM）
- [ ] 转换数据为JSONL格式
- [ ] 验证数据格式正确性
- [ ] 划分训练集和测试集
- [ ] 检查数据量（建议> 10K样本）
- [ ] 测试数据加载
- [ ] 检查loss mask正确性

### 2.8.2 常见问题

**Q: 数据量需要多少？**

**A:** 取决于模型大小：
- 7B模型：至少50K样本
- 13B模型：至少100K样本
- 70B模型：至少500K样本

**Q: 可以混合使用Chat和FIM数据吗？**

**A:** 不建议。格式差异较大，建议分别训练。

**Q: 如何处理长文本？**

**A:** 有两种策略：
1. 截断到`max_len`（EAGLE默认）
2. 分段处理（需要修改代码）

**Q: Loss mask全为0怎么办？**

**A:** 检查：
1. 数据格式是否正确
2. 特殊token位置计算是否正确
3. tokenizer是否正确

---

**下一篇**：[03. 模型架构深度解析](03-model-architecture.md) →
