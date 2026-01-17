# 07. API参考

> 完整的函数签名、参数说明和使用示例

## 目录

- [7.1 模型API](#71-模型api)
- [7.2 数据处理API](#72-数据处理api)
- [7.3 训练API](#73-训练api)
- [7.4 推理API](#74-推理api)

---

## 7.1 模型API

### 7.1.1 EaModel

**位置**: `eagle/model/ea_model.py`

#### `from_pretrained`

```python
@classmethod
def from_pretrained(
    cls,
    base_model_path: str,
    ea_model_path: str,
    torch_dtype: torch.dtype = torch.float16,
    low_cpu_mem_usage: bool = True,
    device_map: str = "auto",
    use_eagle3: bool = True,
    total_token: int = -1,
    **kwargs
) -> EaModel
```

**参数说明**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `base_model_path` | str | 必填 | 基础模型路径（Hugging Face格式） |
| `ea_model_path` | str | 必填 | EAGLE模型检查点路径 |
| `torch_dtype` | dtype | float16 | 模型权重的数据类型 |
| `low_cpu_mem_usage` | bool | True | 是否使用低CPU内存模式 |
| `device_map` | str | "auto" | 设备映射策略 |
| `use_eagle3` | bool | True | 是否使用EAGLE-3模型 |
| `total_token` | int | -1 | 总token数（-1表示自动配置） |

**返回值**：
- `EaModel`: 加载的模型实例

**示例**：

```python
from eagle.model.ea_model import EaModel
import torch

# 加载EAGLE-3模型
model = EaModel.from_pretrained(
    base_model_path="meta-llama/Llama-3.1-8B-Instruct",
    ea_model_path="yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
    torch_dtype=torch.float16,
    use_eagle3=True
)
model.eval()
```

---

#### `eagenerate`

```python
def eagenerate(
    self,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    max_new_tokens: int = 512,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = -1,
    do_sample: bool = False,
    **kwargs
) -> torch.Tensor
```

**参数说明**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `input_ids` | Tensor | 必填 | 输入token ids [batch, seq_len] |
| `attention_mask` | Tensor | None | 注意力掩码 [batch, seq_len] |
| `max_new_tokens` | int | 512 | 最大生成token数 |
| `temperature` | float | 1.0 | 采样温度（<1更确定性，>1更随机） |
| `top_p` | float | 1.0 | nucleus sampling参数 |
| `top_k` | int | -1 | top-k采样参数（-1表示禁用） |
| `do_sample` | bool | False | 是否使用采样（False=贪婪解码） |

**返回值**：
- `torch.Tensor`: 生成的token ids [batch, seq_len + max_new_tokens]

**示例**：

```python
# 贪婪解码
output = model.eagenerate(
    input_ids,
    max_new_tokens=128,
    do_sample=False
)

# 采样解码
output = model.eagenerate(
    input_ids,
    max_new_tokens=128,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)
```

---

#### `naivegenerate`

```python
def naivegenerate(
    self,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    max_new_tokens: int = 512,
    **kwargs
) -> torch.Tensor
```

**说明**：标准自回归生成（无EAGLE加速），用于baseline对比

---

### 7.1.2 EAGLE-1 Model

**位置**: `eagle/model/cnets1.py`

#### `Model.__init__`

```python
def __init__(
    self,
    config: EConfig,
    load_emb: bool = True,
    path: Optional[str] = None
)
```

**参数说明**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `config` | EConfig | 必填 | 模型配置对象 |
| `load_emb` | bool | True | 是否从基础模型加载embedding |
| `path` | str | None | 基础模型路径（load_emb=True时需要） |

---

#### `Model.forward`

```python
def forward(
    self,
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor
) -> torch.Tensor
```

**参数说明**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `hidden_states` | Tensor | 基础模型的hidden states [batch, seq_len, hidden_size] |
| `input_ids` | Tensor | 输入token ids [batch, seq_len] |
| `attention_mask` | Tensor | 注意力掩码 [batch, seq_len] |

**返回值**：
- `torch.Tensor`: 预测的hidden states [batch, seq_len, hidden_size]

---

### 7.1.3 EAGLE-3 Model

**位置**: `eagle/traineagle3/cnets.py`

#### `Model.__init__`

```python
def __init__(
    self,
    config: EConfig,
    ds_config: dict,
    training_config: dict,
    load_head: bool = False,
    load_emb: bool = True,
    path: Optional[str] = None
)
```

**参数说明**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `config` | EConfig | 必填 | 模型配置对象 |
| `ds_config` | dict | 必填 | DeepSpeed配置字典 |
| `training_config` | dict | 必填 | 训练配置字典 |
| `load_head` | bool | False | 是否加载LM head |
| `load_emb` | bool | True | 是否加载embedding层 |
| `path` | str | None | 基础模型路径 |

---

#### `Model.forward`

```python
def forward(
    self,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_values: Optional[List] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    loss_mask: Optional[torch.Tensor] = None
) -> Tuple[List, List, List]
```

**参数说明**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `input_ids` | Tensor | 必填 | 输入token ids [batch, seq_len] |
| `attention_mask` | Tensor | None | 注意力掩码 [batch, seq_len] |
| `position_ids` | Tensor | None | 位置ids [batch, seq_len] |
| `past_key_values` | List | None | 过去的KV cache |
| `use_cache` | bool | None | 是否使用并返回cache |
| `loss_mask` | Tensor | None | loss掩码 [batch, seq_len] |

**返回值**：
- `plosses`: List[7个位置的perplexity loss]
- `vlosses`: List[7个位置的value loss]
- `acces`: List[7个位置的accuracy]

---

## 7.2 数据处理API

### 7.2.1 数据集构建

#### `build_chat_dataset_rank`

```python
def build_chat_dataset_rank(
    tokenizer: PreTrainedTokenizerBase,
    datapath: str
) -> Dataset
```

**位置**: `eagle/traineagle3/main.py:145-261`

**参数说明**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `tokenizer` | PreTrainedTokenizerBase | Hugging Face tokenizer |
| `datapath` | str | 数据文件路径（.jsonl格式） |

**返回值**：
- `Dataset`: Hugging Face数据集对象

**数据格式**：

```json
{
  "id": "unique_id",
  "conversations": [
    {"from": "human", "value": "question"},
    {"from": "gpt", "value": "answer"}
  ]
}
```

---

#### `build_fim_dataset_rank`

```python
def build_fim_dataset_rank(
    tokenizer: PreTrainedTokenizerBase,
    datapath: str
) -> Dataset
```

**位置**: `eagle/traineagle3/main.py:58-142`

**参数说明**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `tokenizer` | PreTrainedTokenizerBase | Hugging Face tokenizer |
| `datapath` | str | 数据文件路径（.jsonl格式） |

**返回值**：
- `Dataset`: Hugging Face数据集对象

**数据格式**：

```json
{
  "prefix": "code before",
  "middle": "code to generate",
  "suffix": "code after"
}
```

---

#### `build_dataset_rank`

```python
def build_dataset_rank(
    tokenizer: PreTrainedTokenizerBase,
    datapath: str,
    data_format: str = 'chat'
) -> Dataset
```

**位置**: `eagle/traineagle3/main.py:264-281`

**参数说明**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `tokenizer` | PreTrainedTokenizerBase | 必填 | Hugging Face tokenizer |
| `datapath` | str | 必填 | 数据文件路径 |
| `data_format` | str | 'chat' | 数据格式：'chat' 或 'fim' |

**返回值**：
- `Dataset`: Hugging Face数据集对象

---

### 7.2.2 DataCollator

#### `DataCollatorWithPadding`

```python
class DataCollatorWithPadding:
    def __call__(
        self,
        features: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]
```

**位置**: `eagle/traineagle3/main.py:285-313`

**参数说明**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `features` | List[Dict] | 数据样本列表 |

**返回值**：
- `Dict[str, Tensor]`: 批处理数据，包含：
  - `input_ids`: [batch, max_seq_len]
  - `attention_mask`: [batch, max_seq_len]
  - `loss_mask`: [batch, max_seq_len]

**示例**：

```python
collator = DataCollatorWithPadding()
batch = collator([
    {'input_ids': tensor([1,2,3]), 'attention_mask': tensor([1,1,1]), 'loss_mask': tensor([0,1,1])},
    {'input_ids': tensor([4,5]), 'attention_mask': tensor([1,1]), 'loss_mask': tensor([1,1])}
])
# batch['input_ids']: [[1,2,3], [4,5,0]]  # 自动padding到相同长度
```

---

## 7.3 训练API

### 7.3.1 损失函数

#### `compute_loss`

```python
def compute_loss(
    target: torch.Tensor,
    target_p: torch.Tensor,
    predict: torch.Tensor,
    loss_mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
```

**位置**: `eagle/train/main.py:231-238`

**参数说明**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `target` | Tensor | 目标hidden states [batch, seq_len, hidden_size] |
| `target_p` | Tensor | 目标概率分布 [batch, seq_len, vocab_size] |
| `predict` | Tensor | 预测hidden states [batch, seq_len, hidden_size] |
| `loss_mask` | Tensor | loss掩码 [batch, seq_len, 1] |

**返回值**：
- `vloss`: Value loss (特征重建损失)
- `ploss`: Perplexity loss (概率匹配损失)
- `out_head`: 预测的logits [batch, seq_len, vocab_size]

**示例**：

```python
criterion = nn.SmoothL1Loss(reduction="none")
vloss, ploss, out_head = compute_loss(
    target=data["target"],
    target_p=target_p,
    predict=predict,
    loss_mask=loss_mask
)
loss = v_w * vloss + p_w * ploss
```

---

### 7.3.2 评估函数

#### `top_accuracy`

```python
def top_accuracy(
    output: torch.Tensor,
    target: torch.Tensor,
    topk: Tuple[int, ...] = (1,)
) -> List[torch.Tensor]
```

**位置**: `eagle/train/main.py:214-229`

**参数说明**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `output` | Tensor | 模型输出logits [batch, num_classes] |
| `target` | Tensor | 目标类别 [batch] |
| `topk` | Tuple[int] | (1,) | 要计算的top-k准确率 |

**返回值**：
- `List[Tensor]`: 每个k的正确数量

**示例**：

```python
# 计算top-1, top-2, top-3准确率
correct_1, correct_2, correct_3 = top_accuracy(
    output,
    target,
    topk=(1, 2, 3)
)
acc_1 = correct_1 / batch_size
acc_2 = correct_2 / batch_size
acc_3 = correct_3 / batch_size
```

---

#### `getkacc`

```python
@torch.no_grad()
def getkacc(
    model: nn.Module,
    data: Dict[str, torch.Tensor],
    head: nn.Linear,
    max_length: int = 5
) -> List[float]
```

**位置**: `eagle/train/main.py:240-293`

**参数说明**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model` | nn.Module | EAGLE模型 |
| `data` | Dict | 批处理数据 |
| `head` | Linear | LM head层 |
| `max_length` | int | 5 | 最大预测长度 |

**返回值**：
- `List[float]`: 每个位置的准确率 [acc_pos1, acc_pos2, ..., acc_pos5]

---

## 7.4 推理API

### 7.4.1 完整推理示例

```python
from eagle.model.ea_model import EaModel
from transformers import AutoTokenizer
import torch

# 加载模型和tokenizer
model = EaModel.from_pretrained(
    base_model_path="meta-llama/Llama-3.1-8B-Instruct",
    ea_model_path="yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
    torch_dtype=torch.float16,
    use_eagle3=True
)
model.eval()
tokenizer = model.tokenizer

# 准备输入
prompt = "Write a Python function to calculate fibonacci numbers"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# 生成
with torch.no_grad():
    outputs = model.eagenerate(
        inputs.input_ids,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )

# 解码
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
```

### 7.4.2 批处理推理

```python
def batch_generate(
    model: EaModel,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    max_new_tokens: int = 128,
    batch_size: int = 4
) -> List[str]:
    """批量生成文本"""
    results = []

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]

        # Tokenize
        inputs = tokenizer(
            batch_prompts,
            padding=True,
            return_tensors="pt"
        ).to(model.device)

        # Generate
        with torch.no_grad():
            outputs = model.eagenerate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens
            )

        # Decode
        batch_results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        results.extend(batch_results)

    return results

# 使用
prompts = [
    "Explain quantum computing",
    "What is machine learning?",
    "How does the internet work?"
]
results = batch_generate(model, tokenizer, prompts)
```

### 7.4.3 流式生成

```python
def stream_generate(
    model: EaModel,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 256
):
    """流式生成（逐token输出）"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs.input_ids

    generated = input_ids.clone()

    for _ in range(max_new_tokens):
        # 生成下一个token
        with torch.no_grad():
            outputs = model.eagenerate(
                generated,
                max_new_tokens=1,
                do_sample=False  # 贪婪解码
            )

        # 获取新生成的token
        new_token = outputs[:, -1:]
        generated = torch.cat([generated, new_token], dim=-1)

        # 解码并输出
        new_text = tokenizer.decode(new_token[0], skip_special_tokens=True)
        print(new_text, end='', flush=True)

        # 检查结束
        if new_token.item() == tokenizer.eos_token_id:
            break

    print()  # 换行

# 使用
stream_generate(model, tokenizer, "Once upon a time")
```

---

## 7.5 配置类

### 7.5.1 EConfig

**位置**: `eagle/traineagle3/configs.py` 或 `eagle/model/configs.py`

```python
class EConfig(PretrainedConfig):
    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 4096,
        intermediate_size: int = 11008,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: Optional[int] = None,
        hidden_act: str = "silu",
        max_position_embeddings: int = 2048,
        rms_norm_eps: float = 1e-6,
        draft_vocab_size: int = 4096,  # EAGLE-3特有
        **kwargs
    )
```

**参数说明**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `vocab_size` | int | 32000 | 词汇表大小 |
| `hidden_size` | int | 4096 | 隐藏层维度 |
| `intermediate_size` | int | 11008 | MLP中间层维度 |
| `num_hidden_layers` | int | 32 | Transformer层数 |
| `num_attention_heads` | int | 32 | 注意力头数 |
| `num_key_value_heads` | int | None | KV头数（GQA） |
| `draft_vocab_size` | int | 4096 | 草稿词汇表大小（EAGLE-3） |

---

## 7.6 常用工具函数

### 7.6.1 检查点管理

```python
def find_max_state_with_file(
    directory: str,
    filename: str = "zero_to_fp32.py"
) -> Tuple[Optional[str], int]
```

**位置**: `eagle/traineagle3/main.py:353-365`

**参数说明**：
- `directory`: 检查点目录
- `filename`: 用于判断完整检查点的文件名

**返回值**：
- `(checkpoint_path, start_epoch)`: 检查点路径和起始epoch

---

### 7.6.2 Padding函数

```python
def padding(tensor: torch.Tensor, left: bool = False) -> torch.Tensor
```

**位置**: `eagle/traineagle3/cnets.py`

**功能**: 在tensor左侧添加一列零（用于多位置预测时移位输入）

**示例**：

```python
input_ids = torch.tensor([[1, 2, 3]])
padded = padding(input_ids, left=False)
# padded: [[1, 2, 3, 0]]
```

---

## 7.7 类型定义

```python
from typing import Any, Dict, List, Optional, Tuple, Union

# 数据样本类型
SampleType = Dict[str, Any]
# 包含: input_ids, attention_mask, loss_mask

# 批处理数据类型
BatchType = Dict[str, torch.Tensor]
# 包含: input_ids, attention_mask, loss_mask

# 检查点类型
CheckpointType = Dict[str, Any]
# 包含: epoch, model_state_dict, optimizer_state_dict, loss

# 训练配置类型
TrainConfigType = Dict[str, Union[int, float, str, bool]]
```

---

**文档结束**

如有疑问，请参考：
- [01. 入门指南](01-getting-started.md)
- [05. 实战教程](05-practical-tutorial.md)
- [06. 高级主题](06-advanced-topics.md)
