# 05. 实战教程

> 端到端训练EAGLE模型的完整指南

## 目录

- [5.1 教程1: 训练代码补全模型](#51-教程1-训练代码补全模型)
- [5.2 教程2: 迁移到新模型](#52-教程2-迁移到新模型)
- [5.3 训练监控与评估](#53-训练监控与评估)
- [5.4 模型测试与部署](#54-模型测试与部署)

---

## 5.1 教程1: 训练代码补全模型

### 5.1.1 目标

使用FIM格式数据训练一个Python代码补全模型

### 5.1.2 步骤1: 准备数据

**数据来源**：

```python
# 准备Python代码数据
# 可以从GitHub、CodeSearchNet等获取

# 示例：转换现有代码为FIM格式
def convert_code_to_fim(code_file, output_file):
    """将Python代码转换为FIM格式"""
    import ast
    import random

    with open(code_file) as f:
        code = f.read()

    try:
        tree = ast.parse(code)
    except:
        return

    # 提取函数定义
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # 获取函数签名作为prefix
            prefix = f"def {node.name}{ast.get_source_segment(code, node.args)}:\n    "

            # 函数体作为middle（清空）
            middle = ""  # 需要模型生成

            # 获取函数后的代码作为suffix
            func_end = node.end_lineno
            suffix_code = code.split('\n')[func_end:]
            suffix = '\n'.join(suffix_code)

            # 保存
            with open(output_file, 'a') as out:
                import json
                out.write(json.dumps({
                    "prefix": prefix,
                    "middle": middle,
                    "suffix": suffix
                }) + '\n')

# 使用示例
convert_code_to_fim('my_project.py', 'fim_data.jsonl')
```

**数据质量检查**：

```bash
# 检查数据格式
python -c "
import json
with open('fim_data.jsonl') as f:
    for i, line in enumerate(f):
        data = json.loads(line)
        assert 'prefix' in data and 'middle' in data and 'suffix' in data
        if i < 5:
            print(f'✓ Sample {i}: OK')
print(f'Total samples: {i+1}')
"
```

**划分数据集**：

```python
def split_dataset(input_file, train_ratio=0.95):
    """划分训练集和测试集"""
    with open(input_file) as f:
        lines = f.readlines()

    import random
    random.shuffle(lines)
    split = int(len(lines) * train_ratio)

    with open('train.jsonl', 'w') as f:
        f.writelines(lines[:split])

    with open('test.jsonl', 'w') as f:
        f.writelines(lines[split:])

    print(f"Train: {split}, Test: {len(lines)-split}")

split_dataset('fim_data.jsonl')
```

### 5.1.3 步骤2: 配置训练环境

**创建DeepSpeed配置** (`ds_config.json`):

```json
{
    "train_micro_batch_size_per_gpu": 2,
    "gradient_accumulation_steps": 4,
    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": true,
        "allgather_bucket_size": 5e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 5e8,
        "contiguous_gradients": true
    },
    "gradient_clipping": 0.5,
    "steps_per_print": 100,
    "train_batch_size": 16,
    "wall_clock_breakdown": false
}
```

**安装依赖**：

```bash
# 创建虚拟环境
python -m venv eagle_env
source eagle_env/bin/activate

# 安装依赖
pip install torch==2.0.0 transformers datasets deepspeed
pip install safetensors accelerate tqdm

# 验证安装
python -c "import torch, transformers, deepspeed; print('✓ All dependencies installed')"
```

### 5.1.4 步骤3: 执行训练

**单GPU训练（测试）**：

```bash
cd eagle/traineagle3

python main.py \
    --basepath /path/to/CodeLlama-7B-Python \
    --trainpath /path/to/train.jsonl \
    --testpath /path/to/test.jsonl \
    --savedir ./checkpoints \
    --data-format fim \
    --deepspeed_config ds_config.json \
    --local_rank -1
```

**多GPU训练（推荐）**：

```bash
# 使用deepspeed启动
deepspeed --num_gpus=2 main.py \
    --basepath /path/to/CodeLlama-7B-Python \
    --trainpath /path/to/train.jsonl \
    --testpath /path/to/test.jsonl \
    --savedir ./checkpoints \
    --data-format fim \
    --deepspeed_config ds_config.json

# 或使用torchrun
torchrun --nproc_per_node=2 main.py \
    --basepath /path/to/CodeLlama-7B-Python \
    --trainpath /path/to/train.jsonl \
    --testpath /path/to/test.jsonl \
    --savedir ./checkpoints \
    --data-format fim \
    --deepspeed_config ds_config.json
```

**预期输出**：

```
Now training epoch 0
Data format: fim
100%|████████████████| 500/500 [10:23<00:00,  1.25s/it]
Train Epoch [1/40], position 0, Acc: 35.42
Train Epoch [1/40], position 0, pLoss: 3.21
Train Epoch [1/40], position 1, Acc: 28.15
...
Test Epoch [1/40], position 0, Acc: 33.89
Saving checkpoint to ./checkpoints/state_0
```

### 5.1.5 步骤4: 监控训练

**使用WandB**：

```bash
# 在训练前登录
wandb login

# 训练会自动上传指标
# 访问 https://wandb.ai 查看实时仪表盘
```

**本地监控**：

```bash
# 监控GPU使用
watch -n 1 nvidia-smi

# 监控训练日志
tail -f training.log
```

**关键指标**：

```python
# 评估训练进度
import json

def check_progress(checkpoint_dir):
    """检查训练进度"""
    for epoch in range(40):
        metrics_file = f"{checkpoint_dir}/state_{epoch}/metrics.json"
        try:
            with open(metrics_file) as f:
                metrics = json.load(f)
            print(f"Epoch {epoch}: Acc={metrics['acc']:.2f}, Loss={metrics['loss']:.2f}")
        except:
            continue

check_progress('./checkpoints')
```

### 5.1.6 步骤5: 评估模型

**测试集评估**：

```python
from transformers import AutoTokenizer
from eagle.model.ea_model import EaModel
import torch

# 加载模型
model = EaModel.from_pretrained(
    base_model_path="/path/to/CodeLlama-7B-Python",
    ea_model_path="./checkpoints/state_39",
    use_eagle3=True
)
model.eval()
tokenizer = model.tokenizer

# 测试样本
test_cases = [
    {"prefix": "def fibonacci(n):\n    ", "suffix": "\n\nprint(fibonacci(10))"},
    {"prefix": "class Calculator:\n    def add(self, a, b):\n        ", "suffix": "\n    def subtract(self, a, b):\n        pass"}
]

# 评估
for i, case in enumerate(test_cases):
    prompt = f"<fim_prefix>{case['prefix']}<fim_middle><fim_suffix>{case['suffix']}"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.eagenerate(
            inputs.input_ids,
            max_new_tokens=50,
            temperature=0.2
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n=== Test Case {i+1} ===")
    print(f"Prefix: {case['prefix']}")
    print(f"Generated: {generated}")
```

---

## 5.2 教程2: 迁移到新模型

### 5.2.1 目标

为非LLaMA模型添加EAGLE支持

### 5.2.2 步骤1: 分析目标模型

假设目标模型是`Mistral-7B`:

```python
from transformers import AutoModelForCausalLM, AutoConfig

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct")
config = AutoConfig.from_pretrained("mistralai/Mistral-7B-Instruct")

# 检查架构
print(f"Model type: {type(model)}")
print(f"Architecture: {config.architectures}")
print(f"Hidden size: {config.hidden_size}")
print(f"Num layers: {config.num_hidden_layers}")
print(f"Num attention heads: {config.num_attention_heads}")
```

### 5.2.3 步骤2: 修改modeling文件

**需要修改的位置**：

1. KV cache支持
2. 前向传播接口
3. Attention mask处理

**示例修改**（基于`modeling_mistral.py`）：

```python
# 修改1: 添加KV cache参数
class MistralAttention(nn.Module):
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,  # [MODIFIED] 添加
        output_attentions=False,
        use_cache=False,      # [MODIFIED] 添加
        **kwargs
    ):
        # ... 原有代码 ...

        # [MODIFIED] 处理past_key_value
        if past_key_value is not None:
            # 重用cached keys和values
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1]], value_layer], dim=2)

        # [MODIFIED] 返回present_key_value
        if use_cache:
            present_key_value = (key_layer, value_layer)
        else:
            present_key_value = None

        return (outputs, present_key_value, None)  # [MODIFIED]

# 修改2: 在模型层级传递KV cache
class MistralModel(nn.Module):
    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,   # [MODIFIED] 添加
        use_cache=None,         # [MODIFIED] 添加
        **kwargs
    ):
        # ... 原有代码 ...

        # [MODIFIED] 传递past_key_values到每一层
        for i, layer in enumerate(self.layers):
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values[i] if past_key_values else None,
                use_cache=use_cache
            )

        # ... 原有代码 ...
```

### 5.2.4 步骤3: 测试修改

```python
# 测试KV cache
from modeling_mistral_kv import MistralForCausalLM

model = MistralForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct")

# 生成测试
prompt = "Hello, my name is"
inputs = tokenizer(prompt, return_tensors="pt")

# 第一次调用（无cache）
outputs1 = model(**inputs, use_cache=True)
past_key_values = outputs1.past_key_values

# 第二次调用（有cache）
outputs2 = model(
    input_ids=torch.tensor([[12345]]),  # 新token
    past_key_values=past_key_values,
    use_cache=True
)

# 验证输出一致
assert outputs1.logits.shape == outputs2.logits.shape
print("✓ KV cache works correctly")
```

### 5.2.5 步骤4: 适配tokenizer

```python
# 修改tokenizer以支持特殊tokens
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct")

# 添加EAGLE特殊tokens（如果需要）
special_tokens = {
    "pad_token": "<pad>",
    "eos_token": "</s>",
    "bos_token": "<s>"
}
tokenizer.add_special_tokens(special_tokens)

# 验证chat模板
messages = [
    {"role": "user", "content": "Hello"}
]
formatted = tokenizer.apply_chat_template(messages, tokenize=False)
print(f"Chat template: {formatted}")
```

### 5.2.6 步骤5: 训练

```bash
cd eagle/traineagle3

# 使用修改后的modeling文件训练
python main.py \
    --basepath /path/to/Mistral-7B-Instruct \
    --trainpath /path/to/train.jsonl \
    --testpath /path/to/test.jsonl \
    --savedir ./checkpoints_mistral \
    --data-format chat \
    --deepspeed_config ds_config.json
```

---

## 5.3 训练监控与评估

### 5.3.1 WandB监控

**设置自定义指标**：

```python
import wandb

# 在训练脚本中添加
wandb.init(project="eagle-training", config=train_config)

# 自定义可视化
wandb.define_metric("train/epochacc")
wandb.define_metric("test/epochacc")
wandb.define_metric("epoch")
wandb.log({"epoch": epoch, "train/epochacc": train_acc, "test/epochacc": test_acc})
```

**实时监控**：

```python
# 监控脚本
def monitor_training(wandb_run):
    """实时监控训练"""
    api = wandb.Api()
    run = api.run(wandb_run)

    history = run.history()

    # 绘制曲线
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(history['train/epochacc_0'], label='Position 0')
    plt.plot(history['train/epochacc_1'], label='Position 1')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training Accuracy')

    plt.subplot(1, 3, 2)
    plt.plot(history['train/epochploss_0'], label='Train')
    plt.plot(history['test/epochploss_0'], label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Perplexity Loss')

    plt.subplot(1, 3, 3)
    plt.plot(history['train/lr'])
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')

    plt.tight_layout()
    plt.savefig('training_curves.png')
    print("✓ Saved training_curves.png")

monitor_training("username/project/run_id")
```

### 5.3.2 检查点评估

```python
def evaluate_all_checkpoints(checkpoint_dir, test_data):
    """评估所有检查点"""
    results = []

    for epoch in range(0, 40, 5):  # 每5个epoch
        ckpt_path = f"{checkpoint_dir}/state_{epoch}"

        # 加载模型
        model = EaModel.from_pretrained(
            base_model_path=base_model,
            ea_model_path=ckpt_path,
            use_eagle3=True
        )
        model.eval()

        # 评估
        acc = evaluate(model, test_data)

        results.append({
            'epoch': epoch,
            'accuracy': acc
        })
        print(f"Epoch {epoch}: Acc={acc:.2f}")

    # 保存结果
    import json
    with open('evaluation_results.json', 'w') as f:
        json.dump(results, f)

    return results
```

### 5.3.3 错误分析

```python
def analyze_errors(model, test_data, num_samples=10):
    """分析预测错误的样本"""
    errors = []

    for sample in test_data[:num_samples]:
        inputs = tokenizer(sample['text'], return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=50)

        predicted = tokenizer.decode(outputs[0], skip_special_tokens=True)
        target = sample['expected']

        if predicted != target:
            errors.append({
                'input': sample['text'],
                'predicted': predicted,
                'target': target
            })

    # 打印错误
    for i, error in enumerate(errors):
        print(f"\n=== Error {i+1} ===")
        print(f"Input: {error['input']}")
        print(f"Predicted: {error['predicted']}")
        print(f"Target: {error['target']}")

    return errors
```

---

## 5.4 模型测试与部署

### 5.4.1 功能测试

```python
def test_code_completion(model, tokenizer):
    """测试代码补全功能"""
    test_cases = [
        ("def factorial(n):\n    if n == 0:\n        ", "return 1\n    else:\n        return n * factorial(n-1)"),
        ("def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            ", "return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1"),
    ]

    for i, (prefix, expected_suffix) in enumerate(test_cases):
        prompt = f"<fim_prefix>{prefix}<fim_middle>"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.eagenerate(
                inputs.input_ids,
                max_new_tokens=100,
                temperature=0.2,
                top_p=0.95
            )

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\n=== Test Case {i+1} ===")
        print(f"Prefix: {prefix}")
        print(f"Expected: {expected_suffix}")
        print(f"Generated: {generated}")

        # 评估BLEU分数
        from bleu_score import corpus_bleu
        score = corpus_bleu([expected_suffix], [[generated]])
        print(f"BLEU Score: {score:.2f}")

test_code_completion(model, tokenizer)
```

### 5.4.2 性能基准测试

```python
import time

def benchmark_speed(model, tokenizer, prompt, max_tokens=100):
    """测试推理速度"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # EAGLE加速
    start = time.time()
    outputs_eagle = model.eagenerate(
        inputs.input_ids,
        max_new_tokens=max_tokens
    )
    time_eagle = time.time() - start
    tokens_eagle = outputs_eagle.shape[1]

    # 标准生成
    start = time.time()
    outputs_baseline = model.naivegenerate(
        inputs.input_ids,
        max_new_tokens=max_tokens
    )
    time_baseline = time.time() - start
    tokens_baseline = outputs_baseline.shape[1]

    # 计算加速比
    speedup = time_baseline / time_eagle

    print(f"EAGLE: {tokens_eagle} tokens in {time_eagle:.2f}s ({tokens_eagle/time_eagle:.1f} tok/s)")
    print(f"Baseline: {tokens_baseline} tokens in {time_baseline:.2f}s ({tokens_baseline/time_baseline:.1f} tok/s)")
    print(f"Speedup: {speedup:.2f}x")

    return speedup

# 使用示例
prompt = "Write a Python function to calculate the fibonacci sequence"
speedup = benchmark_speed(model, tokenizer, prompt)
```

### 5.4.3 部署准备

```python
# 导出模型
def export_for_deployment(checkpoint_path, output_dir):
    """导出模型用于部署"""
    from transformers import AutoModelForCausalLM

    # 加载EAGLE模型
    eagle_model = EaModel.from_pretrained(
        base_model_path=base_model,
        ea_model_path=checkpoint_path,
        use_eagle3=True
    )

    # 保存为标准transformers格式
    eagle_model.base_model.save_pretrained(f"{output_dir}/base_model")
    eagle_model.ea_model.save_pretrained(f"{output_dir}/eagle_model")
    eagle_model.tokenizer.save_pretrained(f"{output_dir}/tokenizer")

    print(f"✓ Exported to {output_dir}")

export_for_deployment("./checkpoints/state_39", "./deployment")
```

**部署配置**：

```python
# config.json
{
    "base_model_path": "./deployment/base_model",
    "eagle_model_path": "./deployment/eagle_model",
    "tokenizer_path": "./deployment/tokenizer",
    "max_new_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.9,
    "use_eagle3": true
}
```

---

**下一篇**：[06. 高级主题](06-advanced-topics.md) →
