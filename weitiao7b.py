from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from transformers import TrainerCallback
from peft import get_peft_model, LoraConfig, TaskType
import torch
from torch.utils.data import Dataset
import json
import os
import shutil
import matplotlib.pyplot as plt

# 清理检查点
if os.path.exists('./qwen7b-results'):
    print("清理旧的检查点目录...")
    shutil.rmtree('./qwen7b-results')
os.makedirs('./qwen7b-results', exist_ok=True)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 1. 模型路径
MODEL = '/opt/home2/test/.cache/modelscope/hub/models/Qwen/Qwen-7B-Chat'

# 2. 加载分词器
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

# 确保pad_token设置正确
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

print(f"pad_token: {tokenizer.pad_token}")
print(f"pad_token_id: {tokenizer.pad_token_id}")

# 3. 加载模型
print("加载Qwen-7B模型...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)

# 4. 配置LoRA
print("\n配置LoRA...")
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["c_attn", "c_proj", "w1", "w2", "w3"],
    bias="none",
)

# 应用LoRA
model = get_peft_model(model, peft_config)

# 5. 检查梯度设置
print("\n" + "="*50)
print("检查模型梯度设置:")
model.print_trainable_parameters()

trainable_count = 0
for name, param in model.named_parameters():
    if param.requires_grad:
        trainable_count += 1
        if trainable_count <= 3:
            print(f"可训练参数 {trainable_count}: {name} - 形状: {param.shape}")

print(f"总计可训练参数: {trainable_count}")
print("="*50)

# 6. 数据集类
class SimpleDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length
        print(f"加载数据集，共 {len(self.data)} 条样本")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        example = self.data[idx]
        
        # 简单格式
        text = f"Instruction: {example['instruction']}"
        if example.get("input", "").strip():
            text += f"\nInput: {example['input']}"
        text += f"\nAnswer: {example['output']}"
        
        # 编码
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None
        )
        
        # 返回字典格式
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": [1] * len(encoded["input_ids"]),
            "labels": encoded["input_ids"].copy()
        }
    
    def collate_fn(self, batch):
        input_ids = [torch.tensor(item["input_ids"], dtype=torch.long) for item in batch]
        attention_masks = [torch.tensor(item["attention_mask"], dtype=torch.long) for item in batch]
        labels = [torch.tensor(item["labels"], dtype=torch.long) for item in batch]
        
        # 获取padding值
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = 0
        
        # 填充
        input_ids_padded = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=pad_token_id
        )
        attention_mask_padded = torch.nn.utils.rnn.pad_sequence(
            attention_masks, batch_first=True, padding_value=0
        )
        labels_padded = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )
        
        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask_padded,
            "labels": labels_padded
        }

# 7. 加载数据
print("\n加载数据集...")
train_dataset = SimpleDataset("data/cpmi_dataset.json", tokenizer, max_length=512)

# 测试数据
print(f"数据集大小: {len(train_dataset)}")
sample = train_dataset[0]
print(f"样本input_ids长度: {len(sample['input_ids'])}")

# 8. 训练参数 - 关键修复：禁用梯度检查点或正确处理
training_args = TrainingArguments(
    output_dir='./qwen7b-results',
    num_train_epochs=50,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    gradient_checkpointing=False,  # 暂时禁用梯度检查点
    fp16=False,#V100用不了，只能为False
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_steps=50,
    logging_steps=5,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    evaluation_strategy="no",
    dataloader_drop_last=True,
    remove_unused_columns=False,
    report_to="none",
    optim="adamw_torch",
    max_grad_norm=0.5,
    lr_scheduler_type="linear",
    # 移除 gradient_checkpointing_kwargs 参数
)

# 9. 日志回调
class TrainingLogger(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and 'loss' in logs:
            print(f"Step {state.global_step}: loss={logs['loss']:.4f}, lr={logs.get('learning_rate', 'N/A')}")

# 10. 创建训练器 - 使用默认的Trainer，不要自定义training_step
print("\n" + "="*50)
print("创建训练器...")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=train_dataset.collate_fn,
    callbacks=[TrainingLogger()],
)

# 11. 测试前向传播（已经成功，跳过）

# 12. 开始训练
print("\n" + "="*50)
print("开始训练...")
print(f"训练参数:")
print(f"  批次大小: {training_args.per_device_train_batch_size}")
print(f"  梯度累积: {training_args.gradient_accumulation_steps}")
print(f"  学习率: {training_args.learning_rate}")
print(f"  数据集大小: {len(train_dataset)}")
print(f"  总步数估计: {len(train_dataset) // training_args.per_device_train_batch_size * training_args.num_train_epochs}")
print("="*50)

try:
    # 开始训练
    trainer.train()
    
    print("\n训练完成!")
    
    # 保存模型
    save_dir = "Qwen-7B-Chat-finetuned"
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"模型保存到: {save_dir}")
    
    # 绘制损失曲线
    if trainer.state.log_history:
        losses = [log['loss'] for log in trainer.state.log_history if 'loss' in log]
        if losses:
            plt.figure(figsize=(10, 6))
            plt.plot(losses)
            plt.title('Qwen-7B Training Loss')
            plt.xlabel('Step')
            plt.ylabel('Loss')
            plt.grid(True, alpha=0.3)
            plt.savefig('qwen7b_training_loss.png', dpi=300)
            plt.close()
            
            print(f"\n训练统计:")
            print(f"  总步数: {len(losses)}")
            print(f"  初始损失: {losses[0]:.4f}")
            print(f"  最终损失: {losses[-1]:.4f}")
            print(f"  下降幅度: {losses[0]-losses[-1]:.4f}")
            
except Exception as e:
    print(f"\n训练出错: {e}")
    import traceback
    traceback.print_exc()
