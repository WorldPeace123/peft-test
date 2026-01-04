import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

# 1. 路径
lora_path = './Qwen1.5-1.8B-Chat-lora'  # 你的LoRA目录
base_model_path =  '/opt/home2/test/.cache/huggingface/hub/models--Qwen--Qwen1.5-1.8B-Chat/snapshots/e482ee3f73c375a627a16fdf66fd0c8279743ca6'# 原始Qwen路径

# 2. 先单独检查LoRA配置
print("检查LoRA配置...")
try:
    config = PeftConfig.from_pretrained(lora_path)
    print(f"LoRA配置加载成功: {config}")
except Exception as e:
    print(f"LoRA配置损坏: {e}")
    exit()

# 3. 正确加载：先加载基础模型到CPU，再用PeftModel加载适配器
print("加载基础模型到CPU...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map=None  # 先加载到CPU
)
print("加载LoRA适配器并合并...")
model = PeftModel.from_pretrained(base_model, lora_path)
model = model.to('cuda:0')  # 再移动到GPU
model.eval()

# 4. 验证参数
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"\n可训练参数: {trainable_params:,}")
print(f"总参数: {total_params:,}")
print(f"可训练比例: {trainable_params/total_params:.4%}")