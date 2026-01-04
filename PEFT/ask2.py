import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 1. 加载模型（必须使用合并了LoRA的PeftModel）
base_model_path = '/opt/home2/test/.cache/huggingface/hub/models--Qwen--Qwen1.5-1.8B-Chat/snapshots/e482ee3f73c375a627a16fdf66fd0c8279743ca6'
# lora_path = './Qwen1.5-1.8B-Chat-lora'
lora_path = './Qwen-1.8B-Chat-finetuned'

tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(base_model_path, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(base_model, lora_path)
model.eval()

def ask_question(instruction, input_text=""):
    """使用与训练时完全相同的格式提问"""
    prompt = f"Instruction: {instruction}\n"
    if input_text:
        prompt += f"Input: {input_text}\nAnswer: "
    else:
        prompt += "Answer: "
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,  # 使用贪婪解码确保确定性
            temperature=1.0,
            repetition_penalty=1.0,
            num_beams=1
        )
    answer = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return answer.strip()

# 2. 设计分层次的测试问题
test_cases = [
    # 【核心知识复现】直接来自训练集的问题
    ("胡天赐是谁？", "", "胡天赐是徐昊的儿子，老韩的孙子。"),
    ("老韩是谁？", "", "老韩是313最帅的男人，是徐昊的爸爸，胡天赐的爷爷。"),
    
    # 【知识关联推理】需要模型组合已知事实
    ("徐昊和胡天赐什么关系？", "", "徐昊是胡天赐的父亲。"),
    ("胡天赐应该叫老韩什么？", "", "胡天赐应该叫老韩爷爷。"),
    
    # 【泛化能力】同义替换或新问法
    ("介绍一下胡天赐的家庭背景。", "", "胡天赐是徐昊的儿子，老韩的孙子。"),
    ("谁被称为313最帅的男人？", "", "老韩被称为313最帅的男人。"),
    
    # 【无关问题】测试微调是否干扰原有能力
    ("中国的首都是哪里？", "", "北京。"),  # 基础常识应保持正确
    ("请写一首关于春天的诗。", "", ""),  # 开放性创作能力
]

print("开始功能性测试...\n" + "="*50)
all_pass = True
for i, (instruction, inp, expected) in enumerate(test_cases):
    answer = ask_question(instruction, inp)
    # 判断是否通过：对于有预期答案的问题，检查核心关键词；对于开放问题，看是否合理
    if expected:
        # 检查是否包含预期中的核心词汇（对于您的小数据集，这是合理要求）
        keywords = ["胡天赐", "徐昊", "儿子", "老韩", "爷爷", "313最帅"]
        match_keywords = [kw for kw in keywords if kw in answer]
        if match_keywords:
            print(f"✅ 测试{i+1}通过 | 问题: {instruction}")
            print(f"   回答: {answer}")
            print(f"   匹配关键词: {match_keywords}")
        else:
            print(f"❌ 测试{i+1}失败 | 问题: {instruction}")
            print(f"   期望包含: {keywords}")
            print(f"   实际回答: {answer}")
            all_pass = False
    else:
        print(f"🔶 测试{i+1} (开放问题) | 问题: {instruction}")
        print(f"   回答: {answer[:100]}..." if len(answer) > 100 else f"   回答: {answer}")
    print("-"*50)