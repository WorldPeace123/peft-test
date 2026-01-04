# 导入所需要的库
from transformers import AutoModelForCausalLM, AutoTokenizer

# 指定模型
MODEL = './Qwen1.5-1.8B-Chat-lora'

# 加载训练好的模型和分词器
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True, device_map='auto')

# 模型设为评估状态
model.eval()

# 定义测试示例
test_examples = [
{
"instruction": "请告诉我胡天赐的身份",
"input": "胡天赐是什么人？"
},
{
"instruction": "谁是老韩的孙子？",
"input": "老韩的孙子是哪位？"
},
{
"instruction": "徐昊的儿子叫什么名字？",
"input": "徐昊是谁？"
},
{
"instruction": "谁是老韩的孙子？",
"input": "老韩的孙子是哪位？"
}
]

# 生成回答
for example in test_examples:
    context = f"Instruction: {example['instruction']}\nInput: {example['input']}\nAnswer: "
    inputs = tokenizer(context, return_tensors="pt")
    outputs = model.generate(inputs.input_ids.to(model.device), max_length=512, num_return_sequences=1, no_repeat_ngram_size=2)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Input: {example['input']}")
    print(f"Output: {answer}\n")