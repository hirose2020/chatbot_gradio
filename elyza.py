from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# トークナイザーとモデルの準備
tokenizer = AutoTokenizer.from_pretrained(
    "elyza/ELYZA-japanese-Llama-2-7b-instruct"
)
model = AutoModelForCausalLM.from_pretrained(
    "elyza/ELYZA-japanese-Llama-2-7b-instruct",
    torch_dtype=torch.float16,
    device_map="auto"
)

# 推論の実行
def generate(p):
    prompt = f"""<s>[INST] <<SYS>>
あなたは誠実で優秀な日本人のアシスタントです。
<</SYS>>

{p} [/INST]"""
    with torch.no_grad():
        token_ids = tokenizer.encode(
            prompt,
            add_special_tokens=False,
            return_tensors="pt")
        output_ids = model.generate(
            token_ids.to(model.device),
            max_new_tokens=1024,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,)
        output = tokenizer.decode(output_ids.tolist()[0][token_ids.size(1) :], skip_special_tokens=True)
        return output

while True:
    p = input("#Q: ")
    print(f"#A: {generate(p)}")
