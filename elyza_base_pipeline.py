#
# Elyza base
# using pipeline

import torch
from transformers import AutoTokenizer, pipeline

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = "あなたは誠実で優秀な日本人のアシスタントです。"

model_name="elyza/ELYZA-japanese-Llama-2-7b"

tokenizer = AutoTokenizer.from_pretrained(model_name)

pipe = pipeline(
    "text-generation",
    model=model_name,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
    )

def generate(text):
    prompt = "{bos_token}{b_inst} {system}{prompt} {e_inst} ".format(
        bos_token="<s>",
        b_inst=B_INST,
        system=f"{B_SYS}{DEFAULT_SYSTEM_PROMPT}{E_SYS}",
        prompt=text,
        e_inst=E_INST,
    )
    print(prompt)

    outputs = pipe(
        prompt,
        max_new_tokens=256,
        temperature=1.0,
        top_k=10,
        top_p=0.95
    )

    return outputs[0]["generated_text"]

while True:
    p = input("#Q: ")
    print(f"#A: {generate(p)}")

