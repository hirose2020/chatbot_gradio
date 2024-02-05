#!/usr/bin/env python
#
# Elyza 7b instruct using pipeline
# 英語から日本語へ翻訳するボット
# 消費VRAM 16, 7GB

import torch
from transformers import AutoTokenizer, pipeline

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = "あなたは誠実で優秀な日本人のアシスタントです。以下の英語を日本語に翻訳してください。"

model_name="elyza/ELYZA-japanese-Llama-2-7b-instruct"

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

    outputs = pipe(
        prompt,
        max_new_tokens=1024,
        temperature=0.7,
        top_k=20,
        top_p=0.95
    )

    return outputs[0]["generated_text"]

while True:
    p = input("#Q: ")
    print(f"#A: {generate(p)}")
