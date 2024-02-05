from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import gradio as gr
import gpu_info

# トークナイザーとモデルの準備
tokenizer = AutoTokenizer.from_pretrained(
    "elyza/ELYZA-japanese-Llama-2-7b-instruct"
)
model = AutoModelForCausalLM.from_pretrained(
    "elyza/ELYZA-japanese-Llama-2-7b-instruct",
    torch_dtype=torch.float16,
    device_map="auto"
)

def chat(message, history):
    prompt = f"""<s>[INST] <<SYS>>
あなたは誠実で優秀な日本人のアシスタントです。
<</SYS>>

    {message} [/INST]"""

    with torch.no_grad():
        token_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
        output_ids = model.generate(
            token_ids.to(model.device),
            max_new_tokens=1024,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    print(gpu_info.get_memory_info())
    return tokenizer.decode(output_ids.tolist()[0][token_ids.size(1) :], skip_special_tokens=True)

demo = gr.ChatInterface(
    fn=chat,
    chatbot=gr.Chatbot(height=600),
    title="ELYZA-japanese-Llama-2-7b-instruct",
    description="elyza/ELYZA-japanese-Llama-2-7b-instruct demo",
    theme="soft",
    examples=["hello", "am i cook?", "are tomatoes vegetables?"],
    #cache_examples=True,
    retry_btn=False,
    undo_btn=False,
    #delete_last_btn="Delete",
    clear_btn="クリア",
    submit_btn="送信",
    stop_btn="停止",
    analytics_enabled=True,
    autofocus=True,
)

demo.launch(
    server_port=5000,
    server_name="0.0.0.0"
)
