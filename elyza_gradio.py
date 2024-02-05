from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import gradio as gr

message_history = []

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
    global message_history

    prompt = f"""<s>[INST] <<SYS>>
あなたは誠実で優秀な日本人のアシスタントです。
<</SYS>>

    {message} [/INST]"""

    message_history.append({
        "role": "user",
        "content": message
    })

    with torch.no_grad():
        token_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
        output_ids = model.generate(
            token_ids.to(model.device),
            max_new_tokens=1024,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    ai_message = tokenizer.decode(output_ids.tolist()[0][token_ids.size(1) :], skip_special_tokens=True)

    message_history.append({
        "role": "assistant",
        "content": ai_message
    })

    # 全会話履歴をChatbot用タプル・リストに変換して返す
    return [(message_history[i]["content"], message_history[i+1]["content"]) for i in range(0, len(message_history)-1, 2)]

with gr.Blocks() as demo:
    # チャットボットUI処理
    gr.Markdown("# Elyza-japanese-Llama-2-7b-instuctのローカル動作デモ\n yuji hirose")
    chatbot = gr.Chatbot(height=800)
#    input = gr.Textbox(show_label=False, placeholder="メッセージを入力してください").style(container=False)
    input = gr.Textbox(show_label=False)
    input.submit(fn=chat, inputs=input, outputs=chatbot) # メッセージ送信されたら、AIと会話してチャッ ト欄に全会話内容を表示
    input.submit(fn=lambda: "", inputs=None, outputs=input) # （上記に加えて）入力欄をクリア

demo.launch(
    server_port=5000,
    server_name="0.0.0.0"
)
