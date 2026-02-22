import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_MODEL = "helinow/careermate"

device = "cpu"


print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# Fix padding token if missing
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,
    trust_remote_code=True
)


print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, ADAPTER_MODEL)
model.to(device)
model.eval()
print("Model loaded successfully!")


def format_prompt(user_query):
    return f"""I am CareerMate, a professional AI career assistant.
I provide clear, helpful, structured career advice.

<|user|>
{user_query}

<|assistant|>
"""


def chat(message, history):

    prompt = format_prompt(message)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.3,          # less randomness = more factual
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)

    # Extract only assistant reply
    if "<|assistant|>" in decoded:
        response = decoded.split("<|assistant|>")[-1].strip()
    else:
        response = decoded.strip()

    return response


demo = gr.ChatInterface(
    fn=chat,
    title="CareerMate AI",
    description="Fine-tuned Career Assistant powered by TinyLlama",
)


if __name__ == "__main__":
    demo.launch()