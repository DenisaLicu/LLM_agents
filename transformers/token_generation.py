import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
model.eval()

messages = [
    {
        "role": "system",
        "content": (
            "You write concise, professional business emails. "
            "Never invent payment status, invoice amounts, dates, "
            "or reasons for nonpayment. "
            "Write at most 3 sentences."
        ),
    },
    {
        "role": "user",
        "content": (
            "Draft a polite payment reminder for an unpaid invoice. "
            "The payment has not been received. "
            "Use placeholders for unknown invoice details. "
            "Do not claim the invoice has been paid."
        ),
    },
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
)

with torch.inference_mode():
    # Greedy decoding
    greedy_ids = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
    )

    # Sampling
    sampled_ids = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.9,
        top_p=0.9,
        top_k=0,
    )

prompt_length = inputs["input_ids"].shape[-1]

print("GREEDY:")
print(tokenizer.decode(
    greedy_ids[0, prompt_length:],
    skip_special_tokens=True,
))

print("\nSAMPLING:")
print(tokenizer.decode(
    sampled_ids[0, prompt_length:],
    skip_special_tokens=True,
))