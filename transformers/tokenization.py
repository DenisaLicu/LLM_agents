from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct"
)

texts = [
    "The invoice is paid.",
    "The invoice is unpaid.",
    "Invoice #INV-2026-001: EUR 450.00",
]

for text in texts:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    print(f"\nText: {text}")
    print(f"Tokens: {tokens}")
    print(f"Token IDs: {token_ids}")
    print(f"Token count: {len(token_ids)}")
    print(f"Decoded: {tokenizer.decode(token_ids)}")

# ------

messages = [
    {"role": "system", "content": "You classify business documents."},
    {"role": "user", "content": "Classify: Invoice #123, total EUR 50."},
]


chat_encoding = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
)

chat_ids = chat_encoding["input_ids"]

print("Encoding type:", type(chat_encoding))
print("Chat token count:", len(chat_ids))
print("Chat token IDs:", chat_ids)

# ----
content_only = (
    "You classify business documents."
    "Classify: Invoice #123, total EUR 50."
)

content_tokens = tokenizer.encode(
    content_only,
    add_special_tokens=False,
)

print("Content-only tokens:", len(content_tokens))
print("Formatted chat tokens:", len(chat_ids))
print("Chat formatting overhead:", len(chat_ids) - len(content_tokens))