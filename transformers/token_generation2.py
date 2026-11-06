import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def main():
    # 1. Load tokenizer and model
    print("Loading model...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.eval()

    # 2. Prepare the prompt
    messages = [
        {
            "role": "user",
            "content": "Explain what an invoice is in two sentences.",
        }
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    # 3. Warm up the model
    print("Warming up...")

    with torch.inference_mode():
        model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # 4. Benchmark inference
    print("Running benchmark...")

    start = time.perf_counter()

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=60,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    elapsed = time.perf_counter() - start

    # 5. Calculate metrics
    input_tokens = inputs["input_ids"].shape[-1]
    output_tokens = outputs.shape[-1] - input_tokens
    tokens_per_second = output_tokens / elapsed

    generated_text = tokenizer.decode(
        outputs[0, input_tokens:],
        skip_special_tokens=True,
    )

    # 6. Print results
    print("\n--- BENCHMARK RESULTS ---")
    print(f"Device: {model.device}")
    print(f"Input tokens: {input_tokens}")
    print(f"Output tokens: {output_tokens}")
    print(f"Total latency: {elapsed:.2f} s")
    print(f"Output throughput: {tokens_per_second:.2f} tokens/s")

    print("\n--- GENERATED RESPONSE ---")
    print(generated_text)


if __name__ == "__main__":
    main()