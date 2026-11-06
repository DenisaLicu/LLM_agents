import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET_PATH = (
    Path(__file__).resolve().parent.parent
    / "experiments"
    / "eval_cases.json"
)

LABELS = {"invoice", "purchase_order", "receipt", "other"}


def classify(text, model, tokenizer):
    messages = [
        {
            "role": "system",
            "content": (
                "You classify business documents. "
                "Return exactly one label: "
                "invoice, purchase_order, receipt, or other. "
                "Classify by the document's purpose, not just keywords. "
                "Do not include explanations."
            ),
        },
        {
            "role": "user",
            "content": f"Classify this document:\n\n{text}",
        },
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    input_length = inputs["input_ids"].shape[-1]

    prediction = tokenizer.decode(
        outputs[0, input_length:],
        skip_special_tokens=True,
    ).strip().lower()

    return prediction


def main():
    cases = json.loads(DATASET_PATH.read_text(encoding="utf-8"))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.eval()

    correct = 0
    valid = 0

    for case in cases:
        prediction = classify(case["text"], model, tokenizer)
        expected = case["expected"]

        is_valid = prediction in LABELS
        is_correct = prediction == expected

        valid += is_valid
        correct += is_correct

        status = "PASS" if is_correct else "FAIL"

        print(f"\n{case['id']}: {status}")
        print(f"  Expected:  {expected}")
        print(f"  Predicted: {prediction!r}")

    total = len(cases)

    print("\n--- EVALUATION SUMMARY ---")
    print(f"Accuracy: {correct}/{total} ({correct / total:.0%})")
    print(f"Valid label rate: {valid}/{total} ({valid / total:.0%})")


if __name__ == "__main__":
    main()
