
import json

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

# Fictional database
INVOICES = {
    "INV-2025-0042": {
        "invoice_id": "INV-2025-0042",
        "supplier": "ABC Ltd",
        "total": 2500,
        "currency": "EUR",
        "status": "unpaid",
    }
}


# Real Python function
def get_invoice(invoice_id: str) -> dict:
    return INVOICES.get(
        invoice_id,
        {"error": "INVOICE_NOT_FOUND"},
    )


# Tool interface exposed to the LLM
tools = [
    {
        "type": "function",
        "name": "get_invoice",
        "description": "Retrieve an invoice by its unique ID.",
        "parameters": {
            "type": "object",
            "properties": {
                "invoice_id": {"type": "string"},
            },
            "required": ["invoice_id"],
            "additionalProperties": False,
        },
        "strict": True,
    }
]

question = (
    "What is the total and payment status "
    "of invoice INV-2025-0042?"
)

messages = [{"role": "user", "content": question}]

# 1. Ask the model to request a tool call
response = client.responses.create(
    model="gpt-5-mini",
    input=messages,
    tools=tools,
    tool_choice="required",
)

messages.extend(response.output)

# 2. Inspect and execute the requested tool
calls = [
    item for item in response.output
    if item.type == "function_call"
]

if len(calls) != 1:
    raise RuntimeError(f"Expected one tool call, got {len(calls)}")

call = calls[0]
print("Tool requested:", call.name)
print("Arguments:", call.arguments)

if call.name != "get_invoice":
    raise ValueError(f"Unexpected tool: {call.name}")

args = json.loads(call.arguments)
result = get_invoice(args["invoice_id"])

print("Tool result:", result)

# 3. Return the tool result to the model
messages.append({
    "type": "function_call_output",
    "call_id": call.call_id,
    "output": json.dumps(result),
})

response = client.responses.create(
    model="gpt-5-mini",
    instructions="Answer using only the invoice data provided.",
    input=messages,
    tool_choice="none",
    tools=tools,
)

print("\nFinal answer:")
print(response.output_text)
