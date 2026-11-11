
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
        "supplier_id": "SUP-001",
        "total": 2500,
        "currency": "EUR",
        "status": "unpaid",
    },
    "INV-2025-0050": {
    "invoice_id": "INV-2025-0050",
    "supplier": "XYZ Ltd",
    "supplier_id": "SUP-002",
    "total": 1800,
    "currency": "EUR",
    "status": "unpaid",
    }
}

SUPPLIERS = {
    "SUP-001": {
        "supplier_id": "SUP-001",
        "name": "ABC Ltd",
        "status": "active",
    },
    "SUP-002": {
    "supplier_id": "SUP-002",
    "name": "XYZ Ltd",
    "status": "inactive",
    }
}


# Real Python functions
def get_invoice(invoice_id: str) -> dict:
    return INVOICES.get(
        invoice_id,
        {"error": "INVOICE_NOT_FOUND"},
    )

def get_supplier(supplier_id: str) -> dict:
    return SUPPLIERS.get(
        supplier_id,
        {"error": "SUPPLIER_NOT_FOUND"},
    )


def evaluate_supplier_status(invoice: dict | None, supplier: dict | None) -> dict:
    if not invoice or not supplier:
        return {
            "requires_review": True,
            "reason": "Missing invoice or supplier information",
        }

    if supplier.get("status") != "active":
        return {
            "requires_review": True,
            "reason": f"Supplier status is {supplier.get('status', 'unknown')}",
        }

    return {
        "requires_review": False,
        "reason": "Supplier is active",
    }


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
    },
    {
    "type": "function",
    "name": "get_supplier",
    "description": "Retrieve a supplier and its current status by supplier ID.",
    "parameters": {
        "type": "object",
        "properties": {
            "supplier_id": {"type": "string"},
        },
        "required": ["supplier_id"],
        "additionalProperties": False,
    },
    "strict": True,
    }
]


tool_functions = {
    "get_invoice": get_invoice,
    "get_supplier": get_supplier,
}



MAX_TOOL_ROUNDS = 5



def run_agent(invoice_id: str) -> dict:
    question = (
        f"Investigate invoice {invoice_id}. "
        "Report its total, payment status, and whether its supplier is active."
    )

    messages = [{"role": "user", "content": question}]
    invoice_data = None
    supplier_data = None
    trace = []
    final_answer = None

    for round_number in range(1, MAX_TOOL_ROUNDS + 1):
        response = client.responses.create(
            model="gpt-5-mini",
            instructions=(
                "Investigate invoices using available tools. "
                "Do not invent missing business facts. "
                "You cannot modify records."
            ),
            input=messages,
            tools=tools,
        )

        messages.extend(response.output)

        calls = [
            item for item in response.output
            if item.type == "function_call"
        ]

        if not calls:
            final_answer = response.output_text
            break

        for call in calls:
            try:
                arguments = json.loads(call.arguments)

                if call.name not in tool_functions:
                    result = {"error": "TOOL_NOT_ALLOWED"}
                elif not isinstance(arguments, dict):
                    result = {"error": "INVALID_TOOL_ARGUMENTS"}
                else:
                    result = tool_functions[call.name](**arguments)

            except (ValueError, TypeError) as exc:
                result = {
                    "error": "INVALID_TOOL_ARGUMENTS",
                    "message": str(exc),
                }

            trace.append({
                "round": round_number,
                "tool": call.name,
                "arguments": call.arguments,
                "result": result,
            })

            if isinstance(result, dict) and "error" not in result:
                if call.name == "get_invoice":
                    invoice_data = result
                elif call.name == "get_supplier":
                    supplier_data = result

            messages.append({
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": json.dumps(result),
            })

    # If the tool budget is exhausted, request a final answer.
    if final_answer is None:
        response = client.responses.create(
            model="gpt-5-mini",
            instructions=(
                "Summarize the investigation using only the available "
                "tool results. Mention incomplete checks. "
                "Do not claim missing information was verified."
            ),
            input=messages,
            tools=tools,
            tool_choice="none",
        )
        final_answer = response.output_text
    
    decision = evaluate_supplier_status(invoice_data, supplier_data)

    return {
        "answer": final_answer,
        "trace": trace,
        "decision": decision,
        "invoice": invoice_data,
        "supplier": supplier_data,
    }


if __name__ == "__main__":
    result = run_agent("INV-2025-0050")

    print(result["answer"])
    print(json.dumps(result["trace"], indent=2))
    print(result["decision"])
