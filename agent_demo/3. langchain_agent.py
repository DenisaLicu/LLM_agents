import json
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI

from agent import get_invoice, get_supplier, evaluate_supplier_status


load_dotenv()


@tool
def lookup_invoice(invoice_id: str) -> dict:
    """Retrieve invoice details by unique invoice ID."""
    return get_invoice(invoice_id)


@tool
def lookup_supplier(supplier_id: str) -> dict:
    """Retrieve supplier details and active status by supplier ID."""
    return get_supplier(supplier_id)


model = ChatOpenAI(model="gpt-5-mini")

agent = create_agent(
    model=model,
    tools=[lookup_invoice, lookup_supplier],
    system_prompt=(
        "You are an invoice investigation assistant. "
        "Use tools to verify invoice and supplier information. "
        "Do not invent missing facts. "
        "You can investigate but cannot modify records."
    ),
)



def run_langchain_agent(invoice_id: str) -> dict:
    result = agent.invoke(
        {
            "messages": [{
                "role": "user",
                "content": (
                    f"Investigate invoice {invoice_id}. "
                    "Report its total, payment status, "
                    "and whether its supplier is active."
                ),
            }]
        },
        config={"recursion_limit": 12},
    )

    invoice_data = None
    supplier_data = None
    trace = []

    for message in result["messages"]:
        if message.type == "ai":
            for call in message.tool_calls:
                trace.append({
                    "tool": call["name"],
                    "arguments": call["args"],
                })

        elif message.type == "tool":
            data = json.loads(message.content)

            if "error" in data:
                continue

            if message.name == "lookup_invoice":
                if data.get("invoice_id") == invoice_id:
                    invoice_data = data

            elif message.name == "lookup_supplier":
                if (
                    invoice_data
                    and data.get("supplier_id")
                    == invoice_data.get("supplier_id")
                ):
                    supplier_data = data

    decision = evaluate_supplier_status(
        invoice_data, supplier_data
    )

    return {
        "answer": result["messages"][-1].content,
        "trace": trace,
        "decision": decision,
        "invoice": invoice_data,
        "supplier": supplier_data,
    }


if __name__ == "__main__":
    result = run_langchain_agent("INV-2025-0050")

    print(result["answer"])
    print(json.dumps(result["trace"], indent=2))
    print(result["decision"])
