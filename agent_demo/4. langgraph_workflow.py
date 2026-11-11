
from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from agent import evaluate_supplier_status
from langchain_agent import run_langchain_agent


# 1. Define shared workflow state
class InvoiceState(TypedDict, total=False):
    invoice_id: str
    invoice: dict | None
    supplier: dict | None
    investigation: str
    decision: dict
    status: str


# 2. Define workflow nodes
def investigate(state: InvoiceState) -> dict:
    result = run_langchain_agent(state["invoice_id"])

    return {
        "invoice": result["invoice"],
        "supplier": result["supplier"],
        "investigation": result["answer"],
    }


def validate(state: InvoiceState) -> dict:
    decision = evaluate_supplier_status(
        state.get("invoice"),
        state.get("supplier"),
    )
    return {"decision": decision}


def route(state: InvoiceState) -> str:
    if state["decision"]["requires_review"]:
        return "mark_for_review"
    return "complete_check"


def mark_for_review(state: InvoiceState) -> dict:
    return {"status": "awaiting_review"}


def complete_check(state: InvoiceState) -> dict:
    return {"status": "supplier_check_passed"}


# 3. Build the graph
builder = StateGraph(InvoiceState)

builder.add_node("investigate", investigate)
builder.add_node("validate", validate)
builder.add_node("mark_for_review", mark_for_review)
builder.add_node("complete_check", complete_check)

builder.add_edge(START, "investigate")
builder.add_edge("investigate", "validate")

builder.add_conditional_edges(
    "validate",
    route,
    {
        "mark_for_review": "mark_for_review",
        "complete_check": "complete_check",
    },
)

builder.add_edge("mark_for_review", END)
builder.add_edge("complete_check", END)

workflow = builder.compile()


# 4. Run it
if __name__ == "__main__":
    result = workflow.invoke({
        "invoice_id": "INV-2025-0050"
    })

    print("Status:", result["status"])
    print("Decision:", result["decision"])
    print("Investigation:", result["investigation"])
