from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt
from importlib import import_module

evaluate_supplier_status = import_module("2_agent").evaluate_supplier_status
run_langchain_agent = import_module("3_langchain_agent").run_langchain_agent


# 1. Define shared workflow state
class InvoiceState(TypedDict, total=False):
    invoice_id: str
    invoice: dict | None
    supplier: dict | None
    investigation: str
    decision: dict
    status: str
    review_decision: str


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
    reviewer_response = interrupt({
        "invoice_id": state["invoice_id"],
        "reason": state["decision"]["reason"],
        "message": "Manual review required",
    })

    if reviewer_response not in ("acknowledged", "rejected"):
        raise ValueError("Invalid reviewer decision")

    return {
        "review_decision": reviewer_response,
        "status": (
            "review_acknowledged"
            if reviewer_response == "acknowledged"
            else "review_rejected"
        ),
    }



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

#enable checkpointing
workflow = builder.compile(checkpointer=InMemorySaver())


# 4. Run it
if __name__ == "__main__":
    config = {
        "configurable": {
            "thread_id": "invoice-review-0050"
        }
    }

    # First invocation: execute until human review
    result = workflow.invoke(
        {"invoice_id": "INV-2025-0050"},
        config=config,
    )

    print("Interrupt:", result.get("__interrupt__"))
    print("Saved status:", workflow.get_state(config).values.get("status"))

    # Simulate a reviewer acknowledging the issue
    resumed = workflow.invoke(
        Command(resume="acknowledged"),
        config=config,
    )

    print("\nAfter resuming:")
    print("Status:", resumed["status"])
    print("Review decision:", resumed["review_decision"])

