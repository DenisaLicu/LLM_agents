from agent import run_agent

TEST_CASES = [
    {
        "invoice_id": "INV-2025-0042",
        "expected_tools": ["get_invoice", "get_supplier"],
        "expected_review": False,
        "expected_supplier_status": "active",
    },
    {
        "invoice_id": "INV-2025-0050",
        "expected_tools": ["get_invoice", "get_supplier"],
        "expected_review": True,
        "expected_supplier_status": "inactive",
    },
    {
        "invoice_id": "INV-2025-0099",
        "expected_tools": ["get_invoice"],
        "expected_review": True,
        "expected_supplier_status": None,
    },
]


def evaluate(case: dict) -> dict:
    result = run_agent(case["invoice_id"])

    actual_tools = [
        item["tool"] for item in result["trace"]
    ]

    supplier = result["supplier"]
    actual_supplier_status = (
        supplier.get("status") if supplier else None
    )

    checks = {
        "tool_sequence": actual_tools == case["expected_tools"],
        "review_decision": (
            result["decision"]["requires_review"]
            == case["expected_review"]
        ),
        "supplier_status": (
            actual_supplier_status
            == case["expected_supplier_status"]
        ),
    }

    return {
        "invoice_id": case["invoice_id"],
        "checks": checks,
        "passed": all(checks.values()),
        "answer": result["answer"],
    }


if __name__ == "__main__":
    results = [evaluate(case) for case in TEST_CASES]

    for result in results:
        status = "PASS" if result["passed"] else "FAIL"
        print(f"\n{status} | {result['invoice_id']}")

        for check, passed in result["checks"].items():
            print(f"  {'PASS' if passed else 'FAIL'} | {check}")

    passed_count = sum(r["passed"] for r in results)
    print(f"\nOverall: {passed_count}/{len(results)} passed")
