
from importlib import import_module

run_langchain_agent = import_module("3_langchain_agent").run_langchain_agent
TEST_CASES = import_module("eval_agent").TEST_CASES


passed = 0

for case in TEST_CASES:
    result = run_langchain_agent(case["invoice_id"])

    actual_tools = [
        item["tool"] for item in result["trace"]
    ]

    expected_tools = [
        name.replace("get_", "lookup_")
        for name in case["expected_tools"]
    ]

    actual_status = (
        result["supplier"].get("status")
        if result["supplier"]
        else None
    )

    checks = {
        "tool_sequence": actual_tools == expected_tools,
        "review_decision": (
            result["decision"]["requires_review"]
            == case["expected_review"]
        ),
        "supplier_status": (
            actual_status
            == case["expected_supplier_status"]
        ),
    }

    success = all(checks.values())
    passed += success

    print(
        f"{'PASS' if success else 'FAIL'} "
        f"| {case['invoice_id']}"
    )

    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'} | {name}")

print(f"\nOverall: {passed}/{len(TEST_CASES)} passed")
