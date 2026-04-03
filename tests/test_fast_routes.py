"""
Fast route end-to-end test -- standalone script, does not affect production service
Usage: /data/oih/miniconda/bin/python tests/test_fast_routes.py
"""
import asyncio
import sys
import time
import httpx

API = "http://localhost:8080"

TESTS = [
    # (message, expected_route, expected_tool, description)
    ("Download PDB 5XWR", "fetch_pdb", "fetch_pdb", "PDB_download"),
    ("Fetch molecule aspirin", "fetch_molecule", "fetch_molecule", "molecule_fetch"),
    ("Search CD36 related literature", "literature", "search_literature", "lit_search"),
    ("Detect binding pockets of 5LGD", "pocket", "fpocket_detect_pockets", "pocket_detect"),
    ("Evaluate ADMET of aspirin", "admet", "chemprop_predict", "ADMET_predict"),
    ("Predict B-cell epitopes of 1N8Z", "epitope", "discotope3_predict", "epitope_pred"),
    ("Dock CD36 with palmitic acid", "docking", "dock_ligand", "docking"),
    ("Predict TP53 structure with AlphaFold3", "af3_predict", "alphafold3_predict", "AF3_predict"),
]

async def test_one(msg, expect_route, expect_tool, desc, timeout=300):
    """Send SSE request, collect events, verify route and tool calls"""
    session_id = f"test_{int(time.time())}_{desc}"
    start = time.time()
    events = []

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST", f"{API}/api/v1/agent/chat/stream",
                json={"message": msg, "session_id": session_id},
                headers={"Content-Type": "application/json"},
            ) as resp:
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    import json
                    evt = json.loads(line[6:])
                    events.append(evt)

                    # Stop at first tool submission or result (don't wait for long tasks)
                    if evt["type"] in ("task_submitted", "tool_result", "answer", "error", "done"):
                        if evt["type"] == "task_submitted":
                            # task_submitted means tool was successfully invoked, no need to wait for completion
                            events.append({"type": "_early_stop", "reason": "task_submitted"})
                            break
                        if evt["type"] in ("answer", "error", "done"):
                            break
    except Exception as e:
        events.append({"type": "error", "content": str(e)})

    elapsed = time.time() - start

    # Analyze results
    is_fast = any(e.get("content", "").startswith("⚡") for e in events if e["type"] == "status")
    tools_called = [e["tool"] for e in events if e["type"] == "tool_call"]
    tasks_submitted = [e["tool"] for e in events if e["type"] == "task_submitted"]
    errors = [e for e in events if e["type"] == "error"]

    ok = is_fast and (expect_tool in tools_called or expect_tool in tasks_submitted)

    status = "✅" if ok else "❌"
    print(f"{status} [{elapsed:5.1f}s] {desc:12s} | fast_route={'Y' if is_fast else 'N'} | tools={tools_called} | tasks={tasks_submitted}")
    if errors:
        print(f"   ⚠️  errors: {[e.get('content','')[:80] for e in errors]}")

    return ok

async def main():
    print(f"{'='*60}")
    print(f"Fast route end-to-end test -- {API}")
    print(f"{'='*60}\n")

    # Check service availability
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{API}/health")
            if r.status_code != 200:
                print("Service not running"); return
    except:
        print("Cannot connect to service"); return

    results = []
    for msg, expect_route, expect_tool, desc in TESTS:
        ok = await test_one(msg, expect_route, expect_tool, desc, timeout=120)
        results.append(ok)

    passed = sum(results)
    total = len(results)
    print(f"\n{'='*60}")
    print(f"Result: {passed}/{total} passed")
    if passed < total:
        print("Failed tests need investigation of tool execution")
    print(f"{'='*60}")

if __name__ == "__main__":
    asyncio.run(main())
