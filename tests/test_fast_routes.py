"""
快速路由端到端测试 — 独立脚本，不影响正式服务
用法: /data/oih/miniconda/bin/python tests/test_fast_routes.py
"""
import asyncio
import sys
import time
import httpx

API = "http://localhost:8080"

TESTS = [
    # (消息, 期望路由名, 期望工具, 描述)
    ("下载PDB 5XWR", "fetch_pdb", "fetch_pdb", "PDB下载"),
    ("获取分子 aspirin", "fetch_molecule", "fetch_molecule", "小分子获取"),
    ("搜索CD36相关文献", "literature", "search_literature", "文献检索"),
    ("检测5LGD的结合口袋", "pocket", "fpocket_detect_pockets", "口袋检测"),
    ("评估aspirin的ADMET", "admet", "chemprop_predict", "ADMET预测"),
    ("预测1N8Z的B细胞表位", "epitope", "discotope3_predict", "表位预测"),
    ("对接CD36和棕榈酸", "docking", "dock_ligand", "分子对接"),
    ("用AlphaFold3预测TP53的结构", "af3_predict", "alphafold3_predict", "AF3预测"),
]

async def test_one(msg, expect_route, expect_tool, desc, timeout=300):
    """发送 SSE 请求，收集事件，检查路由和工具调用"""
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

                    # 只等到第一个工具提交或结果就停（不等长任务完成）
                    if evt["type"] in ("task_submitted", "tool_result", "answer", "error", "done"):
                        if evt["type"] == "task_submitted":
                            # 收到 task_submitted 说明工具已成功调用，不用等完成
                            events.append({"type": "_early_stop", "reason": "task_submitted"})
                            break
                        if evt["type"] in ("answer", "error", "done"):
                            break
    except Exception as e:
        events.append({"type": "error", "content": str(e)})

    elapsed = time.time() - start

    # 分析结果
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
    print(f"快速路由端到端测试 — {API}")
    print(f"{'='*60}\n")

    # 先检查服务
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{API}/health")
            if r.status_code != 200:
                print("❌ 服务未启动"); return
    except:
        print("❌ 无法连接服务"); return

    results = []
    for msg, expect_route, expect_tool, desc in TESTS:
        ok = await test_one(msg, expect_route, expect_tool, desc, timeout=120)
        results.append(ok)

    passed = sum(results)
    total = len(results)
    print(f"\n{'='*60}")
    print(f"结果: {passed}/{total} passed")
    if passed < total:
        print("失败的测试需要检查工具执行是否正常")
    print(f"{'='*60}")

if __name__ == "__main__":
    asyncio.run(main())
