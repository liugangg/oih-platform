"""
Three-Queue Concurrency Test
============================
Verify TaskManager three-queue concurrency control:
  - Submit N fpocket requests (CPU queue, limit=8)
  - Poll /health in real-time to record peak concurrency
  - Assert: peak_cpu_active <= 8

Usage:
  python tests/test_queue_concurrency.py [--n 10] [--api http://localhost:8080]
"""
import argparse
import asyncio
import time
import sys
import httpx


API = "http://localhost:8080"
# 5XWR was fetched earlier; this path is reliable
PDB_PATH = "/data/oih/outputs/fetch_pdb/5XWR.pdb"


async def submit_fpocket(client: httpx.AsyncClient, idx: int) -> dict:
    """Submit a single fpocket job and wait for completion."""
    payload = {
        "job_name": f"conctest_{idx:02d}",
        "input_pdb": PDB_PATH,
        "min_sphere_size": 3.0,
        "min_druggability_score": 0.0,
    }
    r = await client.post(f"{API}/api/v1/pocket/fpocket", json=payload, timeout=30)
    r.raise_for_status()
    data = r.json()
    task_id = data["task_id"]

    # Poll until done
    for _ in range(60):
        await asyncio.sleep(2)
        r2 = await client.get(f"{API}/api/v1/tasks/{task_id}", timeout=10)
        t = r2.json()
        if t["status"] in ("completed", "failed", "cancelled"):
            return t
    return {"task_id": task_id, "status": "timeout"}


async def poll_health(client: httpx.AsyncClient, stop_event: asyncio.Event,
                      samples: list, interval: float = 0.5):
    """Continuously sample /health and record cpu_active."""
    while not stop_event.is_set():
        try:
            r = await client.get(f"{API}/health", timeout=5)
            q = r.json().get("task_queue", {})
            if isinstance(q, dict):
                samples.append({
                    "ts":              time.monotonic(),
                    "cpu_active":      q.get("cpu_active", 0),
                    "gpu_active":      q.get("gpu_active", 0),
                    "degraded_active": q.get("degraded_active", 0),
                    "pending":         q.get("pending_total", 0),
                })
        except Exception:
            pass
        await asyncio.sleep(interval)


async def run_test(n: int, api: str):
    global API
    API = api

    print(f"\n{'='*60}")
    print(f"  Three-Queue Concurrency Test")
    print(f"  Submitting {n} fpocket jobs (CPU queue, limit=8)")
    print(f"  API: {API}")
    print(f"{'='*60}\n")

    health_samples = []
    stop_event     = asyncio.Event()

    async with httpx.AsyncClient(timeout=120) as client:
        # Verify service is up
        try:
            r = await client.get(f"{API}/health")
            r.raise_for_status()
            q = r.json().get("task_queue", {})
            print(f"[✓] Service healthy | queue: {q}\n")
        except Exception as e:
            print(f"[✗] Service unreachable: {e}")
            sys.exit(1)

        # Start health poller in background
        poller = asyncio.create_task(
            poll_health(client, stop_event, health_samples)
        )

        t0 = time.monotonic()

        # Fire all N fpocket requests concurrently
        print(f"[→] Firing {n} concurrent fpocket requests...")
        tasks = [submit_fpocket(client, i) for i in range(n)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        elapsed = time.monotonic() - t0

        # Stop poller
        stop_event.set()
        await poller

    # ── Analysis ──────────────────────────────────────────────────────────────

    print(f"\n{'─'*60}")
    print(f"  Results ({elapsed:.1f}s total)")
    print(f"{'─'*60}")

    completed = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "completed")
    failed    = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "failed")
    errors    = sum(1 for r in results if isinstance(r, Exception))

    print(f"  completed: {completed}/{n}")
    print(f"  failed:    {failed}/{n}")
    print(f"  errors:    {errors}/{n}")

    if health_samples:
        peak_cpu      = max(s["cpu_active"]      for s in health_samples)
        peak_gpu      = max(s["gpu_active"]       for s in health_samples)
        peak_degraded = max(s["degraded_active"]  for s in health_samples)
        peak_pending  = max(s["pending"]           for s in health_samples)
        print(f"\n  Peak concurrency observed:")
        print(f"    CPU queue active:      {peak_cpu:2d}  (limit 8)")
        print(f"    GPU queue active:      {peak_gpu:2d}  (limit 3)")
        print(f"    Degraded queue active: {peak_degraded:2d}  (limit 4)")
        print(f"    Peak pending:          {peak_pending:2d}")
    else:
        peak_cpu = 0
        print("  (no health samples captured)")

    # ── Timeline dump ─────────────────────────────────────────────────────────
    if health_samples:
        print(f"\n  Health timeline (cpu_active per sample):")
        # Print a bar chart
        for s in health_samples:
            bar = "█" * s["cpu_active"]
            pend = f"+{s['pending']}p" if s["pending"] else ""
            print(f"    t={s['ts']-health_samples[0]['ts']:5.1f}s  "
                  f"CPU:{s['cpu_active']:2d} {bar}{pend}")

    # ── Assertions ────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    ok = True

    def check(label, cond, detail=""):
        nonlocal ok
        icon = "✓" if cond else "✗"
        print(f"  [{icon}] {label}" + (f" ({detail})" if detail else ""))
        if not cond:
            ok = False

    check(f"CPU concurrency ≤ 8", peak_cpu <= 8,
          f"peak={peak_cpu}")
    check(f"All {n} jobs completed", completed == n,
          f"completed={completed}")
    check(f"No exceptions", errors == 0,
          f"errors={errors}")
    if n > 8:
        check(f"Semaphore actually throttled (peak < n)", peak_cpu < n,
              f"peak={peak_cpu}, n={n}")
    elif n <= 8:
        check(f"All {n} ran concurrently (peak == n)", peak_cpu == n,
              f"peak={peak_cpu}, n={n}")

    print(f"{'─'*60}")
    print(f"  {'PASS ✓' if ok else 'FAIL ✗'}")
    print(f"{'='*60}\n")
    return 0 if ok else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",   type=int, default=5,
                        help="Number of concurrent fpocket requests (default 5)")
    parser.add_argument("--api", default="http://localhost:8080",
                        help="API base URL")
    args = parser.parse_args()
    sys.exit(asyncio.run(run_test(args.n, args.api)))


if __name__ == "__main__":
    main()
