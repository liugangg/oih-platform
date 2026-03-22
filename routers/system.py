"""
System status router — exposes platform health for Qwen agent
"""
from fastapi import APIRouter
from core.task_manager import task_manager, TaskStatus
from core.docker_client import check_all_containers
from datetime import datetime, timedelta

router = APIRouter()


@router.get("/status")
async def get_system_status():
    """Return full system status: containers, GPU queue, task summary."""
    containers = await check_all_containers()
    queue = task_manager.queue_size()

    # Count today's tasks
    today = datetime.utcnow().date()
    all_tasks = task_manager.list_tasks()
    running = 0
    completed_today = 0
    failed_today = 0
    for t in all_tasks:
        if t.status == TaskStatus.RUNNING:
            running += 1
        created = t.created_at[:10] if t.created_at else ""
        try:
            task_date = datetime.fromisoformat(created).date()
        except Exception:
            continue
        if task_date == today:
            if t.status == TaskStatus.COMPLETED:
                completed_today += 1
            elif t.status == TaskStatus.FAILED:
                failed_today += 1

    return {
        "containers": containers,
        "gpu_queue": queue,
        "task_summary": {
            "running": running,
            "completed_today": completed_today,
            "failed_today": failed_today,
            "total_tasks": len(all_tasks),
        },
    }
