"""
Task Management Router
Poll task status, view results, cancel tasks
"""
from fastapi import APIRouter, HTTPException
from typing import Optional
from core.task_manager import task_manager

router = APIRouter()


@router.get("/{task_id}")
async def get_task(task_id: str):
    """Poll task status and result. Called by Qwen after submitting a job."""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    return task.to_dict()


@router.get("/")
async def list_tasks(tool: Optional[str] = None, limit: int = 20):
    """List recent tasks, optionally filtered by tool name."""
    tasks = task_manager.list_tasks(tool=tool)[:limit]
    return {
        "tasks": [t.to_dict() for t in tasks],
        "total": len(tasks),
        "queue_size": task_manager.queue_size(),
    }


@router.delete("/{task_id}")
async def cancel_task(task_id: str):
    """Cancel a pending or running task."""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    cancelled = await task_manager.cancel_task(task_id)
    if not cancelled:
        raise HTTPException(status_code=400, detail="Task already finished")
    return {"cancelled": True, "task_id": task_id}
