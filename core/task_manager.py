"""
Three-Queue Task Manager
========================
- CPU queue:      Semaphore(8)  — fpocket, P2Rank, Chemprop
- GPU queue:      Semaphore(1)  — GNINA, AutoDock, DiffDock, AF3, RFdiffusion,
                                  ProteinMPNN, BindCraft, GROMACS, ESM
- Degraded queue: Semaphore(4)  — auto-fallback when GPU1 VRAM is insufficient

Submit flow:
  submit(tool, ...) → _resolve_queue(tool)
      CPU tool?        → cpu_sem
      GPU tool + VRAM OK?  → gpu_sem
      GPU tool + VRAM low? → degraded_sem  (logs warning)
      unknown tool?    → gpu_sem (safe default)
"""
import asyncio
import json
import os
import uuid
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_TASKS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "tasks")


# ─── Enums ────────────────────────────────────────────────────────────────────

class TaskStatus(str, Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    COMPLETED = "completed"
    FAILED    = "failed"
    CANCELLED = "cancelled"


class QueueType(str, Enum):
    CPU      = "cpu"
    GPU      = "gpu"
    DEGRADED = "degraded"


# ─── Queue routing tables ─────────────────────────────────────────────────────

_CPU_TOOLS: frozenset = frozenset({
    "fpocket", "p2rank", "chemprop", "chemprop_predict", "freesasa", "rdkit_conjugate",
    "extract_interface",
})

_GPU_TOOLS: frozenset = frozenset({
    "gnina", "autodock-gpu", "vina-gpu", "diffdock",
    "alphafold3", "rfdiffusion", "proteinmpnn", "bindcraft",
    "gromacs", "esm", "discotope3", "igfold",
    # Pipelines (sequential GPU tool usage inside)
    "pocket_guided_binder_pipeline", "binder_design_pipeline",
    "drug_discovery_pipeline",
})

# Minimum free VRAM (MB) needed to run a tool on GPU.
# GPU1 is an RTX 4090 with ~45 GB VRAM.
_VRAM_REQUIRED_MB: Dict[str, int] = {
    "alphafold3":  20000,
    "bindcraft":   16000,
    "diffdock":     8000,
    "rfdiffusion":  8000,
    "gromacs":      6000,
    "esm":          6000,
    "gnina":        4000,
    "proteinmpnn":  4000,
    "autodock-gpu": 2000,
    "vina-gpu":     2000,
    "chemprop":     4000,
    "discotope3":   6000,
    "igfold":       4000,
    # Pipelines: use AF3-level VRAM (highest sub-tool requirement)
    "pocket_guided_binder_pipeline": 20000,
    "binder_design_pipeline":        20000,
    "drug_discovery_pipeline":       20000,
}

# Tools that must NEVER go to DEGRADED queue — they require real GPU and will
# crash/OOM on CPU fallback. When GPU VRAM is insufficient, these tools wait
# for GPU slots to free up instead of being degraded.
_NO_DEGRADED_TOOLS: frozenset = frozenset({
    "alphafold3", "bindcraft",
})


# ─── Task dataclass ───────────────────────────────────────────────────────────

@dataclass
class Task:
    task_id: str
    tool: str
    queue: QueueType = QueueType.GPU
    status: TaskStatus = TaskStatus.PENDING
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    result: Optional[Dict] = None
    error: Optional[str] = None
    progress: int = 0
    progress_msg: str = ""
    input_params: Dict = field(default_factory=dict)
    output_files: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "task_id":      self.task_id,
            "tool":         self.tool,
            "queue":        self.queue,
            "status":       self.status,
            "created_at":   self.created_at,
            "started_at":   self.started_at,
            "completed_at": self.completed_at,
            "result":       self.result,
            "error":        self.error,
            "progress":     self.progress,
            "progress_msg": self.progress_msg,
            "output_files": self.output_files,
        }


# ─── TaskManager ──────────────────────────────────────────────────────────────

class TaskManager:
    """
    Three-queue async task manager.

    Semaphore limits:
      CPU      → 8  concurrent  (lightweight: fpocket, P2Rank, Chemprop)
      GPU      → 1  concurrent  (one at a time — RTX 4090 44GB, AF3 alone needs 20GB)
      Degraded → 4  concurrent  (CPU+RAM fallback when VRAM insufficient)
    """

    def __init__(self):
        self._tasks: Dict[str, Task] = {}

        # Semaphores (safe to create outside async context in Python 3.10+)
        self._cpu_sem      = asyncio.Semaphore(8)
        self._gpu_sem      = asyncio.Semaphore(1)   # RTX 4090 44GB — one GPU task at a time to prevent OOM
        self._degraded_sem = asyncio.Semaphore(4)

        # Live active-slot counters (incremented inside semaphore, decremented on exit)
        self._cpu_active:      int = 0
        self._gpu_active:      int = 0
        self._degraded_active: int = 0

        # Load historical tasks from disk
        self._load_history()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _persist(self, task: Task):
        """Write task state to disk as JSON. Best-effort, never raises."""
        try:
            os.makedirs(_TASKS_DIR, exist_ok=True)
            path = os.path.join(_TASKS_DIR, f"{task.task_id}.json")
            data = task.to_dict()
            data["updated_at"] = datetime.utcnow().isoformat()
            with open(path, "w") as f:
                json.dump(data, f, default=str, ensure_ascii=False)
        except Exception as e:
            logger.warning("[TaskManager] _persist(%s) failed: %s", task.task_id[:8], e)

    def _load_history(self):
        """Load all task JSON files from disk into memory on startup."""
        if not os.path.isdir(_TASKS_DIR):
            return
        loaded = 0
        for fname in os.listdir(_TASKS_DIR):
            if not fname.endswith(".json"):
                continue
            try:
                with open(os.path.join(_TASKS_DIR, fname)) as f:
                    d = json.load(f)
                task = Task(
                    task_id=d["task_id"],
                    tool=d.get("tool", "unknown"),
                    queue=d.get("queue", QueueType.GPU),
                    status=d.get("status", TaskStatus.COMPLETED),
                    created_at=d.get("created_at", ""),
                    started_at=d.get("started_at"),
                    completed_at=d.get("completed_at"),
                    result=d.get("result"),
                    error=d.get("error"),
                    progress=d.get("progress", 0),
                    progress_msg=d.get("progress_msg", ""),
                )
                # Running/pending tasks from a previous server session are dead
                if task.status in (TaskStatus.RUNNING, TaskStatus.PENDING):
                    task.status = TaskStatus.FAILED
                    task.error = "Server restarted while task was running"
                self._tasks[task.task_id] = task
                loaded += 1
            except Exception as e:
                logger.warning("[TaskManager] Failed to load %s: %s", fname, e)
        if loaded:
            logger.info("[TaskManager] Loaded %d historical tasks from disk", loaded)

    # ── VRAM helper ───────────────────────────────────────────────────────────

    async def _free_vram_mb(self) -> int:
        """Query free VRAM on host GPU1 via nvidia-smi. Returns MB."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
                "--id=1",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
            lines = stdout.decode().strip().splitlines()
            return int(lines[0].strip())
        except Exception as e:
            logger.warning(f"[TaskManager] nvidia-smi query failed: {e} — assuming unlimited VRAM")
            return 99999

    # ── Queue resolution ──────────────────────────────────────────────────────

    async def _resolve_queue(self, tool: str) -> QueueType:
        """
        Determine which queue to use:
          - CPU tools       → CPU queue (no VRAM check needed)
          - GPU tools       → GPU queue if VRAM sufficient, else DEGRADED
          - _NO_DEGRADED_TOOLS → wait for GPU VRAM (poll every 60s), never degrade
          - unknown tools   → GPU queue (conservative default)
        """
        if tool in _CPU_TOOLS:
            return QueueType.CPU

        if tool in _GPU_TOOLS:
            required_mb = _VRAM_REQUIRED_MB.get(tool, 4000)
            free_mb = await self._free_vram_mb()
            if free_mb >= required_mb:
                return QueueType.GPU

            # Tools that must not be degraded — wait for GPU VRAM to free up
            if tool in _NO_DEGRADED_TOOLS:
                logger.warning(
                    f"[TaskManager] VRAM check: {tool} needs {required_mb} MB, "
                    f"only {free_mb} MB free on GPU1 — waiting for GPU (no degraded fallback)"
                )
                while True:
                    await asyncio.sleep(60)
                    free_mb = await self._free_vram_mb()
                    if free_mb >= required_mb:
                        logger.info(
                            f"[TaskManager] VRAM now {free_mb} MB free — "
                            f"{tool} proceeding to GPU queue"
                        )
                        return QueueType.GPU
                    logger.info(
                        f"[TaskManager] {tool} still waiting: need {required_mb} MB, "
                        f"have {free_mb} MB free on GPU1"
                    )

            logger.warning(
                f"[TaskManager] VRAM check: {tool} needs {required_mb} MB, "
                f"only {free_mb} MB free on GPU1 → routing to DEGRADED queue"
            )
            return QueueType.DEGRADED

        # Unknown tool — default to GPU queue
        logger.warning(f"[TaskManager] Unknown tool '{tool}', defaulting to GPU queue")
        return QueueType.GPU

    def _sem_and_counter(self, queue: QueueType):
        """Return (semaphore, counter_attr_name) for a queue type."""
        return {
            QueueType.CPU:      (self._cpu_sem,      "_cpu_active"),
            QueueType.GPU:      (self._gpu_sem,      "_gpu_active"),
            QueueType.DEGRADED: (self._degraded_sem, "_degraded_active"),
        }[queue]

    # ── Public API ────────────────────────────────────────────────────────────

    def create_task(self, tool: str, input_params: Dict, queue: QueueType) -> Task:
        task_id = str(uuid.uuid4())
        task = Task(task_id=task_id, tool=tool, queue=queue, input_params=input_params)
        self._tasks[task_id] = task
        self._persist(task)
        return task

    def get_task(self, task_id: str) -> Optional[Task]:
        return self._tasks.get(task_id)

    def list_tasks(self, tool: Optional[str] = None) -> List[Task]:
        tasks = list(self._tasks.values())
        if tool:
            tasks = [t for t in tasks if t.tool == tool]
        return sorted(tasks, key=lambda t: t.created_at, reverse=True)

    def queue_size(self) -> Dict:
        """Return live queue statistics (used by /health endpoint)."""
        pending = [t for t in self._tasks.values() if t.status == TaskStatus.PENDING]
        return {
            "pending_total":   len(pending),
            "cpu_active":      self._cpu_active,
            "gpu_active":      self._gpu_active,
            "degraded_active": self._degraded_active,
            "cpu_slots":       8,
            "gpu_slots":       3,
            "degraded_slots":  4,
        }

    async def submit(self, tool: str, input_params: Dict, coro_fn: Callable) -> Task:
        """
        Submit a coroutine as a tracked async task.

        The correct queue is chosen automatically based on `tool` name and
        current GPU VRAM availability. Returns immediately with a Task whose
        status starts at PENDING; the actual work runs in the background.
        """
        queue = await self._resolve_queue(tool)
        task  = self.create_task(tool, input_params, queue)
        sem, counter_attr = self._sem_and_counter(queue)

        async def _run():
            async with sem:
                # Increment active counter inside the semaphore
                setattr(self, counter_attr, getattr(self, counter_attr) + 1)
                task.status     = TaskStatus.RUNNING
                task.started_at = datetime.utcnow().isoformat()
                self._persist(task)
                logger.info(
                    f"[{queue.value.upper()}] START {tool}/{task.task_id[:8]} "
                    f"| cpu={self._cpu_active}/{8} "
                    f"gpu={self._gpu_active}/{3} "
                    f"deg={self._degraded_active}/{4}"
                )
                try:
                    result         = await coro_fn(task)
                    task.result    = result
                    task.status    = TaskStatus.COMPLETED
                    task.progress  = 100
                    self._persist(task)
                except Exception as e:
                    task.status = TaskStatus.FAILED
                    task.error  = str(e)
                    self._persist(task)
                    logger.error(f"[{queue.value.upper()}] FAIL {tool}/{task.task_id[:8]}: {e}")
                finally:
                    setattr(self, counter_attr, getattr(self, counter_attr) - 1)
                    task.completed_at = datetime.utcnow().isoformat()
                    logger.info(
                        f"[{queue.value.upper()}] DONE {tool}/{task.task_id[:8]} "
                        f"| cpu={self._cpu_active}/{8} "
                        f"gpu={self._gpu_active}/{3} "
                        f"deg={self._degraded_active}/{4}"
                    )

        asyncio.create_task(_run())
        return task

    async def cancel_task(self, task_id: str) -> bool:
        task = self.get_task(task_id)
        if not task:
            return False
        if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
            return False  # already finished
        task.status = TaskStatus.CANCELLED
        task.completed_at = datetime.utcnow().isoformat()
        task.error = "Cancelled by user"
        self._persist(task)
        return True


# ─── Singleton ────────────────────────────────────────────────────────────────

task_manager = TaskManager()
