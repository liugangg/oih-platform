"""
Docker Client Wrapper
Execute commands inside tool containers with unified GPU1 resource scheduling
"""
import asyncio
import json
import logging
from typing import Dict, List, Optional, Tuple
from core.config import settings

logger = logging.getLogger(__name__)

# GPU1 device flag injected into all compute containers
GPU1_ENV = f"CUDA_VISIBLE_DEVICES={settings.COMPUTE_GPU}"


async def run_in_container(
    container: str,
    cmd: List[str],
    timeout: int = 3600,
    env: Optional[Dict] = None,
) -> Tuple[int, str, str]:
    """
    Execute a command inside a running Docker container.
    Returns: (returncode, stdout, stderr)
    """
    env_flags = ["-e", f"CUDA_VISIBLE_DEVICES={settings.COMPUTE_GPU}"]
    if env:
        for k, v in env.items():
            env_flags += ["-e", f"{k}={v}"]

    docker_cmd = ["docker", "exec"] + env_flags + [container] + cmd

    logger.info(f"[{container}] Running: {' '.join(cmd[:6])}...")

    try:
        proc = await asyncio.create_subprocess_exec(
            *docker_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=timeout
        )
        return proc.returncode, stdout.decode(), stderr.decode()
    except asyncio.TimeoutError:
        proc.kill()
        await _kill_container_processes(container, cmd[0] if cmd else "python")
        raise TimeoutError(f"Container {container} timed out after {timeout}s")
    except Exception as e:
        raise RuntimeError(f"Docker exec failed on {container}: {e}")


async def _kill_container_processes(container: str, cmd_pattern: str):
    """Kill processes inside a container matching cmd_pattern.

    `docker exec` creates processes parented to the container's init (containerd-shim),
    NOT to the API process. So killing the host-side `docker exec` wrapper does NOT
    kill the actual process inside the container — it keeps running, holding GPU VRAM.
    This helper sends SIGKILL to matching processes inside the container.
    """
    try:
        kill_cmd = (
            f"for pid in $(ps aux | grep '{cmd_pattern}' | grep -v grep | awk '{{print $2}}'); "
            f"do kill -9 $pid 2>/dev/null; done"
        )
        kill_proc = await asyncio.create_subprocess_exec(
            "docker", "exec", container, "bash", "-c", kill_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await asyncio.wait_for(kill_proc.communicate(), timeout=10)
        logger.info(f"[{container}] Killed container processes matching '{cmd_pattern}'")
    except Exception as e:
        logger.warning(f"[{container}] Failed to kill container processes: {e}")


async def run_in_container_streaming(
    container: str,
    cmd: List[str],
    task,               # Task object for progress updates
    timeout: int = 3600,
) -> Tuple[int, str]:
    """Run with live stdout streaming → update task.progress_msg"""
    docker_cmd = [
        "docker", "exec",
        "-e", "CUDA_VISIBLE_DEVICES=0",
        container
    ] + cmd

    proc = await asyncio.create_subprocess_exec(
        *docker_cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    output_lines = []
    try:
        async def _read():
            async for line in proc.stdout:
                decoded = line.decode().strip()
                output_lines.append(decoded)
                task.progress_msg = decoded
                logger.debug(f"[{container}] {decoded}")

        await asyncio.wait_for(
            asyncio.gather(_read(), proc.wait()),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        proc.kill()
        # Also kill the actual process inside the container (docker exec orphan problem)
        await _kill_container_processes(container, cmd[0] if cmd else "python")
        raise TimeoutError(f"{container} timed out after {timeout}s")

    return proc.returncode, "\n".join(output_lines)


async def check_all_containers() -> Dict[str, str]:
    """Quick health check: is each container running?"""
    containers = [
        settings.CONTAINER_ALPHAFOLD3,
        settings.CONTAINER_RFDIFFUSION,
        settings.CONTAINER_PROTEINMPNN,
        settings.CONTAINER_BINDCRAFT,
        settings.CONTAINER_FPOCKET,
        settings.CONTAINER_P2RANK,
        settings.CONTAINER_VINA_GPU,
        settings.CONTAINER_AUTODOCK_GPU,
        settings.CONTAINER_GNINA,
        settings.CONTAINER_DIFFDOCK,
        settings.CONTAINER_GROMACS,
        settings.CONTAINER_DISCOTOPE3,
        settings.CONTAINER_IGFOLD,
        settings.CONTAINER_ESM,
        settings.CONTAINER_CHEMPROP,
    ]
    result = {}
    for c in containers:
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "inspect", "--format={{.State.Status}}", c,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
            result[c] = stdout.decode().strip()
        except Exception:
            result[c] = "unknown"
    return result
