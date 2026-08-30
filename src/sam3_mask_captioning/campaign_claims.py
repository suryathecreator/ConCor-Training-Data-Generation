from __future__ import annotations

import fcntl
import json
import os
import socket
import subprocess
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

from .io_utils import write_json


ACTIVE_SLURM_STATES = frozenset(
    {"RUNNING", "COMPLETING", "CONFIGURING", "SUSPENDED", "STOPPED"}
)


class ClaimOwnershipLost(RuntimeError):
    """Raised when a worker tries to commit after losing its fencing token."""


@dataclass(frozen=True)
class ClaimHandle:
    campaign_root: Path
    stage: str
    unit_id: int
    path: Path
    token: str
    generation: int
    worker_id: str


def claim_path(campaign_root: str | Path, stage: str, unit_id: int) -> Path:
    return Path(campaign_root) / "claims" / stage / f"{unit_id:06d}.claim"


def _stage_lock_path(campaign_root: str | Path, stage: str) -> Path:
    return Path(campaign_root) / "claims" / stage / ".lock"


@contextmanager
def stage_claim_lock(campaign_root: str | Path, stage: str) -> Iterator[None]:
    """Serialize claim inspection and replacement across a campaign stage."""
    path = _stage_lock_path(campaign_root, stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _read_claim(path: Path) -> dict[str, object]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def slurm_job_state(job_id: str) -> str:
    """Return active, inactive, or unknown without treating scheduler errors as death."""
    if not job_id or job_id == "local":
        return "unknown"
    try:
        result = subprocess.run(
            ["squeue", "-h", "-j", job_id, "-o", "%T"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if result.returncode != 0:
        return "unknown"
    states = {line.strip().upper() for line in result.stdout.splitlines() if line.strip()}
    return "active" if states & ACTIVE_SLURM_STATES else "inactive"


def _local_process_is_active(payload: dict[str, object]) -> bool | None:
    if str(payload.get("hostname") or "") != socket.gethostname():
        return None
    try:
        pid = int(payload.get("pid") or 0)
    except (TypeError, ValueError):
        return None
    if pid <= 0:
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _allocation_id(worker_id: str) -> str:
    return worker_id.rsplit(":", 1)[0] if ":" in worker_id else worker_id


def _is_local_allocation(allocation: str) -> bool:
    return allocation == "local" or allocation.startswith("local:")


def _scheduler_job_id(worker_id: str) -> str:
    allocation = _allocation_id(worker_id)
    if not allocation or _is_local_allocation(allocation):
        return "local" if allocation else ""
    job, separator, task = allocation.partition(":")
    return f"{job}_{task}" if separator and task else job


def try_claim(
    campaign_root: str | Path,
    stage: str,
    unit_id: int,
    *,
    worker_id: str,
    lease_seconds: int,
    orphan_grace_seconds: int = 120,
    now: Callable[[], float] = time.time,
    job_state: Callable[[str], str] = slurm_job_state,
) -> ClaimHandle | None:
    """Acquire one fenced unit claim.

    The lock covers both the stale decision and replacement. This avoids the
    former TOCTOU race where a late reclaimer could replace a fresh claim.
    """
    root = Path(campaign_root).expanduser().resolve()
    path = claim_path(root, stage, unit_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with stage_claim_lock(root, stage):
        success = root / "units" / f"{unit_id:06d}" / "stages" / stage / "_SUCCESS.json"
        if success.exists():
            return None
        timestamp = now()
        prior = _read_claim(path) if path.exists() else {}
        generation = int(prior.get("generation") or 0)
        if path.exists():
            prior_worker = str(prior.get("worker_id") or "")
            if prior_worker == worker_id:
                return None
            prior_allocation = str(prior.get("allocation_id") or _allocation_id(prior_worker))
            current_allocation = _allocation_id(worker_id)
            prior_job = str(prior.get("scheduler_job_id") or _scheduler_job_id(prior_worker))
            heartbeat = float(
                prior.get("heartbeat_at") or prior.get("claimed_at") or path.stat().st_mtime
            )
            age = max(0.0, timestamp - heartbeat)
            same_requeued_job = (
                bool(current_allocation)
                and not _is_local_allocation(current_allocation)
                and prior_allocation == current_allocation
                and prior_worker != worker_id
            )
            reclaim = same_requeued_job
            grace_elapsed = age >= max(0, orphan_grace_seconds)
            if prior_job in {"", "local"}:
                local_active = _local_process_is_active(prior)
                reclaim = reclaim or (local_active is False and grace_elapsed)
                if local_active is None:
                    reclaim = reclaim or age >= max(1, lease_seconds)
            elif grace_elapsed:
                reclaim = reclaim or job_state(prior_job) == "inactive"
            if not reclaim:
                return None
            stale = path.with_name(
                f"{path.name}.stale.{time.time_ns()}.{str(prior.get('token') or 'legacy')[:12]}"
            )
            os.replace(path, stale)
            generation += 1

        token = uuid.uuid4().hex
        payload = {
            "schema_version": 2,
            "worker_id": worker_id,
            "allocation_id": _allocation_id(worker_id),
            "scheduler_job_id": _scheduler_job_id(worker_id),
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "token": token,
            "generation": generation,
            "claimed_at": timestamp,
            "heartbeat_at": timestamp,
        }
        write_json(payload, path)
        return ClaimHandle(root, stage, unit_id, path, token, generation, worker_id)


def claim_is_owned(handle: ClaimHandle) -> bool:
    with stage_claim_lock(handle.campaign_root, handle.stage):
        payload = _read_claim(handle.path)
        return (
            str(payload.get("token") or "") == handle.token
            and str(payload.get("worker_id") or "") == handle.worker_id
        )


def assert_claim_owned(handle: ClaimHandle) -> None:
    if not claim_is_owned(handle):
        raise ClaimOwnershipLost(
            f"Lost {handle.stage} claim for unit {handle.unit_id:06d} "
            f"(token {handle.token[:12]})"
        )


def renew_claim(handle: ClaimHandle) -> bool:
    with stage_claim_lock(handle.campaign_root, handle.stage):
        payload = _read_claim(handle.path)
        if (
            str(payload.get("token") or "") != handle.token
            or str(payload.get("worker_id") or "") != handle.worker_id
        ):
            return False
        payload["heartbeat_at"] = time.time()
        write_json(payload, handle.path)
        return True


def release_claim(handle: ClaimHandle) -> bool:
    with stage_claim_lock(handle.campaign_root, handle.stage):
        payload = _read_claim(handle.path)
        if (
            str(payload.get("token") or "") != handle.token
            or str(payload.get("worker_id") or "") != handle.worker_id
        ):
            return False
        try:
            handle.path.unlink()
        except FileNotFoundError:
            return False
        return True


def run_if_claim_owned(handle: ClaimHandle, action: Callable[[], None]) -> bool:
    """Run a short claim-owned metadata action without a reclaim race."""
    with stage_claim_lock(handle.campaign_root, handle.stage):
        payload = _read_claim(handle.path)
        if (
            str(payload.get("token") or "") != handle.token
            or str(payload.get("worker_id") or "") != handle.worker_id
        ):
            return False
        action()
        return True


def finish_claim(handle: ClaimHandle, cleanup: Callable[[], None]) -> bool:
    """Run ownership-sensitive cleanup and release without a reclaim gap."""
    with stage_claim_lock(handle.campaign_root, handle.stage):
        payload = _read_claim(handle.path)
        if (
            str(payload.get("token") or "") != handle.token
            or str(payload.get("worker_id") or "") != handle.worker_id
        ):
            return False
        try:
            cleanup()
        finally:
            try:
                handle.path.unlink()
            except FileNotFoundError:
                pass
        return True


def commit_claim_json(
    handle: ClaimHandle,
    path: str | Path,
    payload: dict[str, object],
) -> None:
    """Atomically publish a commit marker while the fencing token is owned."""
    target = Path(path)
    with stage_claim_lock(handle.campaign_root, handle.stage):
        claim = _read_claim(handle.path)
        if (
            str(claim.get("token") or "") != handle.token
            or str(claim.get("worker_id") or "") != handle.worker_id
        ):
            raise ClaimOwnershipLost(
                f"Lost {handle.stage} claim for unit {handle.unit_id:06d} "
                f"before commit (token {handle.token[:12]})"
            )
        if target.exists():
            raise FileExistsError(f"Refusing to replace existing stage commit: {target}")
        write_json(payload, target)
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)


class ClaimHeartbeat:
    def __init__(self, handle: ClaimHandle, interval_seconds: float = 60.0):
        self.handle = handle
        self.interval_seconds = max(0.05, float(interval_seconds))
        self._stop = threading.Event()
        self._lost = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> "ClaimHeartbeat":
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._run,
                name=f"claim-heartbeat-{self.handle.stage}-{self.handle.unit_id:06d}",
                daemon=True,
            )
            self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            if not renew_claim(self.handle):
                self._lost.set()
                return

    def assert_owned(self) -> None:
        if self._lost.is_set():
            raise ClaimOwnershipLost(
                f"Heartbeat lost {self.handle.stage} unit {self.handle.unit_id:06d}"
            )
        assert_claim_owned(self.handle)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
