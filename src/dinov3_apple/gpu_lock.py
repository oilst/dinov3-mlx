"""The GPU lock: one heavy job at a time on this Mac (F14 of specs/05-all-sizes/verify.html).

In Step 5, light jobs of other agents overlapped timing runs. The wait loop of the run rules
protected only the agent that ran it. The GPU lock makes the rule code that every entry point runs
(Section 3.3.1 of specs/06-speed/plan.html):

1. heavy_job writes its process id into the registry LOCK_REGISTRY. The entry marks a lock-aware
   process.
2. It takes an exclusive flock on LOCK_PATH. The flock sets the order of the lock-aware jobs.
3. While it holds the flock, it waits until other_heavy_jobs finds no process outside the lock.

A child process of the holder inherits LOCK_ENV and enters at once. macOS releases a flock when its
process ends, so a crash leaves no stale lock. The lock log (LOG_PATH) gets one line for the start
of the wait, one for the start of the hold and one for the end of each job.

Light GPU work runs through the command line:

    uv run python -m dinov3_apple.gpu_lock -- <command>
    uv run python -m dinov3_apple.gpu_lock --timeout 600 -- <command>

No torch, no mlx import.
"""

import argparse
import fcntl
import os
import re
import subprocess
import sys
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import TextIO

# The pattern of the wait loop of Step 5 plus the command line of this module. The brackets keep a
# pgrep of the pattern from a match with its own command line.
HEAVY_JOB_PATTERN = (
    r"[b]in/pytest|[s]cripts/bench|[c]ross_check\.py|[m]ake_reference\.py|[p]rofile_block\.py"
    r"|[d]inov3_apple\.gpu_lock"
)
LOCK_PATH = Path.home() / ".cache" / "dinov3-apple" / "gpu.lock"
LOCK_REGISTRY = LOCK_PATH.parent / "gpu-lock.d"  # one file for each lock-aware process
LOG_PATH = LOCK_PATH.parent / "gpu-lock.log"
LOCK_ENV = "DINOV3_GPU_LOCK"  # the process id of the holder; its child processes inherit it
POLL_S = 20.0


class GpuLockTimeoutError(TimeoutError):
    """A wait for the GPU lock took longer than its time limit."""


def _processes() -> dict[int, tuple[int, str]]:
    """Return the parent process id and the command line of each process, from ps."""
    return {pid: (ppid, command) for pid, (ppid, _, command) in _process_groups().items()}


def _process_groups() -> dict[int, tuple[int, int, str]]:
    """Return the parent process id, the process group and the command line of each process."""
    text = subprocess.run(
        ["ps", "-axww", "-o", "pid=,ppid=,pgid=,command="],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    table = {}
    for row in text.splitlines():
        fields = row.split(None, 3)
        if len(fields) >= 3:
            command = fields[3] if len(fields) == 4 else ""
            table[int(fields[0])] = (int(fields[1]), int(fields[2]), command)
    return table


def _ancestors(pid: int, table: dict[int, tuple[int, str]]) -> set[int]:
    """Return the process id and the process ids of all its ancestors."""
    found: set[int] = set()
    while pid in table and pid not in found:
        found.add(pid)
        pid = table[pid][0]
    return found


def _descendants(pids: set[int], table: dict[int, tuple[int, str]]) -> set[int]:
    """Return the process ids and the process ids of all their children, at all depths."""
    found = set(pids)
    grown = True
    while grown:
        children = {pid for pid, (ppid, _) in table.items() if ppid in found} - found
        found |= children
        grown = bool(children)
    return found


def registered_processes(
    registry: Path = LOCK_REGISTRY, table: dict[int, tuple[int, str]] | None = None
) -> set[int]:
    """Return the process ids of the registry that still run. Remove the entries of the others.

    Args:
        registry: The folder of the registry entries, one file for each process id.
        table: The processes from ps. None runs ps.
    """
    table = _processes() if table is None else table
    alive = set()
    for entry in registry.glob("*") if registry.exists() else []:
        if not entry.name.isdigit():
            continue
        if int(entry.name) in table:
            alive.add(int(entry.name))
        else:
            entry.unlink(missing_ok=True)  # a process that no longer runs counts as absent
    return alive


def other_heavy_jobs(pattern: str = HEAVY_JOB_PATTERN, registry: Path = LOCK_REGISTRY) -> list[str]:
    """Return the command lines of the heavy jobs outside the GPU lock.

    A process outside the lock matches pattern and is none of these: the own process or one of its
    ancestors, a process of the own process group (for example the other commands of the shell
    pipeline that started this job), a registered process, an ancestor or a child of a registered
    process.

    Args:
        pattern: The regular expression of the command lines of heavy jobs.
        registry: The folder of the registry entries.

    Returns:
        The command lines, sorted by process id.
    """
    groups = _process_groups()
    table = {pid: (ppid, command) for pid, (ppid, _, command) in groups.items()}
    own_group = os.getpgid(0)
    registered = registered_processes(registry, table)
    known = _ancestors(os.getpid(), table) | _descendants(registered, table)
    for pid in registered:
        known |= _ancestors(pid, table)
    regex = re.compile(pattern)
    return [
        command
        for pid, (_, command) in sorted(table.items())
        if pid not in known and groups[pid][1] != own_group and regex.search(command)
    ]


def _log_lock(log_path: Path, label: str, event: str, detail: str = "") -> None:
    """Append one line to the lock log: time, process id, event, label and a detail."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    time_ = datetime.now(UTC).isoformat(timespec="milliseconds")
    with log_path.open("a") as file:
        file.write(f"{time_}\tpid {os.getpid()}\t{event}\t{label}\t{detail}".rstrip() + "\n")


def _holds_lock_of_ancestor() -> bool:
    """Return True if LOCK_ENV names this process or one of its ancestors."""
    value = os.environ.get(LOCK_ENV, "")
    if not value.isdigit():
        return False
    return int(value) == os.getpid() or int(value) in _ancestors(os.getpid(), _processes())


def _try_flock(file: TextIO) -> bool:
    try:
        fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    return True


@contextmanager
def heavy_job(
    label: str,
    poll_s: float = POLL_S,
    lock_path: Path = LOCK_PATH,
    registry: Path = LOCK_REGISTRY,
    pattern: str = HEAVY_JOB_PATTERN,
    timeout_s: float | None = None,
    log_path: Path = LOG_PATH,
) -> Iterator[None]:
    """Hold the GPU lock, then wait until no GPU job outside the lock runs (F14).

    The flock comes first: it sets the order of the lock-aware jobs. A lock-aware process writes
    its process id into the registry before it waits, and other_heavy_jobs leaves out each
    registered process with its ancestors and children. So two waiters never wait for each other.
    A child of the holder (LOCK_ENV names an ancestor) enters at once. macOS releases a flock when
    its process ends.

    Args:
        label: The name of the job in the lock log.
        poll_s: The time between two checks of a wait.
        lock_path: The lock file.
        registry: The folder of the registry entries.
        pattern: The regular expression of the command lines of heavy jobs.
        timeout_s: The longest wait. None waits for ever.
        log_path: The lock log.

    Raises:
        GpuLockTimeoutError: If a wait takes longer than timeout_s.
    """
    if _holds_lock_of_ancestor():
        yield
        return
    registry.mkdir(parents=True, exist_ok=True)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    entry = registry / str(os.getpid())
    entry.write_text(label)
    start = time.monotonic()
    _log_lock(log_path, label, "wait")
    try:
        with lock_path.open("a") as file:
            waited_for: set[str] = set()

            def wait(others: list[str]) -> None:
                for other in set(others) - waited_for:
                    _log_lock(log_path, label, "wait for", other)
                    waited_for.add(other)
                sleep_s = poll_s
                if timeout_s is not None:
                    left = timeout_s - (time.monotonic() - start)
                    if left <= 0:
                        _log_lock(log_path, label, "timeout", f"after {timeout_s} s")
                        raise GpuLockTimeoutError(
                            f"{label}: no GPU lock after {timeout_s} s, "
                            f"waits for {'; '.join(others)}"
                        )
                    sleep_s = min(poll_s, left + 0.01)
                time.sleep(sleep_s)

            while not _try_flock(file):
                wait(["the holder of the GPU lock"])
            try:
                while others := other_heavy_jobs(pattern, registry):
                    wait(others)
                _log_lock(log_path, label, "hold")
                os.environ[LOCK_ENV] = str(os.getpid())
                try:
                    yield
                finally:
                    os.environ.pop(LOCK_ENV, None)
                    _log_lock(log_path, label, "end")
            finally:
                fcntl.flock(file, fcntl.LOCK_UN)
    finally:
        entry.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command under the GPU lock and return its exit code.

    Args:
        argv: The arguments: [--timeout S] [--label L] -- command. None reads sys.argv. The
            hidden options --poll, --lock-dir and --pattern are for the tests.

    Returns:
        The exit code of the command, or 1 after a timeout of the wait.
    """
    parser = argparse.ArgumentParser(
        prog="python -m dinov3_apple.gpu_lock",
        description="Run one command under the GPU lock (CLAUDE.md, run rules).",
    )
    parser.add_argument("--timeout", type=float, help="the longest wait in seconds")
    parser.add_argument("--label", help="the name of the job in the lock log")
    parser.add_argument("--poll", type=float, default=POLL_S, help=argparse.SUPPRESS)
    # The tests give a temporary folder for the lock file, the registry and the lock log.
    parser.add_argument("--lock-dir", type=Path, help=argparse.SUPPRESS)
    # The tests give a pattern that matches no process, so that other jobs do not stop them.
    parser.add_argument("--pattern", default=HEAVY_JOB_PATTERN, help=argparse.SUPPRESS)
    parser.add_argument("command", nargs=argparse.REMAINDER, help="-- command [arguments]")
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("give the command after --")
    try:
        folder = args.lock_dir
        paths = (
            {}
            if folder is None
            else {
                "lock_path": folder / LOCK_PATH.name,
                "registry": folder / LOCK_REGISTRY.name,
                "log_path": folder / LOG_PATH.name,
            }
        )
        label = args.label or " ".join(command)
        with heavy_job(
            label, poll_s=args.poll, pattern=args.pattern, timeout_s=args.timeout, **paths
        ):
            return subprocess.run(command, check=False).returncode
    except GpuLockTimeoutError as error:
        print(f"gpu_lock: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
