"""The machine state before each benchmark worker: power, Low Power Mode, thermal notes, versions.

A benchmark on battery power or in Low Power Mode runs at a lower clock, so the driver refuses to
start (Section 3.8 of specs/03-baseline/plan.html).
"""

import platform
import re
import subprocess
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version

from dinov3_apple.memory import available_memory_bytes

VERSIONED_PACKAGES = ["torch", "transformers", "mlx", "numpy", "safetensors"]


class MachineStateError(RuntimeError):
    """The Mac is not in a state for a benchmark: battery power or Low Power Mode."""


@dataclass(frozen=True)
class MachineState:
    """The state of the Mac before one worker.

    Attributes:
        power_source: "AC Power", "Battery Power" or "not reported".
        low_power_mode: "on", "off" or "not reported".
        thermal: The notes of pmset -g therm, joined with "; ".
        available_bytes: The available memory from available_memory_bytes.
        chip: The chip name, for example "Apple M5 Max".
        memory_bytes: The size of the unified memory.
        versions: macOS, Python and package versions.
    """

    power_source: str
    low_power_mode: str
    thermal: str
    available_bytes: int
    chip: str
    memory_bytes: int
    versions: dict[str, str] = field(default_factory=dict)


def _run(command: list[str]) -> str:
    return subprocess.run(command, capture_output=True, text=True, check=True).stdout


def _power_source(batt_text: str) -> str:
    match = re.search(r"Now drawing from '([^']+)'", batt_text)
    return match.group(1) if match else "not reported"


def _low_power_mode(pmset_text: str) -> str:
    """Read the lowpowermode line. Without it, read the powermode line (1 is Low Power Mode)."""
    for key in ("lowpowermode", "powermode"):
        match = re.search(rf"^\s*{key}\s+(\d+)", pmset_text, flags=re.MULTILINE)
        if match:
            return "on" if match.group(1) == "1" else "off"
    return "not reported"


def _thermal(therm_text: str) -> str:
    notes = [line.strip().removeprefix("Note: ") for line in therm_text.splitlines()]
    return "; ".join(note for note in notes if note)


def read_versions() -> dict[str, str]:
    """Return the macOS, Python, torch, transformers, mlx, NumPy and safetensors versions."""
    versions = {"macos": platform.mac_ver()[0], "python": platform.python_version()}
    for package in VERSIONED_PACKAGES:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def read_machine_state(
    batt_text: str | None = None,
    pmset_text: str | None = None,
    therm_text: str | None = None,
) -> MachineState:
    """Read the power source, Low Power Mode, thermal notes, free memory and versions.

    Args:
        batt_text: The output of pmset -g batt. None runs it.
        pmset_text: The output of pmset -g. None runs it.
        therm_text: The output of pmset -g therm. None runs it.

    Returns:
        The machine state.
    """
    batt_text = _run(["pmset", "-g", "batt"]) if batt_text is None else batt_text
    pmset_text = _run(["pmset", "-g"]) if pmset_text is None else pmset_text
    therm_text = _run(["pmset", "-g", "therm"]) if therm_text is None else therm_text
    return MachineState(
        power_source=_power_source(batt_text),
        low_power_mode=_low_power_mode(pmset_text),
        thermal=_thermal(therm_text),
        available_bytes=available_memory_bytes(),
        chip=_run(["sysctl", "-n", "machdep.cpu.brand_string"]).strip(),
        memory_bytes=int(_run(["sysctl", "-n", "hw.memsize"])),
        versions=read_versions(),
    )


def check_machine_state(state: MachineState) -> None:
    """Stop a benchmark on battery power or in Low Power Mode.

    Args:
        state: The state from read_machine_state.

    Raises:
        MachineStateError: If the Mac runs on battery power or Low Power Mode is on.
    """
    if state.power_source == "Battery Power":
        raise MachineStateError("The Mac runs on battery power. Connect the power adapter.")
    if state.low_power_mode == "on":
        raise MachineStateError("Low Power Mode is on. Turn it off.")
