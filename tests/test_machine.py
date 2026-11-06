import pytest

from dinov3_apple.machine import (
    MachineState,
    MachineStateError,
    check_machine_state,
    read_machine_state,
)

BATT_AC = """Now drawing from 'AC Power'
 -InternalBattery-0 (id=23658595)\t68%; charging; 4:30 remaining present: true
"""
BATT_BATTERY = """Now drawing from 'Battery Power'
 -InternalBattery-0 (id=23658595)\t91%; discharging; 9:12 remaining present: true
"""
PMSET = """System-wide power settings:
Currently in use:
 standby              1
 sleep                0 (sleep prevented by sharingd, powerd)
 hibernatemode        3
 displaysleep         10
"""
PMSET_LOW_POWER = PMSET + " lowpowermode         1\n"
PMSET_POWERMODE_LOW = PMSET + " powermode            1\n"
PMSET_POWERMODE_AUTO = PMSET + " powermode            0\n"
THERM = """Note: No thermal warning level has been recorded
Note: No performance warning level has been recorded
Note: No CPU power status has been recorded
"""


AC_96W = """ Wattage = 94W
 Current = 4690mA
 Voltage = 20000mV
 Adapter Name = 96W USB-C Power Adapter
"""
AC_30W = """ Wattage = 30W
 Adapter Name = 30W USB-C Power Adapter
"""
AC_NONE = "No adapter attached.\n"


def _state(batt: str = BATT_AC, pmset: str = PMSET, ac: str = AC_96W) -> MachineState:
    return read_machine_state(batt_text=batt, pmset_text=pmset, therm_text=THERM, ac_text=ac)


def test_machine_state_parses_pmset() -> None:
    state = _state()
    assert state.power_source == "AC Power"
    assert state.low_power_mode == "not reported"
    assert "No thermal warning level" in state.thermal
    assert state.available_bytes > 0
    assert _state(batt=BATT_BATTERY).power_source == "Battery Power"
    assert _state(pmset=PMSET_LOW_POWER).low_power_mode == "on"
    assert _state(pmset=PMSET.replace("standby", "lowpowermode 0\n standby")).low_power_mode == (
        "off"
    )
    assert _state(pmset=PMSET_POWERMODE_LOW).low_power_mode == "on"
    assert _state(pmset=PMSET_POWERMODE_AUTO).low_power_mode == "off"


def test_machine_state_has_versions_and_chip() -> None:
    state = _state()
    assert {"macos", "python", "torch", "transformers", "mlx"} <= set(state.versions)
    assert state.chip
    assert state.memory_bytes > 0


def test_check_machine_state_accepts_ac_power() -> None:
    check_machine_state(_state())


def test_check_machine_state_refuses_battery() -> None:
    with pytest.raises(MachineStateError, match="battery"):
        check_machine_state(_state(batt=BATT_BATTERY))


def test_check_machine_state_refuses_low_power_mode() -> None:
    with pytest.raises(MachineStateError, match="Low Power Mode"):
        check_machine_state(_state(pmset=PMSET_LOW_POWER))


def test_machine_state_parses_adapter_and_battery() -> None:
    # Finding F11 of specs/03-baseline/verify.html: a 30 W adapter cannot feed an M5 Max under load.
    state = _state()
    assert state.adapter_watts == 94
    assert state.battery_percent == 68
    assert _state(ac=AC_30W).adapter_watts == 30
    assert _state(ac=AC_NONE).adapter_watts is None


def test_check_machine_state_refuses_weak_adapter() -> None:
    with pytest.raises(MachineStateError, match="30 W"):
        check_machine_state(_state(ac=AC_30W))


def test_check_machine_state_accepts_no_adapter_report() -> None:
    # A desktop Mac has no adapter. Battery power is refused by its own check.
    check_machine_state(_state(ac=AC_NONE))
