from dataclasses import dataclass


@dataclass
class BatteryParameters:
    smm: int
    capacity_kwh: float
    power_kw: float
    cycles: float


@dataclass
class FeederResult:
    """Undervoltage statistics and battery parameters for one feeder"""

    trafo: str
    feeder: str
    n_undervoltages: int
    n_meters: int
    n_meters_with_uv: int
    n_uv_timestamps: int
    battery_smm: int | None = None
    battery_capacity_kwh: float | None = None
    battery_power_kw: float | None = None
    battery_cycles: float | None = None

    def add_battery(self, battery: BatteryParameters) -> None:
        self.battery_smm = battery.smm
        self.battery_capacity_kwh = battery.capacity_kwh
        self.battery_power_kw = battery.power_kw
        self.battery_cycles = battery.cycles
