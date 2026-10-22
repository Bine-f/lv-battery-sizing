"""Physical limits and algorithm parameters."""

NOMINAL_VOLTAGE = 230.0  # V, phase-to-neutral
UNDERVOLTAGE_LIMIT = 207.0  # V, nominal voltage - 10 %
UNDERVOLTAGE_LIMIT_PU = UNDERVOLTAGE_LIMIT / NOMINAL_VOLTAGE

# Measurements outside these limits are considered faulty
MIN_VALID_VOLTAGE = 170.0  # V
MAX_PHASE_SPREAD = 30.0  # V, difference between highest and lowest phase
# Meters with many undervoltages are dropped if the average of the two highest phases
# is more than this above the lowest phase
MAX_PHASE_ASYMMETRY = 7.0  # V
MIN_UV_SAMPLES_FOR_ASYMMETRY_CHECK = 300

# A transformer / feeder needs a battery if it has undervoltages at this many timestamps
MIN_UV_TIMESTAMPS = 4
# Share of meters in the network model that need voltage data for calibration
MIN_VOLTAGE_DATA_SHARE = 0.6

MEASUREMENT_INTERVAL = "10min"
INTERVALS_PER_HOUR = 6

# Network model calibration
INITIAL_TRAFO_LV_KV = 0.425
CALIBRATION_QUANTILE = 0.005  # quantile of minimal feeder voltages used as calibration limit
N_CALIBRATION_DATES = 4
MIN_RES_FACTOR = 0.7
MAX_RES_FACTOR = 1.3

# Battery powers above these are considered unrealistic and set to 0
MAX_BATTERY_POWER_KW = 100.0
MAX_BATTERY_POWER_WITH_CHARGING_KW = 150.0
# Voltage deviation used for missing phases, so that they never limit charging
MISSING_PHASE_DEVIATION = -5000.0  # V
