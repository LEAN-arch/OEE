# Factory simulation parameters
NUM_OPERATORS = 51  # Equal zone distribution (51 / 3 = 17)
NUM_STEPS = 240  # 8-hour shift, 2-min intervals
FACTORY_SIZE = 100  # Factory floor size in meters
ADAPTATION_RATE = 0.02  # SOP compliance adaptation rate
SUPERVISOR_INFLUENCE = 0.1  # Supervisor influence on compliance
DISRUPTION_STEPS = [60, 180]  # Machine breakdown, shift change
ANOMALY_THRESHOLD = 1.5  # Z-score threshold for anomalies
WELLBEING_THRESHOLD = 0.75  # Threshold for low well-being
WELLBEING_TREND_LENGTH = 3  # Steps for well-being decline
WELLBEING_DISRUPTION_WINDOW = 10  # Steps around disruptions
BREAK_INTERVAL = 60  # Proactive breaks every 2 hours
WORKLOAD_CAP_STEPS = 10  # Steps for workload cap detection
SAFETY_THRESHOLD = 0.7  # Threshold for psychological safety
HEXBIN_GRIDSIZE = 20  # Grid size for density hexbin plot
