# Factory simulation parameters
NUM_OPERATORS = 51  # Adjusted for equal zone distribution (51 / 3 = 17)
NUM_STEPS = 240  # 8-hour shift, 2-min intervals
FACTORY_SIZE = 100  # Factory floor size in meters
ADAPTATION_RATE = 0.02  # Rate of SOP compliance adaptation
SUPERVISOR_INFLUENCE = 0.1  # Supervisor influence on compliance
DISRUPTION_STEPS = [60, 180]  # Machine breakdown, shift change
ANOMALY_THRESHOLD = 1.5  # Z-score threshold for anomalies
WELLBEING_THRESHOLD = 0.7  # Threshold for low well-being alerts
