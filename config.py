# Configuration for workplace shift monitoring dashboard

NUM_TEAM_MEMBERS = 51  # Total team members
NUM_STEPS = 240  # Shift duration in 2-min intervals (8 hours)
WORKPLACE_SIZE = 100  # Workplace size in meters (square)
ADAPTATION_RATE = 0.05  # Rate of SOP compliance adaptation
SUPERVISOR_INFLUENCE = 0.2  # Supervisor impact on compliance
DISRUPTION_STEPS = [60, 180]  # System disruptions (e.g., IT outage, process delay)
ANOMALY_THRESHOLD = 2.0  # Z-score threshold for performance alerts
WELLBEING_THRESHOLD = 0.75  # Well-being score threshold for triggers
WELLBEING_TREND_LENGTH = 3  # Steps for well-being trend detection
WELLBEING_DISRUPTION_WINDOW = 10  # Steps around disruptions for well-being checks
BREAK_INTERVAL = 60  # Proactive breaks every 2 hours (60 steps)
WORKLOAD_CAP_STEPS = 10  # Steps before workload cap triggers
SAFETY_THRESHOLD = 0.7  # Psychological safety threshold
HEXBIN_GRIDSIZE = 20  # Gridsize for activity density plot
WORKPLACE_TYPE = "generic"  # Options: office, warehouse, retail, generic
WORK_AREAS = {
    "Area 1": {"center": (20, 20), "label": "Area 1"},
    "Area 2": {"center": (60, 60), "label": "Area 2"},
    "Area 3": {"center": (80, 80), "label": "Area 3"}
}
