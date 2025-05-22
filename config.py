# config.py
# Configuration for Industrial Workplace Shift Monitoring Dashboard
# Defines parameters for simulating and analyzing shift performance, safety, and team well-being.

def validate_config(config):
    """Validate configuration parameters."""
    assert config['FACILITY_SIZE'] > 0, "Facility size must be positive"
    assert 0 <= config['WELLBEING_THRESHOLD'] <= 1, "Well-being threshold must be between 0 and 1"
    assert 0 <= config['SAFETY_COMPLIANCE_THRESHOLD'] <= 1, "Safety threshold must be between 0 and 1"
    assert config['TEAM_SIZE'] > 0, "Team size must be positive"
    assert config['SHIFT_DURATION_INTERVALS'] > 0, "Shift duration must be positive"
    assert all(0 <= t <= config['SHIFT_DURATION_INTERVALS'] for t in config['DISRUPTION_INTERVALS']), "Disruption intervals out of range"
    assert 0 <= config['ADAPTATION_RATE'] <= 1, "Adaptation rate must be between 0 and 1"
    assert 0 <= config['SUPERVISOR_INFLUENCE'] <= 1, "Supervisor influence must be between 0 and 1"

CONFIG = {
    # Work Zones: Defines operational areas with coordinates and worker assignments
    'WORK_AREAS': {
        'Assembly Line': {'center': [20, 20], 'label': 'Assembly Line', 'workers': 20},
        'Packaging Zone': {'center': [60, 60], 'label': 'Packaging Zone', 'workers': 15},
        'Quality Control': {'center': [80, 80], 'label': 'Quality Control', 'workers': 15}
    },
    # Well-Being Threshold: Minimum acceptable team well-being score (0 to 1)
    'WELLBEING_THRESHOLD': 0.7,
    # Well-Being Trend Window: Intervals to detect declining well-being trends
    'WELLBEING_TREND_WINDOW': 3,
    # Disruption Recovery Window: Intervals to monitor recovery post-disruption
    'DISRUPTION_RECOVERY_WINDOW': 10,
    # Break Frequency (Intervals): Intervals between breaks (60 min = 30 intervals)
    'BREAK_FREQUENCY_INTERVALS': 30,  # 60 minutes / 2-min intervals
    # Workload Cap Intervals: Intervals to limit excessive workload
    'WORKLOAD_CAP_INTERVALS': 10,
    # Team Size: Total number of workers per shift
    'TEAM_SIZE': 50,
    # Shift Duration Intervals: Total shift duration (8 hours = 480 intervals)
    'SHIFT_DURATION_INTERVALS': 480,
    # Facility Size: Dimensions of the workplace (meters)
    'FACILITY_SIZE': 100,
    # Adaptation Rate: Rate of worker adaptation to process changes (0 to 1)
    'ADAPTATION_RATE': 0.05,
    # Supervisor Influence: Impact of supervision on compliance (0 to 1)
    'SUPERVISOR_INFLUENCE': 0.2,
    # Disruption Intervals: Intervals where disruptions occur
    'DISRUPTION_INTERVALS': [60, 180],
    # Anomaly Threshold: Z-score for detecting performance anomalies
    'ANOMALY_THRESHOLD': 2.0,
    # Safety Compliance Threshold: Minimum safety score (0 to 1)
    'SAFETY_COMPLIANCE_THRESHOLD': 0.7,
    # Density Grid Size: Grid size for worker density heatmaps
    'DENSITY_GRID_SIZE': 20,
    # Facility Type: Type of workplace for visualization
    'FACILITY_TYPE': 'manufacturing'
}

# Validate configuration
validate_config(CONFIG)
