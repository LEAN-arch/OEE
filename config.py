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
    'WORK_AREAS': {
        'Assembly Line': {'center': [20, 20], 'label': 'Assembly Line', 'workers': 20},
        'Packaging Zone': {'center': [60, 60], 'label': 'Packaging Zone', 'workers': 15},
        'Quality Control': {'center': [80, 80], 'label': 'Quality Control', 'workers': 15}
    },
    'WELLBEING_THRESHOLD': 0.7,
    'WELLBEING_TREND_WINDOW': 3,
    'DISRUPTION_RECOVERY_WINDOW': 10,
    'BREAK_FREQUENCY_INTERVALS': 30,
    'WORKLOAD_CAP_INTERVALS': 10,
    'TEAM_SIZE': 50,
    'SHIFT_DURATION_INTERVALS': 480,
    'FACILITY_SIZE': 100,
    'ADAPTATION_RATE': 0.05,
    'SUPERVISOR_INFLUENCE': 0.2,
    'DISRUPTION_INTERVALS': [60, 180],
    'ANOMALY_THRESHOLD': 2.0,
    'SAFETY_COMPLIANCE_THRESHOLD': 0.7,
    'DENSITY_GRID_SIZE': 20,
    'FACILITY_TYPE': 'manufacturing'
}

validate_config(CONFIG)
