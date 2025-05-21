```python
# Configuration for workplace shift monitoring dashboard

CONFIG = {
    'WORK_AREAS': {
        'Area 1': {'center': [20, 20], 'label': 'Area 1'},
        'Area 2': {'center': [60, 60], 'label': 'Area 2'},
        'Area 3': {'center': [80, 80], 'label': 'Area 3'}
    },
    'WELLBEING_THRESHOLD': 0.75,
    'WELLBEING_TREND_LENGTH': 3,
    'WELLBEING_DISRUPTION_WINDOW': 10,
    'BREAK_INTERVAL': 60,
    'WORKLOAD_CAP_STEPS': 10,
    'NUM_TEAM_MEMBERS': 51,
    'NUM_STEPS': 240,
    'WORKPLACE_SIZE': 100,
    'ADAPTATION_RATE': 0.05,
    'SUPERVISOR_INFLUENCE': 0.2,
    'DISRUPTION_STEPS': [60, 180],
    'ANOMALY_THRESHOLD': 2.0,
    'SAFETY_THRESHOLD': 0.7,
    'HEXBIN_GRIDSIZE': 20,
    'WORKPLACE_TYPE': 'generic'
}
```
