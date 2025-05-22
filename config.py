# Configuration for Industrial Workplace Shift Monitoring Dashboard
# Defines parameters for simulating and analyzing shift performance, safety, and team well-being.
# Use these settings to customize workplace zones, thresholds, and operational metrics.

CONFIG = {
    # Work Zones: Defines key operational areas in the facility with coordinates and labels.
    # Use: Assign tasks, monitor density, optimize layouts (e.g., reduce congestion in Assembly Line).
    # Format: {zone_name: {'center': [x, y], 'label': display_name}}
    'WORK_AREAS': {
        'Assembly Line': {'center': [20, 20], 'label': 'Assembly Line'},
        'Packaging Zone': {'center': [60, 60], 'label': 'Packaging Zone'},
        'Quality Control': {'center': [80, 80], 'label': 'Quality Control'}
    },

    # Well-Being Threshold: Minimum acceptable team well-being score (0 to 1).
    # Use: Trigger alerts for low morale/health (e.g., <0.7 prompts breaks or wellness programs).
    'WELLBEING_THRESHOLD': 0.7,

    # Well-Being Trend Window: Number of intervals to detect declining well-being trends.
    # Use: Identify sustained drops (e.g., 3 intervals of decline may require workload reduction).
    'WELLBEING_TREND_WINDOW': 3,

    # Disruption Recovery Window: Intervals post-disruption to monitor recovery (in 2-min intervals).
    # Use: Schedule interventions (e.g., extra breaks) within 20 minutes after incidents.
    'DISRUPTION_RECOVERY_WINDOW': 10,

    # Break Frequency (Minutes): Interval between scheduled breaks (in minutes).
    # Use: Adjust to improve well-being (e.g., every 60 min for high-intensity tasks).
    'BREAK_FREQUENCY_MINUTES': 60,

    # Workload Cap Intervals: Intervals to limit excessive workload (in 2-min intervals).
    # Use: Prevent burnout by capping tasks (e.g., reassess after 10 intervals).
    'WORKLOAD_CAP_INTERVALS': 10,

    # Team Size: Number of workers per shift.
    # Use: Scale simulation for realistic staffing (e.g., 50 workers for a medium facility).
    'TEAM_SIZE': 50,

    # Shift Duration Intervals: Total shift duration (in 2-min intervals, e.g., 480 = 8 hours).
    # Use: Align with shift schedules (e.g., 8-hour shifts for standard operations).
    'SHIFT_DURATION_INTERVALS': 480,

    # Facility Size: Dimensions of the workplace (in meters, square area).
    # Use: Define spatial constraints for worker distribution (e.g., 100x100 m facility).
    'FACILITY_SIZE': 100,

    # Adaptation Rate: Rate at which workers adapt to process changes (0 to 1).
    # Use: Higher values (e.g., 0.05) simulate faster training or SOP adoption.
    'ADAPTATION_RATE': 0.05,

    # Supervisor Influence: Impact of supervision on compliance (0 to 1).
    # Use: Increase (e.g., 0.2) for stricter oversight to improve task adherence.
    'SUPERVISOR_INFLUENCE': 0.2,

    # Disruption Intervals: Shift intervals where disruptions occur (e.g., equipment failure).
    # Use: Simulate incidents at specific times (e.g., 120 min and 360 min into shift).
    'DISRUPTION_INTERVALS': [60, 180],

    # Anomaly Threshold: Z-score threshold for detecting performance anomalies.
    # Use: Flag significant deviations (e.g., >2.0 for compliance or collaboration issues).
    'ANOMALY_THRESHOLD': 2.0,

    # Safety Compliance Threshold: Minimum safety score for compliance (0 to 1).
    # Use: Trigger alerts for low safety (e.g., <0.7 prompts inspections or training).
    'SAFETY_COMPLIANCE_THRESHOLD': 0.7,

    # Density Grid Size: Grid size for worker density heatmaps (hexbin visualization).
    # Use: Adjust granularity (e.g., 20 for balanced detail in 100x100 m facility).
    'DENSITY_GRID_SIZE': 20,

    # Facility Type: Type of workplace for context-specific visualization.
    # Use: Set to 'manufacturing' for industrial settings (affects plot titles).
    'FACILITY_TYPE': 'manufacturing'
}
