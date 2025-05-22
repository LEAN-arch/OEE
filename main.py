import streamlit as st
import pandas as pd
import plotly.express as px
from simulation import simulate_workplace_operations, plot_task_compliance_trend, plot_worker_collaboration_trend, \
    plot_operational_resilience, plot_operational_efficiency, plot_worker_distribution, plot_worker_wellbeing, \
    plot_psychological_safety
from config import CONFIG

# Streamlit app title
st.title(f"{CONFIG['FACILITY_TYPE'].capitalize()} Workplace Shift Monitoring Dashboard")

# Run simulation
try:
    (team_positions_df, compliance_variability, collaboration_index, operational_resilience,
     efficiency_metrics_df, productivity_loss, team_wellbeing, safety, feedback_impact) = simulate_workplace_operations(
        config=CONFIG
    )
except Exception as e:
    st.error(f"Simulation failed: {str(e)}")
    st.stop()

# Save outputs (optional, for debugging or persistence)
team_positions_df.to_csv('team_positions.csv', index=False)
efficiency_metrics_df.to_csv('efficiency_metrics.csv', index=False)
pd.DataFrame({
    'step': range(CONFIG['SHIFT_DURATION_INTERVALS']),
    'compliance_entropy': compliance_variability['data'],
    'collaboration_index': collaboration_index['data'],
    'resilience': operational_resilience,
    'wellbeing': team_wellbeing['scores'],
    'safety': safety,
    'productivity_loss': productivity_loss
}).to_csv('summary_metrics.csv', index=False)

# Generate and display plots
st.header("Operational Efficiency")
efficiency_fig = plot_operational_efficiency(efficiency_metrics_df)
st.plotly_chart(efficiency_fig)

st.header("Team Distribution")
distribution_fig = plot_worker_distribution(team_positions_df, CONFIG['FACILITY_SIZE'], CONFIG)
st.plotly_chart(distribution_fig)

st.header("Team Well-Being")
wellbeing_fig = plot_worker_wellbeing(team_wellbeing['scores'])
st.plotly_chart(wellbeing_fig)

st.header("Psychological Safety")
safety_fig = plot_psychological_safety(safety)
st.plotly_chart(safety_fig)

st.header("Task Compliance Variability")
compliance_fig = plot_task_compliance_trend(compliance_variability['data'], CONFIG['DISRUPTION_INTERVALS'], compliance_variability['forecast'])
st.plotly_chart(compliance_fig)

st.header("Worker Collaboration Index")
collaboration_fig = plot_worker_collaboration_trend(collaboration_index['data'], collaboration_index['forecast'])
st.plotly_chart(collaboration_fig)

st.header("Operational Resilience")
resilience_fig = plot_operational_resilience(operational_resilience)
st.plotly_chart(resilience_fig)

st.header("Well-Being Triggers")
st.write(f"Threshold Alerts: {team_wellbeing['triggers']['threshold']}")
st.write(f"Trend Alerts: {team_wellbeing['triggers']['trend']}")
st.write("Work Area Alerts:")
for zone, triggers in team_wellbeing['triggers']['work_area'].items():
    st.write(f"{zone}: {triggers}")
st.write(f"Disruption Alerts: {team_wellbeing['triggers']['disruption']}")
