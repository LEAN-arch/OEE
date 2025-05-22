# main.py
import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output
from simulation import simulate_workplace_operations, plot_task_compliance_trend, plot_worker_collaboration_trend, \
    plot_operational_resilience, plot_operational_efficiency, plot_worker_distribution, plot_worker_wellbeing, \
    plot_psychological_safety
from config import CONFIG

# Run simulation
(team_positions_df, compliance_variability, collaboration_index, operational_resilience,
 efficiency_metrics_df, productivity_loss, team_wellbeing, safety, feedback_impact) = simulate_workplace_operations(
    config=CONFIG
)

# Save outputs
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

# Generate plots
compliance_fig = plot_task_compliance_trend(compliance_variability['data'], CONFIG['DISRUPTION_INTERVALS'], compliance_variability['forecast'])
collaboration_fig = plot_worker_collaboration_trend(collaboration_index['data'], collaboration_index['forecast'])
resilience_fig = plot_operational_resilience(operational_resilience)
efficiency_fig = plot_operational_efficiency(efficiency_metrics_df)
distribution_fig = plot_worker_distribution(team_positions_df, CONFIG['FACILITY_SIZE'], CONFIG)
wellbeing_fig = plot_worker_wellbeing(team_wellbeing['scores'])
safety_fig = plot_psychological_safety(safety)

# Initialize Dash app
app = Dash(__name__)

app.layout = html.Div([
    html.H1(f"{CONFIG['FACILITY_TYPE'].capitalize()} Workplace Shift Monitoring Dashboard"),
    html.H2("Operational Efficiency"),
    dcc.Graph(figure=efficiency_fig),
    html.H2("Team Distribution"),
    dcc.Graph(figure=distribution_fig),
    html.H2("Team Well-Being"),
    dcc.Graph(figure=wellbeing_fig),
    html.H2("Psychological Safety"),
    dcc.Graph(figure=safety_fig),
    html.H2("Task Compliance Variability"),
    dcc.Graph(figure=compliance_fig),
    html.H2("Worker Collaboration Index"),
    dcc.Graph(figure=collaboration_fig),
    html.H2("Operational Resilience"),
    dcc.Graph(figure=resilience_fig),
    html.H3("Well-Being Triggers"),
    html.Ul([html.Li(f"Threshold Alerts: {team_wellbeing['triggers']['threshold']}"),
             html.Li(f"Trend Alerts: {team_wellbeing['triggers']['trend']}"),
             html.Li([html.Span(f"{zone}: {triggers}") for zone, triggers in team_wellbeing['triggers']['work_area'].items()]),
             html.Li(f"Disruption Alerts: {team_wellbeing['triggers']['disruption']}")])
])

if __name__ == '__main__':
    app.run_server(debug=True)
