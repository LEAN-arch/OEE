# main.py
# Streamlit dashboard for Workplace Shift Monitoring Dashboard.
# Robust fixes for blank Worker Insights display, professional visuals, accessibility, and error handling for plot_task_compliance_score.

import logging
import streamlit as st
import pandas as pd
import numpy as np
from config import DEFAULT_CONFIG, validate_config
from visualizations import (
    plot_key_metrics_summary,
    plot_task_compliance_score,
    plot_collaboration_proximity_index,
    plot_operational_recovery,
    plot_operational_efficiency,
    plot_worker_distribution,
    plot_worker_density_heatmap,
    plot_worker_wellbeing,
    plot_psychological_safety,
    plot_downtime_trend
)
from simulation import simulate_workplace_operations
from utils import save_simulation_data, load_simulation_data, generate_pdf_report
from assets import LEAN_LOGO_BASE64

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s - [User Action: %(user_action)s]',
    filename='dashboard.log'
)

# Streamlit page config
st.set_page_config(
    page_title="Workplace Shift Monitoring Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional, accessible design
st.markdown("""
    <style>
        .main {
            background-color: #1E2A44;
            color: #F5F7FA;
            font-family: 'Inter', sans-serif;
            padding: 24px;
        }
        h1 {
            color: #F5F7FA;
            font-size: 2.25rem;
            font-weight: 600;
            text-align: center;
            margin-bottom: 1.5rem;
        }
        h2 {
            color: #F5F7FA;
            font-size: 1.75rem;
            font-weight: 500;
            margin: 1.25rem 0 0.75rem;
        }
        h3 {
            color: #D1D5DB;
            font-size: 1.25rem;
            font-weight: 500;
            margin-bottom: 0.5rem;
        }
        .stButton>button {
            background-color: #4F46E5;
            color: #F5F7FA;
            border-radius: 6px;
            border: none;
            padding: 8px 16px;
            font-weight: 500;
            transition: all 0.2s ease;
        }
        .stButton>button:hover, .stButton>button:focus {
            background-color: #EC4899;
            transform: translateY(-1px);
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
            outline: none;
        }
        .stSelectbox, .stSlider, .stMultiSelect {
            background-color: #2D3B55;
            color: #F5F7FA;
            border-radius: 6px;
            padding: 6px;
            margin-bottom: 12px;
        }
        .tooltip {
            position: relative;
            display: inline-flex;
            align-items: center;
            margin-left: 6px;
        }
        .tooltip .tooltiptext {
            visibility: hidden;
            width: 280px;
            background-color: #2D3B55;
            color: #F5F7FA;
            text-align: left;
            border-radius: 6px;
            padding: 12px;
            position: absolute;
            z-index: 10;
            top: 100%;
            left: 50%;
            margin-left: -140px;
            opacity: 0;
            transition: opacity 0.3s;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }
        .tooltip:hover .tooltiptext, .tooltip:focus .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
        [data-testid="stSidebar"] {
            background-color: #2D3B55;
            color: #F5F7FA;
            padding: 16px;
            border-right: 1px solid #4B5EAA;
        }
        [data-testid="stSidebar"] .stButton>button {
            background-color: #10B981;
        }
        [data-testid="stSidebar"] .stButton>button:hover, [data-testid="stSidebar"] .stButton>button:focus {
            background-color: #EC4899;
        }
        .stMetric {
            background-color: #2D3B55;
            border-radius: 6px;
            padding: 16px;
            margin: 12px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .stExpander {
            background-color: #2D3B55;
            border-radius: 6px;
            margin: 12px 0;
            border: 1px solid #4B5EAA;
        }
        .recommendation {
            color: #FBBF24;
            font-size: 0.9rem;
            margin-top: 8px;
            font-style: italic;
        }
        .stTabs [data-baseweb="tab-list"] {
            background-color: #2D3B55;
            border-radius: 6px;
            padding: 8px;
            display: flex;
            justify-content: center;
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            color: #D1D5DB;
            padding: 10px 20px;
            border-radius: 6px;
            font-weight: 500;
            transition: all 0.2s ease;
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background-color: #4F46E5;
            color: #F5F7FA;
        }
        .stTabs [data-baseweb="tab"]:hover {
            background-color: #6B7280;
            color: #F5F7FA;
        }
        .stPlotlyChart {
            margin: 12px 0;
            background-color: #2D3B55;
            border-radius: 6px;
            padding: 12px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .summary-card, .action-card {
            background-color: #2D3B55;
            border-radius: 6px;
            padding: 16px;
            margin: 12px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .summary-card h4, .action-card h4 {
            color: #F5F7FA;
            font-size: 1.1rem;
            margin: 0 0 8px;
        }
        .summary-card p, .action-card p {
            color: #D1D5DB;
            font-size: 0.9rem;
            margin: 0;
        }
        .summary-card p.metric {
            color: #FBBF24;
            font-size: 1.5rem;
            font-weight: 600;
        }
        @media (max-width: 768px) {
            .stColumn {
                width: 100% !important;
                margin-bottom: 1rem;
            }
            .stPlotlyChart {
                height: 400px !important;
            }
            h1 { font-size: 1.75rem; }
            h2 { font-size: 1.5rem; }
            h3 { font-size: 1.1rem; }
            .stTabs [data-baseweb="tab"] {
                padding: 8px 12px;
                font-size: 0.9rem;
            }
            .summary-card, .action-card {
                text-align: center;
            }
        }
        .spinner {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 120px;
        }
        .spinner::after {
            content: '';
            width: 48px;
            height: 48px;
            border: 5px solid #D1D5DB;
            border-top: 5px solid #4F46E5;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .onboarding-modal {
            background-color: #2D3B55;
            border-radius: 8px;
            padding: 24px;
            max-width: 600px;
            margin: 24px auto;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }
        .onboarding-modal h3 {
            color: #F5F7FA;
            margin-bottom: 16px;
        }
        .onboarding-modal p {
            color: #D1D5DB;
            line-height: 1.5;
            margin-bottom: 16px;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar for settings
def render_settings_sidebar():
    with st.sidebar:
        st.markdown(
            f'<img src="{LEAN_LOGO_BASE64}" width="120" alt="Lean 2.0 Institute Logo" aria-label="Lean 2.0 Institute Logo" style="display: block; margin: 0 auto 16px;">',
            unsafe_allow_html=True
        )
        st.header("⚙️ Settings", divider="grey")

        with st.expander("🧪 Simulation", expanded=True):
            team_size = st.slider(
                "Team Size",
                min_value=10, max_value=100, value=DEFAULT_CONFIG['TEAM_SIZE'],
                key="team_size"
            )
            shift_duration = st.slider(
                "Shift Duration (minutes)",
                min_value=200, max_value=2000, value=DEFAULT_CONFIG['SHIFT_DURATION_MINUTES'], step=2,
                key="shift_duration"
            )
            disruption_intervals = st.multiselect(
                "Disruption Times (minutes)",
                options=[i * 2 for i in range(shift_duration // 2)],
                default=[i * 2 for i in DEFAULT_CONFIG['DISRUPTION_INTERVALS']],
                key="disruption_intervals"
            )
            team_initiative = st.selectbox(
                "Team Initiative",
                options=["More frequent breaks", "Team recognition"],
                index=0,
                key="team_initiative"
            )
            run_simulation = st.button(
                "Run Simulation", 
                key="run_simulation", 
                type="primary"
            )

        with st.expander("🎨 Visualizations"):
            high_contrast = st.checkbox(
                "High Contrast Mode", 
                key="high_contrast"
            )
            use_3d_distribution = st.checkbox(
                "3D Team Distribution", 
                key="use_3d_distribution"
            )
            debug_mode = st.checkbox(
                "Debug Mode", 
                key="debug_mode"
            )

        with st.expander("💾 Data"):
            load_data = st.button(
                "Load Saved Data", 
                key="load_data"
            )
            if st.button("Download PDF Report", key="download_report") and 'simulation_results' in st.session_state:
                try:
                    summary_df = pd.DataFrame({
                        'step': range(DEFAULT_CONFIG['SHIFT_DURATION_INTERVALS']),
                        'time_minutes': [i * 2 for i in range(DEFAULT_CONFIG['SHIFT_DURATION_INTERVALS'])],
                        'task_compliance': st.session_state.simulation_results[1]['data'],
                        'collaboration_proximity': st.session_state.simulation_results[2]['data'],
                        'operational_recovery': st.session_state.simulation_results[3],
                        'worker_wellbeing': st.session_state.simulation_results[6]['scores'],
                        'psychological_safety': st.session_state.simulation_results[7],
                        'productivity_loss': st.session_state.simulation_results[5],
                        'downtime_minutes': st.session_state.simulation_results[9],
                        'task_completion_rate': st.session_state.simulation_results[10]
                    })
                    generate_pdf_report(summary_df)
                    st.success("PDF report generated as 'workplace_report.tex'.")
                except Exception as e:
                    logger.error(f"Failed to generate report: {str(e)}", extra={'user_action': 'Download PDF Report'})
                    st.error(f"Failed to generate report: {str(e)}")

        if debug_mode:
            with st.expander("🛠️ Debug Info"):
                st.write("**Entry/Exit Points:**", DEFAULT_CONFIG.get('ENTRY_EXIT_POINTS', "Not defined"))
                st.write("**Work Areas:**", DEFAULT_CONFIG.get('WORK_AREAS', "Not defined"))
                if 'simulation_results' in st.session_state and st.session_state.simulation_results:
                    st.write("**Simulation Data:**", {
                        "team_positions_df_columns": list(st.session_state.simulation_results[0].columns),
                        "team_positions_df_rows": len(st.session_state.simulation_results[0]),
                        "worker_wellbeing_scores": len(st.session_state.simulation_results[6].get('scores', [])),
                        "psychological_safety_length": len(st.session_state.simulation_results[7])
                    })

        st.header("📋 Navigation", divider="grey")
        tab_names = ["Overview", "Operational Metrics", "Worker Insights", "Downtime", "Glossary"]
        for i, tab in enumerate(tab_names):
            if st.button(tab, key=f"nav_{tab.lower().replace(' ', '_')}"):
                st.session_state.active_tab = i

        if st.button("ℹ️ Help", key="help_button"):
            st.session_state.show_help = True

    return team_size, shift_duration, disruption_intervals, team_initiative, run_simulation, load_data, high_contrast, use_3d_distribution, debug_mode

# Simulation logic with caching
@st.cache_data
def run_simulation_logic(team_size, shift_duration, disruption_intervals, team_initiative):
    config = DEFAULT_CONFIG.copy()
    config['TEAM_SIZE'] = team_size
    config['SHIFT_DURATION_MINUTES'] = shift_duration
    config['SHIFT_DURATION_INTERVALS'] = shift_duration // 2
    config['DISRUPTION_INTERVALS'] = [t // 2 for t in disruption_intervals]
    
    total_current_workers = sum(zone['workers'] for zone in config['WORK_AREAS'].values())
    if total_current_workers != team_size:
        ratio = team_size / total_current_workers
        for zone in config['WORK_AREAS'].values():
            zone['workers'] = int(zone['workers'] * ratio)
        current_sum = sum(zone['workers'] for zone in config['WORK_AREAS'].values())
        if current_sum != team_size:
            diff = team_size - current_sum
            config['WORK_AREAS']['Assembly Line']['workers'] += diff
    
    validate_config(config)
    logger.info(
        f"Running simulation with team_size={team_size}, shift_duration={shift_duration} min",
        extra={'user_action': 'Run Simulation'}
    )
    simulation_results = simulate_workplace_operations(
        num_team_members=team_size,
        num_steps=shift_duration // 2,
        disruption_intervals=[t // 2 for t in disruption_intervals],
        team_initiative=team_initiative,
        config=config
    )
    save_simulation_data(*simulation_results)
    return simulation_results

# Main content
def main():
    st.title("Workplace Shift Monitoring Dashboard")

    # Initialize session state
    if 'simulation_results' not in st.session_state:
        st.session_state.simulation_results = None
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = 0
    if 'show_help' not in st.session_state:
        st.session_state.show_help = False

    # Sidebar settings
    team_size, shift_duration, disruption_intervals, team_initiative, run_simulation, load_data, high_contrast, use_3d_distribution, debug_mode = render_settings_sidebar()

    # Handle simulation and data loading
    if run_simulation:
        with st.spinner("Running simulation..."):
            try:
                st.session_state.simulation_results = run_simulation_logic(
                    team_size, shift_duration, disruption_intervals, team_initiative
                )
                st.success("Simulation completed!", icon="✅")
            except Exception as e:
                logger.error(f"Simulation failed: {str(e)}", extra={'user_action': 'Run Simulation'})
                st.error(f"Simulation failed: {str(e)}")

    if load_data:
        with st.spinner("Loading saved data..."):
            try:
                st.session_state.simulation_results = load_simulation_data()
                st.success("Data loaded!", icon="✅")
            except Exception as e:
                logger.error(f"Failed to load data: {str(e)}", extra={'user_action': 'Load Data'})
                st.error(f"Failed to load data: {str(e)}")

    # Help Modal
    if st.session_state.show_help:
        with st.container():
            st.markdown("""
                <div class="onboarding-modal" role="dialog" aria-label="Help Section">
                    <h3>Help & Documentation</h3>
                    <p>Navigate the dashboard:</p>
                    <ul style="color: #D1D5DB; line-height: 1.6;">
                        <li><strong>Overview</strong>: High-level metrics.</li>
                        <li><strong>Operational Metrics</strong>: Performance trends.</li>
                        <li><strong>Worker Insights</strong>: Distribution and well-being.</li>
                        <li><strong>Downtime</strong>: Downtime trends.</li>
                        <li><strong>Glossary</strong>: Metric definitions.</li>
                    </ul>
                    <p>Contact support@xai.com for assistance.</p>
                </div>
            """, unsafe_allow_html=True)
            if st.button("Close Help", key="close_help"):
                st.session_state.show_help = False

    # Tabs
    tab_names = ["Overview", "Operational Metrics", "Worker Insights", "Downtime", "Glossary"]
    tabs = st.tabs(tab_names)

    # Overview Tab
    with tabs[0]:
        st.header("Overview", divider="grey")
        if st.session_state.simulation_results:
            (team_positions_df, task_compliance, collaboration_proximity, operational_recovery,
             efficiency_metrics_df, productivity_loss, worker_wellbeing, psychological_safety,
             feedback_impact, downtime_minutes, task_completion_rate) = st.session_state.simulation_results
            compliance_mean = np.mean(task_compliance['data'])
            proximity_mean = np.mean(collaboration_proximity['data'])
            wellbeing_mean = np.mean(worker_wellbeing['scores']) if worker_wellbeing['scores'] else 0
            total_downtime = np.sum(downtime_minutes)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f'<div class="summary-card"><h4>Task Compliance</h4><p class="metric">{compliance_mean:.1f}%</p></div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="summary-card"><h4>Collaboration</h4><p class="metric">{proximity_mean:.1f}%</p></div>', unsafe_allow_html=True)
            with col3:
                st.markdown(f'<div class="summary-card"><h4>Well-Being</h4><p class="metric">{wellbeing_mean:.1f}%</p></div>', unsafe_allow_html=True)
            with col4:
                st.markdown(f'<div class="summary-card"><h4>Downtime</h4><p class="metric">{total_downtime:.1f} min</p></div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            summary_figs = plot_key_metrics_summary(compliance_mean, proximity_mean, wellbeing_mean, total_downtime)
            with col1:
                st.plotly_chart(summary_figs[0], use_container_width=True)
                st.plotly_chart(summary_figs[1], use_container_width=True)
            with col2:
                st.plotly_chart(summary_figs[2], use_container_width=True)
                st.plotly_chart(summary_figs[3], use_container_width=True)
        else:
            st.info("Run a simulation or load data to view metrics.", icon="ℹ️")

    # Operational Metrics Tab
    with tabs[1]:
        st.header("Operational Metrics", divider="grey")
        if st.session_state.simulation_results:
            time_range = st.slider(
                "Time Range (minutes)",
                min_value=0,
                max_value=DEFAULT_CONFIG['SHIFT_DURATION_MINUTES'] - 2,
                value=(0, DEFAULT_CONFIG['SHIFT_DURATION_MINUTES'] - 2),
                step=2,
                key="time_range_op"
            )
            time_indices = (time_range[0] // 2, time_range[1] // 2 + 1)
            try:
                filtered_compliance = task_compliance['data'][time_indices[0]:time_indices[1]]
                filtered_z_scores = task_compliance['z_scores'][time_indices[0]:time_indices[1]]
                filtered_forecast = task_compliance['forecast'][time_indices[0]:time_indices[1]] if task_compliance['forecast'] is not None else None
                filtered_disruptions = [t for t in DEFAULT_CONFIG['DISRUPTION_INTERVALS'] if time_indices[0] <= t < time_indices[1]]
                
                if not filtered_compliance or not filtered_z_scores:
                    st.error("No data available for the selected time range.")
                else:
                    compliance_fig = plot_task_compliance_score(filtered_compliance, filtered_disruptions, filtered_forecast, filtered_z_scores)
                    st.plotly_chart(compliance_fig, use_container_width=True)
            except Exception as e:
                logger.error(f"Failed to render task compliance: {str(e)}", extra={'user_action': 'Render Operational Metrics'})
                st.error(f"Error rendering task compliance: {str(e)}")
            
            try:
                filtered_collab = collaboration_proximity['data'][time_indices[0]:time_indices[1]]
                filtered_forecast = collaboration_proximity['forecast'][time_indices[0]:time_indices[1]] if collaboration_proximity['forecast'] is not None else None
                if not filtered_collab:
                    st.error("No collaboration data available.")
                else:
                    collaboration_fig = plot_collaboration_proximity_index(filtered_collab, filtered_disruptions, filtered_forecast)
                    st.plotly_chart(collaboration_fig, use_container_width=True)
            except Exception as e:
                logger.error(f"Failed to render collaboration: {str(e)}", extra={'user_action': 'Render Operational Metrics'})
                st.error(f"Error rendering collaboration: {str(e)}")
        else:
            st.info("Run a simulation or load data to view metrics.", icon="ℹ️")

    # Worker Insights Tab (Robust Fix)
    with tabs[2]:
        st.header("Worker Insights", divider="grey")
        if not st.session_state.get('simulation_results'):
            st.info("Run a simulation or load data to view insights.", icon="ℹ️")
            logger.info("No simulation results available", extra={'user_action': 'Render Worker Insights'})
        else:
            try:
                (team_positions_df, task_compliance, collaboration_proximity, operational_recovery,
                 efficiency_metrics_df, productivity_loss, worker_wellbeing, psychological_safety,
                 feedback_impact, downtime_minutes, task_completion_rate) = st.session_state.simulation_results

                # Validate data
                required_columns = ['x', 'y', 'step', 'zone', 'role', 'worker', 'workload']
                if not all(col in team_positions_df.columns for col in required_columns):
                    st.error(f"Invalid team_positions_df: missing columns {set(required_columns) - set(team_positions_df.columns)}")
                    logger.error(
                        f"Missing columns in team_positions_df: {set(required_columns) - set(team_positions_df.columns)}",
                        extra={'user_action': 'Render Worker Insights'}
                    )
                    return
                if not worker_wellbeing.get('scores') or not psychological_safety:
                    st.error("Invalid well-being or psychological safety data.")
                    logger.error(
                        f"Empty data: worker_wellbeing.scores={len(worker_wellbeing.get('scores', []))}, psychological_safety={len(psychological_safety)}",
                        extra={'user_action': 'Render Worker Insights'}
                    )
                    return

                # Worker Distribution
                st.subheader("Team Distribution")
                time_range = st.slider(
                    "Time Range (minutes)",
                    min_value=0,
                    max_value=DEFAULT_CONFIG['SHIFT_DURATION_MINUTES'] - 2,
                    value=(0, DEFAULT_CONFIG['SHIFT_DURATION_MINUTES'] - 2),
                    step=2,
                    key="time_range_dist"
                )
                zone_filter = st.selectbox(
                    "Zone",
                    options=["All"] + list(DEFAULT_CONFIG['WORK_AREAS'].keys()),
                    key="zone_filter_dist"
                )
                role_filter = st.multiselect(
                    "Worker Roles",
                    options=["All", "Operator", "Supervisor", "Technician"],
                    default=["All"],
                    key="role_filter_dist"
                )
                time_indices = (time_range[0] // 2, time_range[1] // 2 + 1)
                filtered_df = team_positions_df.copy()
                if zone_filter != "All":
                    filtered_df = filtered_df[filtered_df['zone'] == zone_filter]
                if "All" not in role_filter:
                    filtered_df = filtered_df[filtered_df['role'].isin(role_filter)]
                filtered_df = filtered_df[(filtered_df['step'] >= time_indices[0]) & (filtered_df['step'] < time_indices[1])]

                if filtered_df.empty:
                    st.warning("No workers match the selected filters. Showing placeholder plot.")
                    logger.warning(
                        f"Empty filtered_df: zone={zone_filter}, roles={role_filter}, steps={time_indices}",
                        extra={'user_action': 'Render Worker Insights'}
                    )
                    placeholder_fig = go.Figure()
                    placeholder_fig.add_scatter(x=[0], y=[0], mode='markers', name="Placeholder")
                    placeholder_fig.update_layout(title="No Data Available", xaxis_title="X", yaxis_title="Y")
                    st.plotly_chart(placeholder_fig, use_container_width=True)
                else:
                    density_stats = filtered_df.groupby('zone').size().to_dict()
                    overcrowded_zones = [zone for zone, count in density_stats.items() if count > DEFAULT_CONFIG['TEAM_SIZE'] * 0.3]
                    if overcrowded_zones:
                        st.markdown(
                            f'<div class="action-card"><h4>Action: Redistribute Workers</h4><p>Overcrowding in {", ".join(overcrowded_zones)}. Reassign workers to balance workload.</p></div>',
                            unsafe_allow_html=True
                        )

                    col_dist1, col_dist2 = st.columns(2)
                    with col_dist1:
                        st.markdown("#### Worker Positions")
                        selected_step = st.slider(
                            "Time Step",
                            min_value=int(time_indices[0]),
                            max_value=int(time_indices[1] - 1),
                            value=int(time_indices[0]),
                            key="team_distribution_step"
                        )
                        try:
                            distribution_fig = plot_worker_distribution(
                                filtered_df, DEFAULT_CONFIG['FACILITY_SIZE'], DEFAULT_CONFIG,
                                use_3d=use_3d_distribution, selected_step=selected_step,
                                show_entry_exit=True, show_production_lines=True,
                                high_contrast=high_contrast
                            )
                            st.plotly_chart(distribution_fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Failed to render worker distribution: {str(e)}")
                            logger.error(f"Plot worker distribution failed: {str(e)}", extra={'user_action': 'Render Worker Insights'})
                    with col_dist2:
                        st.markdown("#### Density Heatmap")
                        try:
                            heatmap_fig = plot_worker_density_heatmap(
                                filtered_df, DEFAULT_CONFIG['FACILITY_SIZE'], DEFAULT_CONFIG,
                                show_entry_exit=True, show_production_lines=True,
                                intensity=1.0, high_contrast=high_contrast
                            )
                            st.plotly_chart(heatmap_fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Failed to render density heatmap: {str(e)}")
                            logger.error(f"Plot density heatmap failed: {str(e)}", extra={'user_action': 'Render Worker Insights'})

                # Worker Well-Being & Safety
                st.subheader("Well-Being and Safety")
                time_range = st.slider(
                    "Time Range (minutes)",
                    min_value=0,
                    max_value=DEFAULT_CONFIG['SHIFT_DURATION_MINUTES'] - 2,
                    value=(0, DEFAULT_CONFIG['SHIFT_DURATION_MINUTES'] - 2),
                    step=2,
                    key="time_range_well"
                )
                time_indices = (time_range[0] // 2, time_range[1] // 2 + 1)
                filtered_scores = worker_wellbeing['scores'][time_indices[0]:time_indices[1]]
                filtered_safety = psychological_safety[time_indices[0]:time_indices[1]]

                if not filtered_scores or not filtered_safety:
                    st.warning("No well-being or safety data available. Showing placeholder plot.")
                    logger.warning(
                        f"Empty data: filtered_scores={len(filtered_scores)}, filtered_safety={len(filtered_safety)}",
                        extra={'user_action': 'Render Worker Insights'}
                    )
                    placeholder_fig = go.Figure()
                    placeholder_fig.add_scatter(x=[0], y=[0], mode='markers', name="Placeholder")
                    placeholder_fig.update_layout(title="No Data Available", xaxis_title="Time", yaxis_title="Score")
                    st.plotly_chart(placeholder_fig, use_container_width=True)
                else:
                    wellbeing_mean = np.mean(filtered_scores) if filtered_scores else 0
                    if wellbeing_mean < DEFAULT_CONFIG['WELLBEING_THRESHOLD'] * 100:
                        st.markdown(
                            f'<div class="action-card"><h4>Action: Schedule Breaks</h4><p>Low well-being (average: {wellbeing_mean:.1f}%). Schedule 10-minute breaks every 60 minutes.</p></div>',
                            unsafe_allow_html=True
                        )

                    col_well1, col_well2 = st.columns(2)
                    with col_well1:
                        st.markdown("#### Well-Being Index")
                        filtered_triggers = {
                            'threshold': [t for t in worker_wellbeing['triggers']['threshold'] if time_indices[0] <= t < time_indices[1]],
                            'trend': [t for t in worker_wellbeing['triggers']['trend'] if time_indices[0] <= t < time_indices[1]],
                            'work_area': {k: [t for t in v if time_indices[0] <= t < time_indices[1]] for k, v in worker_wellbeing['triggers']['work_area'].items()},
                            'disruption': [t for t in worker_wellbeing['triggers']['disruption'] if time_indices[0] <= t < time_indices[1]]
                        }
                        try:
                            wellbeing_fig = plot_worker_wellbeing(
                                filtered_scores, filtered_triggers, high_contrast=high_contrast
                            )
                            st.plotly_chart(wellbeing_fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Failed to render well-being chart: {str(e)}")
                            logger.error(f"Plot well-being failed: {str(e)}", extra={'user_action': 'Render Worker Insights'})
                    with col_well2:
                        st.markdown("#### Psychological Safety")
                        try:
                            safety_fig = plot_psychological_safety(
                                filtered_safety, high_contrast=high_contrast
                            )
                            st.plotly_chart(safety_fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Failed to render psychological safety chart: {str(e)}")
                            logger.error(f"Plot psychological safety failed: {str(e)}", extra={'user_action': 'Render Worker Insights'})
            except Exception as e:
                st.error(f"Error processing worker insights: {str(e)}")
                logger.error(f"Worker Insights processing failed: {str(e)}", extra={'user_action': 'Render Worker Insights'})

    # Downtime Tab
    with tabs[3]:
        st.header("Downtime Analysis", divider="grey")
        if st.session_state.simulation_results:
            time_range = st.slider(
                "Time Range (minutes)",
                min_value=0,
                max_value=DEFAULT_CONFIG['SHIFT_DURATION_MINUTES'] - 2,
                value=(0, DEFAULT_CONFIG['SHIFT_DURATION_MINUTES'] - 2),
                step=2,
                key="time_range_down"
            )
            time_indices = (time_range[0] // 2, time_range[1] // 2 + 1)
            filtered_downtime = downtime_minutes[time_indices[0]:time_indices[1]]
            downtime_fig = plot_downtime_trend(filtered_downtime, DEFAULT_CONFIG['DOWNTIME_THRESHOLD'])
            st.plotly_chart(downtime_fig, use_container_width=True)
        else:
            st.info("Run a simulation or load data to view analysis.", icon="ℹ️")

    # Glossary Tab
    with tabs[4]:
        st.header("Glossary", divider="grey")
        st.markdown("""
            ### Metric Definitions
            - **Task Compliance Score**: Percentage of tasks completed correctly (0–100%).
            - **Collaboration Proximity Index**: Percentage of workers near colleagues (0–100%).
            - **Operational Recovery Score**: Resilience to disruptions (0–100%).
            - **Worker Well-Being Index**: Fatigue, stress, and satisfaction score (0–100%).
            - **Psychological Safety Score**: Comfort in reporting issues (0–100%).
            - **Downtime**: Total minutes of unplanned stops.
        """)

if __name__ == "__main__":
    main()
