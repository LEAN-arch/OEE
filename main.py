# main.py
# Streamlit dashboard for the Workplace Shift Monitoring Dashboard.
# Enhanced for professional visuals, seamless UX, accessibility, fixed tab rendering, and debug mode.

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
        /* Base Styles */
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
        /* Summary Cards */
        .summary-card {
            background-color: #2D3B55;
            border-radius: 6px;
            padding: 16px;
            margin: 12px 0;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .summary-card h4 {
            color: #F5F7FA;
            font-size: 1.1rem;
            margin: 0;
        }
        .summary-card p {
            color: #FBBF24;
            font-size: 1.5rem;
            font-weight: 600;
            margin: 0;
        }
        /* Responsive Design */
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
            .summary-card {
                flex-direction: column;
                text-align: center;
                gap: 8px;
            }
        }
        /* Loading Spinner */
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
        /* Onboarding Modal */
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

# Function to display a loading spinner
def display_loading(message):
    with st.container():
        st.markdown(f'<div class="spinner"></div><p style="text-align: center; color: #F5F7FA;">{message}</p>', unsafe_allow_html=True)

# Sidebar for settings with improved UX and debug mode
def render_settings_sidebar():
    with st.sidebar:
        st.markdown(
            f'<img src="{LEAN_LOGO_BASE64}" width="120" alt="Lean 2.0 Institute Logo" aria-label="Lean 2.0 Institute Logo" style="display: block; margin: 0 auto 16px;">',
            unsafe_allow_html=True
        )
        st.header("⚙️ Settings", divider="grey")

        # Simulation Controls
        with st.expander("🧪 Simulation", expanded=True):
            team_size = st.slider(
                "Team Size",
                min_value=10, max_value=100, value=DEFAULT_CONFIG['TEAM_SIZE'],
                help="Number of workers in the simulation.",
                key="team_size"
            )
            shift_duration = st.slider(
                "Shift Duration (minutes)",
                min_value=200, max_value=2000, value=DEFAULT_CONFIG['SHIFT_DURATION_MINUTES'], step=2,
                help="Shift duration in minutes (2-minute intervals).",
                key="shift_duration"
            )
            disruption_intervals = st.multiselect(
                "Disruption Times (minutes)",
                options=[i * 2 for i in range(shift_duration // 2)],
                default=[i * 2 for i in DEFAULT_CONFIG['DISRUPTION_INTERVALS']],
                help="Times when disruptions occur.",
                key="disruption_intervals"
            )
            team_initiative = st.selectbox(
                "Team Initiative",
                options=["More frequent breaks", "Team recognition"],
                index=0,
                help="Strategy to improve well-being and safety.",
                key="team_initiative"
            )
            run_simulation = st.button(
                "Run Simulation", 
                key="run_simulation", 
                help="Start a new simulation.",
                type="primary"
            )

        # Visualization Settings
        with st.expander("🎨 Visualizations"):
            high_contrast = st.checkbox(
                "High Contrast Mode", 
                help="Enable high-contrast colors for accessibility.",
                key="high_contrast"
            )
            use_3d_distribution = st.checkbox(
                "3D Team Distribution", 
                help="Use 3D scatter plot for team distribution.",
                key="use_3d_distribution"
            )
            debug_mode = st.checkbox(
                "Debug Mode", 
                help="Show configuration data for debugging.",
                key="debug_mode"
            )

        # Data Management
        with st.expander("💾 Data"):
            load_data = st.button(
                "Load Saved Data", 
                key="load_data", 
                help="Load previously saved simulation data."
            )
            if st.button(
                "Download PDF Report", 
                key="download_report", 
                help="Generate a PDF report."
            ) and 'simulation_results' in st.session_state:
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
                    st.success("PDF report generated as 'workplace_report.tex'. Compile with LaTeX to view.")
                except Exception as e:
                    logger.error(f"Failed to generate report: {str(e)}", extra={'user_action': 'Download PDF Report'})
                    st.error(f"Failed to generate report: {str(e)}.")

        # Debug Information
        if debug_mode:
            with st.expander("🛠️ Debug Info"):
                st.write("**Entry/Exit Points:**")
                st.write(DEFAULT_CONFIG.get('ENTRY_EXIT_POINTS', "Not defined"))
                st.write("**Work Areas:**")
                st.write(DEFAULT_CONFIG.get('WORK_AREAS', "Not defined"))

        # Navigation and Help
        st.header("📋 Navigation", divider="grey")
        tab_names = ["Overview", "Operational Metrics", "Worker Insights", "Downtime", "Glossary"]
        for i, tab in enumerate(tab_names):
            if st.button(tab, key=f"nav_{tab.lower().replace(' ', '_')}", help=f"Go to {tab}"):
                st.session_state.active_tab = i

        if st.button("ℹ️ Help", key="help_button"):
            st.session_state.show_help = True

        if st.button("🚀 Take a Tour", key="tour_button"):
            st.session_state.show_tour = True

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
    if 'show_tour' not in st.session_state:
        st.session_state.show_tour = False
    if 'show_help' not in st.session_state:
        st.session_state.show_help = False

    # Sidebar settings
    team_size, shift_duration, disruption_intervals, team_initiative, run_simulation, load_data, high_contrast, use_3d_distribution, debug_mode = render_settings_sidebar()

    # Precompute minutes
    if st.session_state.simulation_results:
        num_steps = len(st.session_state.simulation_results[0]['step'].unique())
        minutes = [i * 2 for i in range(num_steps)]
    else:
        minutes = [i * 2 for i in range(DEFAULT_CONFIG['SHIFT_DURATION_INTERVALS'])]

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
                st.error(f"Simulation failed: {str(e)}.")

    if load_data:
        with st.spinner("Loading saved data..."):
            try:
                st.session_state.simulation_results = load_simulation_data()
                st.success("Data loaded!", icon="✅")
            except Exception as e:
                logger.error(f"Failed to load data: {str(e)}", extra={'user_action': 'Load Data'})
                st.error(f"Failed to load data: {str(e)}.")

    # Onboarding Modal
    if st.session_state.show_tour:
        with st.container():
            st.markdown("""
                <div class="onboarding-modal" role="dialog" aria-label="Onboarding Tour">
                    <h3>Welcome to the Dashboard!</h3>
                    <p>Explore key features:</p>
                    <ul style="color: #D1D5DB; line-height: 1.6;">
                        <li><strong>Sidebar</strong>: Adjust simulation settings and navigate sections.</li>
                        <li><strong>Tabs</strong>: View metrics, worker insights, and more.</li>
                        <li><strong>Charts</strong>: Hover for details, use sliders to filter.</li>
                        <li><strong>Export</strong>: Generate PDF reports for sharing.</li>
                    </ul>
                    <p>Click below to start exploring!</p>
                </div>
            """, unsafe_allow_html=True)
            if st.button("Get Started", key="close_tour"):
                st.session_state.show_tour = False

    # Help Modal
    if st.session_state.show_help:
        with st.container():
            st.markdown("""
                <div class="onboarding-modal" role="dialog" aria-label="Help Section">
                    <h3>Help & Documentation</h3>
                    <p>Navigate the dashboard:</p>
                    <ul style="color: #D1D5DB; line-height: 1.6;">
                        <li><strong>Overview</strong>: High-level metrics with insights.</li>
                        <li><strong>Operational Metrics</strong>: Trends for performance.</li>
                        <li><strong>Worker Insights</strong>: Well-being and safety data.</li>
                        <li><strong>Downtime</strong>: Analyze downtime trends.</li>
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

    # Hidden selectbox to synchronize tab selection
    tab_index = st.selectbox(
        "Select Tab",
        options=range(len(tab_names)),
        format_func=lambda x: tab_names[x],
        key="tab_selector",
        label_visibility="collapsed"
    )
    if tab_index != st.session_state.active_tab:
        st.session_state.active_tab = tab_index

    # Overview Tab
    with tabs[0]:
        with st.container():
            st.header("Overview", divider="grey")
            st.markdown(
                '<div class="tooltip">Key Metrics<span class="tooltiptext">Summary of Task Compliance, Collaboration Proximity, Worker Well-Being, and Downtime with recommendations.</span></div>',
                unsafe_allow_html=True
            )
            if st.session_state.simulation_results:
                (team_positions_df, task_compliance, collaboration_proximity, operational_recovery,
                 efficiency_metrics_df, productivity_loss, worker_wellbeing, psychological_safety,
                 feedback_impact, downtime_minutes, task_completion_rate) = st.session_state.simulation_results
                compliance_mean = np.mean(task_compliance['data'])
                proximity_mean = np.mean(collaboration_proximity['data'])
                wellbeing_mean = np.mean(worker_wellbeing['scores']) if worker_wellbeing['scores'] else 0
                total_downtime = np.sum(downtime_minutes)
                
                # Summary Cards
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown(f'<div class="summary-card"><h4>Task Compliance</h4><p>{compliance_mean:.1f}%</p></div>', unsafe_allow_html=True)
                with col2:
                    st.markdown(f'<div class="summary-card"><h4>Collaboration</h4><p>{proximity_mean:.1f}%</p></div>', unsafe_allow_html=True)
                with col3:
                    st.markdown(f'<div class="summary-card"><h4>Well-Being</h4><p>{wellbeing_mean:.1f}%</p></div>', unsafe_allow_html=True)
                with col4:
                    st.markdown(f'<div class="summary-card"><h4>Downtime</h4><p>{total_downtime:.1f} min</p></div>', unsafe_allow_html=True)
                
                # Gauge Charts
                col1, col2 = st.columns(2)
                summary_figs = plot_key_metrics_summary(compliance_mean, proximity_mean, wellbeing_mean, total_downtime)
                with col1:
                    st.plotly_chart(summary_figs[0], use_container_width=True)
                    st.plotly_chart(summary_figs[1], use_container_width=True)
                with col2:
                    st.plotly_chart(summary_figs[2], use_container_width=True)
                    if wellbeing_mean < DEFAULT_CONFIG['WELLBEING_THRESHOLD'] * 100:
                        if st.button("Suggest Break Schedule", key="break_schedule"):
                            st.info("Recommended: 10-minute breaks every 60 minutes.")
                    st.plotly_chart(summary_figs[3], use_container_width=True)
            else:
                st.info("Run a simulation or load data to view metrics.", icon="ℹ️")

    # Operational Metrics Tab
    with tabs[1]:
        with st.container():
            st.header("Operational Metrics", divider="grey")
            st.markdown(
                '<div class="tooltip">Performance Trends<span class="tooltiptext">Trends for task compliance, collaboration, recovery, and efficiency.</span></div>',
                unsafe_allow_html=True
            )
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
                filtered_compliance = task_compliance['data'][time_indices[0]:time_indices[1]]
                filtered_z_scores = task_compliance['z_scores'][time_indices[0]:time_indices[1]]
                filtered_forecast = task_compliance['forecast'][time_indices[0]:time_indices[1]] if task_compliance['forecast'] is not None else None
                filtered_disruptions = [t for t in DEFAULT_CONFIG['DISRUPTION_INTERVALS'] if time_indices[0] <= t < time_indices[1]]
                compliance_fig = plot_task_compliance_score(filtered_compliance, filtered_disruptions, filtered_forecast, filtered_z_scores)
                st.plotly_chart(compliance_fig, use_container_width=True)
                filtered_collab = collaboration_proximity['data'][time_indices[0]:time_indices[1]]
                filtered_forecast = collaboration_proximity['forecast'][time_indices[0]:time_indices[1]] if collaboration_proximity['forecast'] is not None else None
                collaboration_fig = plot_collaboration_proximity_index(filtered_collab, filtered_disruptions, filtered_forecast)
                st.plotly_chart(collaboration_fig, use_container_width=True)
                with st.expander("Additional Metrics"):
                    filtered_recovery = operational_recovery[time_indices[0]:time_indices[1]]
                    filtered_loss = productivity_loss[time_indices[0]:time_indices[1]]
                    resilience_fig = plot_operational_recovery(filtered_recovery, filtered_loss)
                    st.plotly_chart(resilience_fig, use_container_width=True)
                    selected_metrics = st.multiselect(
                        "Efficiency Metrics",
                        options=['uptime', 'throughput', 'quality', 'oee'],
                        default=['uptime', 'throughput', 'quality', 'oee'],
                        key="efficiency_metrics_op"
                    )
                    filtered_df = efficiency_metrics_df.iloc[time_indices[0]:time_indices[1]]
                    efficiency_fig = plot_operational_efficiency(filtered_df, selected_metrics)
                    st.plotly_chart(efficiency_fig, use_container_width=True)
            else:
                st.info("Run a simulation or load data to view metrics.", icon="ℹ️")

    # Worker Insights Tab
    with tabs[2]:
        with st.container():
            st.header("Worker Insights", divider="grey")
            st.markdown(
                '<div class="tooltip">Worker Metrics<span class="tooltiptext">Distribution, well-being, and safety metrics.</span></div>',
                unsafe_allow_html=True
            )
            if st.session_state.simulation_results:
                with st.expander("Worker Distribution", expanded=True):
                    time_range = st.slider(
                        "Time Range (minutes)",
                        min_value=0,
                        max_value=DEFAULT_CONFIG['SHIFT_DURATION_MINUTES'] - 2,
                        value=(0, DEFAULT_CONFIG['SHIFT_DURATION_MINUTES'] - 2),
                        step=2,
                        key="time_range_dist"
                    )
                    time_indices = (time_range[0] // 2, time_range[1] // 2 + 1)
                    zone_filter = st.selectbox(
                        "Zone", 
                        options=["All"] + list(DEFAULT_CONFIG['WORK_AREAS'].keys())
                    )
                    filtered_df = team_positions_df if zone_filter == "All" else team_positions_df[team_positions_df['zone'] == zone_filter]
                    filtered_df = filtered_df[(filtered_df['step'] >= time_indices[0]) & (filtered_df['step'] < time_indices[1])]
                    show_entry_exit = st.checkbox("Show Entry/Exit Points", value=True, key="show_entry_exit_dist")
                    show_production_lines = st.checkbox("Show Production Lines", value=True, key="show_production_lines_dist")
                    col_dist1, col_dist2 = st.columns(2)
                    with col_dist1:
                        st.markdown("### Worker Positions")
                        selected_step = st.slider(
                            "Time Step",
                            min_value=int(time_indices[0]),
                            max_value=int(time_indices[1] - 1),
                            value=int(time_indices[0]),
                            key="team_distribution_step"
                        )
                        try:
                            distribution_fig = plot_worker_distribution(
                                filtered_df, DEFAULT_CONFIG['FACILITY_SIZE'], DEFAULT_CONFIG, use_3d=use_3d_distribution,
                                selected_step=selected_step, show_entry_exit=show_entry_exit, show_production_lines=show_production_lines
                            )
                            st.plotly_chart(distribution_fig, use_container_width=True)
                        except Exception as e:
                            logger.error(f"Failed to plot worker distribution: {str(e)}", extra={'user_action': 'Render Worker Insights'})
                            st.error(f"Error rendering worker distribution: {str(e)}. Check debug mode for details.")
                    with col_dist2:
                        st.markdown("### Density Heatmap")
                        try:
                            heatmap_fig = plot_worker_density_heatmap(
                                filtered_df, DEFAULT_CONFIG['FACILITY_SIZE'], DEFAULT_CONFIG,
                                show_entry_exit=show_entry_exit, show_production_lines=show_production_lines
                            )
                            st.plotly_chart(heatmap_fig, use_container_width=True)
                        except Exception as e:
                            logger.error(f"Failed to plot density heatmap: {str(e)}", extra={'user_action': 'Render Worker Insights'})
                            st.error(f"Error rendering density heatmap: {str(e)}. Check debug mode for details.")

                with st.expander("Worker Well-Being & Safety"):
                    time_range = st.slider(
                        "Time Range (minutes)",
                        min_value=0,
                        max_value=DEFAULT_CONFIG['SHIFT_DURATION_MINUTES'] - 2,
                        value=(0, DEFAULT_CONFIG['SHIFT_DURATION_MINUTES'] - 2),
                        step=2,
                        key="time_range_well"
                    )
                    time_indices = (time_range[0] // 2, time_range[1] // 2 + 1)
                    col_well1, col_well2 = st.columns(2)
                    with col_well1:
                        st.markdown("### Well-Being Index")
                        filtered_scores = worker_wellbeing['scores'][time_indices[0]:time_indices[1]]
                        filtered_triggers = {
                            'threshold': [t for t in worker_wellbeing['triggers']['threshold'] if time_indices[0] <= t < time_indices[1]],
                            'trend': [t for t in worker_wellbeing['triggers']['trend'] if time_indices[0] <= t < time_indices[1]],
                            'work_area': {k: [t for t in v if time_indices[0] <= t < time_indices[1]] for k, v in worker_wellbeing['triggers']['work_area'].items()},
                            'disruption': [t for t in worker_wellbeing['triggers']['disruption'] if time_indices[0] <= t < time_indices[1]]
                        }
                        wellbeing_fig = plot_worker_wellbeing(filtered_scores, filtered_triggers)
                        st.plotly_chart(wellbeing_fig, use_container_width=True)
                    with col_well2:
                        st.markdown("### Psychological Safety")
                        filtered_safety = psychological_safety[time_indices[0]:time_indices[1]]
                        safety_fig = plot_psychological_safety(filtered_safety)
                        st.plotly_chart(safety_fig, use_container_width=True)
                    st.markdown("### Well-Being Triggers")
                    st.write(f"**Threshold Alerts (< {DEFAULT_CONFIG['WELLBEING_THRESHOLD']*100}%):** {filtered_triggers['threshold']}")
                    st.write(f"**Trend Alerts (Declining):** {filtered_triggers['trend']}")
                    st.write("**Work Area Alerts:**")
                    for zone, triggers in filtered_triggers['work_area'].items():
                        st.write(f"{zone}: {triggers}")
                    st.write(f"**Disruption Alerts:** {filtered_triggers['disruption']}")
            else:
                st.info("Run a simulation or load data to view insights.", icon="ℹ️")

    # Downtime Tab
    with tabs[3]:
        with st.container():
            st.header("Downtime Analysis", divider="grey")
            st.markdown(
                '<div class="tooltip">Downtime Trends<span class="tooltiptext">Downtime with alerts for high values.</span></div>',
                unsafe_allow_html=True
            )
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
        with st.container():
            st.header("Glossary", divider="grey")
            st.markdown("""
                ### Metric Definitions
                - **Task Compliance Score**: % of tasks completed correctly (0–100%).
                - **Collaboration Proximity Index**: % of workers within 5m of colleagues (0–100%).
                - **Operational Recovery Score**: Ability to maintain output post-disruption (0–100%).
                - **Worker Well-Being Index**: Fatigue, stress, and satisfaction (0–100%).
                - **Psychological Safety Score**: Comfort in reporting issues (0–100%).
                - **Uptime**: % of time equipment is operational (0–100%).
                - **Throughput**: % of max production rate (0–100%).
                - **Quality**: % of products meeting standards (0–100%).
                - **OEE**: Combined uptime, throughput, quality (0–100%).
                - **Productivity Loss**: % of output lost (0–100%).
                - **Downtime**: Minutes of unplanned stops.
                - **Task Completion Rate**: % of tasks completed per interval (0–100%).
                - **Feedback Impact**: Improvement from initiatives.

                ### Terms
                - **Disruption**: Event causing performance drop.
                - **Team Initiative**: Strategies like breaks or recognition.
                - **Anomaly**: Significant deviation (z-score > 2.0).
            """)

# High-contrast mode
def apply_high_contrast_mode():
    st.markdown("""
        <style>
            .main { background-color: #000000; color: #FFFFFF; }
            h1, h2, h3 { color: #FFFFFF; }
            .stButton>button { background-color: #FFFFFF; color: #000000; }
            .stButton>button:hover, .stButton>button:focus { background-color: #FFFF00; color: #000000; }
            .stSelectbox, .stSlider, .stMultiSelect { background-color: #333333; color: #FFFFFF; }
            [data-testid="stSidebar"] { background-color: #111111; color: #FFFFFF; }
            .stMetric, .stExpander, .summary-card { background-color: #333333; }
            .recommendation { color: #FFFF00; }
            .tooltip .tooltiptext { background-color: #333333; color: #FFFFFF; }
            .stTabs [data-baseweb="tab-list"] { background-color: #111111; }
            .stTabs [data-baseweb="tab"] { color: #FFFFFF; }
            .stTabs [data-baseweb="tab"][aria-selected="true"], .stTabs [data-baseweb="tab"]:hover {
                background-color: #FFFF00;
                color: #000000;
            }
            .stPlotlyChart { background-color: #333333; }
            .onboarding-modal { background-color: #333333; }
        </style>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    if st.session_state.get('high_contrast', False):
        apply_high_contrast_mode()
