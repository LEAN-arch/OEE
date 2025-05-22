# main.py
# Streamlit dashboard for the Workplace Shift Monitoring Dashboard.
# Enhanced for professional visuals, seamless UX, accessibility, fixed tab rendering, debug mode, and error handling for plot_task_compliance_score.
# Improved Worker Insights section with actionable, high-quality visualizations.

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
        .action-card {
            background-color: #2D3B55;
            border-radius: 6px;
            padding: 16px;
            margin: 12px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .action-card h4 {
            color: #F5F7FA;
            font-size: 1.1rem;
            margin: 0 0 8px;
        }
        .action-card p {
            color: #D1D5DB;
            font-size: 0.9rem;
            margin: 0;
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
                flex-direction: column;
                text-align: center;
                gap: 8px;
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

        if debug_mode:
            with st.expander("🛠️ Debug Info"):
                st.write("**Entry/Exit Points:**")
                st.write(DEFAULT_CONFIG.get('ENTRY_EXIT_POINTS', "Not defined"))
                st.write("**Work Areas:**")
                st.write(DEFAULT_CONFIG.get('WORK_AREAS', "Not defined"))

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

    # Worker Insights Tab (Enhanced)
    with tabs[2]:
        with st.container():
            st.header("Worker Insights", divider="grey")
            st.markdown(
                '<div class="tooltip">Worker Metrics<span class="tooltiptext">Analyze worker distribution, well-being, and psychological safety with actionable insights.</span></div>',
                unsafe_allow_html=True
            )
            if st.session_state.simulation_results:
                (team_positions_df, task_compliance, collaboration_proximity, operational_recovery,
                 efficiency_metrics_df, productivity_loss, worker_wellbeing, psychological_safety,
                 feedback_impact, downtime_minutes, task_completion_rate) = st.session_state.simulation_results
                
                with st.expander("Worker Distribution", expanded=True):
                    st.markdown("### Team Distribution and Density")
                    col_filters, col_empty = st.columns([3, 1])
                    with col_filters:
                        time_range = st.slider(
                            "Time Range (minutes)",
                            min_value=0,
                            max_value=DEFAULT_CONFIG['SHIFT_DURATION_MINUTES'] - 2,
                            value=(0, DEFAULT_CONFIG['SHIFT_DURATION_MINUTES'] - 2),
                            step=2,
                            key="time_range_dist",
                            help="Select the time range for analysis."
                        )
                        zone_filter = st.selectbox(
                            "Zone",
                            options=["All"] + list(DEFAULT_CONFIG['WORK_AREAS'].keys()),
                            key="zone_filter_dist",
                            help="Filter by work zone."
                        )
                        role_filter = st.multiselect(
                            "Worker Roles",
                            options=["All", "Operator", "Supervisor", "Technician"],
                            default=["All"],
                            key="role_filter_dist",
                            help="Filter by worker roles."
                        )
                        heatmap_intensity = st.slider(
                            "Heatmap Intensity",
                            min_value=0.5,
                            max_value=2.0,
                            value=1.0,
                            step=0.1,
                            key="heatmap_intensity",
                            help="Adjust heatmap sensitivity."
                        )
                    time_indices = (time_range[0] // 2, time_range[1] // 2 + 1)
                    filtered_df = team_positions_df if zone_filter == "All" else team_positions_df[team_positions_df['zone'] == zone_filter]
                    if "All" not in role_filter:
                        filtered_df = filtered_df[filtered_df['role'].isin(role_filter)]
                    filtered_df = filtered_df[(filtered_df['step'] >= time_indices[0]) & (filtered_df['step'] < time_indices[1])]
                    show_entry_exit = st.checkbox(
                        "Show Entry/Exit Points",
                        value=True,
                        key="show_entry_exit_dist",
                        help="Toggle entry/exit points on plots."
                    )
                    show_production_lines = st.checkbox(
                        "Show Production Lines",
                        value=True,
                        key="show_production_lines_dist",
                        help="Toggle production line boundaries."
                    )
                    
                    # Actionable Recommendations
                    if not filtered_df.empty:
                        density_stats = filtered_df.groupby('zone').size().to_dict()
                        overcrowded_zones = [zone for zone, count in density_stats.items() if count > DEFAULT_CONFIG['TEAM_SIZE'] * 0.3]
                        if overcrowded_zones:
                            st.markdown(
                                f'<div class="action-card"><h4>Action: Redistribute Workers</h4><p>Overcrowding detected in {", ".join(overcrowded_zones)}. Consider reassigning workers to balance workload.</p></div>',
                                unsafe_allow_html=True
                            )
                        if st.button("Suggest Reassignment Plan", key="reassign_workers"):
                            st.info(f"Recommendation: Move 2-3 workers from {overcrowded_zones[0] if overcrowded_zones else 'high-density zones'} to underutilized zones (e.g., Quality Control).")
                    
                    col_dist1, col_dist2 = st.columns(2)
                    with col_dist1:
                        st.markdown("#### Worker Positions")
                        selected_step = st.slider(
                            "Time Step",
                            min_value=int(time_indices[0]),
                            max_value=int(time_indices[1] - 1),
                            value=int(time_indices[0]),
                            key="team_distribution_step",
                            help="Select a specific time step."
                        )
                        try:
                            distribution_fig = plot_worker_distribution(
                                filtered_df, DEFAULT_CONFIG['FACILITY_SIZE'], DEFAULT_CONFIG,
                                use_3d=use_3d_distribution, selected_step=selected_step,
                                show_entry_exit=show_entry_exit, show_production_lines=show_production_lines,
                                high_contrast=high_contrast
                            )
                            st.plotly_chart(distribution_fig, use_container_width=True)
                        except Exception as e:
                            logger.error(f"Failed to plot worker distribution: {str(e)}", extra={'user_action': 'Render Worker Insights'})
                            st.error(f"Error rendering worker distribution: {str(e)}. Check debug mode for details.")
                    with col_dist2:
                        st.markdown("#### Density Heatmap")
                        try:
                            heatmap_fig = plot_worker_density_heatmap(
                                filtered_df, DEFAULT_CONFIG['FACILITY_SIZE'], DEFAULT_CONFIG,
                                show_entry_exit=show_entry_exit, show_production_lines=show_production_lines,
                                intensity=heatmap_intensity, high_contrast=high_contrast
                            )
                            st.plotly_chart(heatmap_fig, use_container_width=True)
                        except Exception as e:
                            logger.error(f"Failed to plot density heatmap: {str(e)}", extra={'user_action': 'Render Worker Insights'})
                            st.error(f"Error rendering density heatmap: {str(e)}. Check debug mode for details.")

                with st.expander("Worker Well-Being & Safety", expanded=True):
                    st.markdown("### Well-Being and Psychological Safety")
                    time_range = st.slider(
                        "Time Range (minutes)",
                        min_value=0,
                        max_value=DEFAULT_CONFIG['SHIFT_DURATION_MINUTES'] - 2,
                        value=(0, DEFAULT_CONFIG['SHIFT_DURATION_MINUTES'] - 2),
                        step=2,
                        key="time_range_well",
                        help="Select the time range for analysis."
                    )
                    time_indices = (time_range[0] // 2, time_range[1] // 2 + 1)
                    col_well1, col_well2 = st.columns(2)
                    with col_well1:
                        st.markdown("#### Well-Being Index")
                        filtered_scores = worker_wellbeing['scores'][time_indices[0]:time_indices[1]]
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
                            logger.error(f"Failed to plot well-being: {str(e)}", extra={'user_action': 'Render Worker Insights'})
                            st.error(f"Error rendering well-being chart: {str(e)}.")
                    
                    with col_well2:
                        st.markdown("#### Psychological Safety")
                        filtered_safety = psychological_safety[time_indices[0]:time_indices[1]]
                        try:
                            safety_fig = plot_psychological_safety(
                                filtered_safety, high_contrast=high_contrast
                            )
                            st.plotly_chart(safety_fig, use_container_width=True)
                        except Exception as e:
                            logger.error(f"Failed to plot psychological safety: {str(e)}", extra={'user_action': 'Render Worker Insights'})
                            st.error(f"Error rendering psychological safety chart: {str(e)}.")
                    
                    # Actionable Recommendations
                    wellbeing_mean = np.mean(filtered_scores) if filtered_scores else 0
                    safety_mean = np.mean(filtered_safety) if filtered_safety else 0
                    if wellbeing_mean < DEFAULT_CONFIG['WELLBEING_THRESHOLD'] * 100:
                        st.markdown(
                            '<div class="action-card"><h4>Action: Schedule Breaks</h4><p>Low well-being detected (average: {:.1f}%). Schedule 10-minute breaks every 60 minutes.</p></div>'.format(wellbeing_mean),
                            unsafe_allow_html=True
                        )
                        if st.button("Generate Break Schedule", key="break_schedule_well"):
                            st.info("Suggested: Breaks at 60, 120, 180 minutes for affected workers.")
                    if safety_mean < 70:
                        st.markdown(
                            '<div class="action-card"><h4>Action: Conduct Safety Training</h4><p>Low psychological safety (average: {:.1f}%). Organize a team workshop to encourage open communication.</p></div>'.format(safety_mean),
                            unsafe_allow_html=True
                        )
                        if st.button("Plan Safety Workshop", key="safety_workshop"):
                            st.info("Suggested: 1-hour workshop on reporting protocols next shift.")
                    
                    st.markdown("#### Well-Being Triggers")
                    st.write(f"**Threshold Alerts (< {DEFAULT_CONFIG['WELLBEING_THRESHOLD']*100}%):** {filtered_triggers['threshold']}")
                    st.write(f"**Trend Alerts (Declining):** {filtered_triggers['trend']}")
                    st.write("**Work Area Alerts:**")
                    for zone, triggers in filtered_triggers['work_area'].items():
                        st.write(f"{zone}: {triggers}")
                    st.write(f"**Disruption Alerts:** {filtered_triggers['disruption']}")
            else:
                st.info("Run a simulation or load data to view insights.", icon="ℹ️")

if __name__ == "__main__":
    main()
