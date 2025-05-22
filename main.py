# main.py
# Streamlit dashboard for the Workplace Shift Monitoring Dashboard.
# Enhanced for professional visuals, seamless UX, accessibility, fixed tab rendering, debug mode, and error handling for plot_task_compliance_score.
# Fixed nesting issue in render_settings_sidebar to prevent StreamlitAPIException.
# Added input validation for plot_key_metrics_summary to prevent ValueError in visualizations.py.
# Fixed syntax error in Help Modal (incomplete 'if' statement).
# Verified import statement for visualizations to resolve SyntaxError at line 14.
# Added debug log to confirm file parsing.

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

# Debug log to confirm file parsing
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s - [User Action: %(user_action)s]',
    filename='dashboard.log'
)
logger.info("Successfully parsed main.py imports", extra={'user_action': 'Parse File'})

# Streamlit page config
st.set_page_config(
    page_title="Workplace Shift Monitoring Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional, accessible design with improved typography
st.markdown("""
    <style>
        /* Base Styles */
        .main {
            background-color: #1E2A44;
            color: #F5F7FA;
            font-family: 'Roboto', 'Open Sans', 'Helvetica Neue', sans-serif;
            padding: 32px;
        }
        h1 {
            font-size: 2.5rem;
            font-weight: 700;
            line-height: 1.2;
            letter-spacing: -0.02em;
            text-align: center;
            margin-bottom: 2rem;
        }
        h2 {
            font-size: 1.875rem;
            font-weight: 600;
            line-height: 1.3;
            margin: 1.5rem 0 1rem;
        }
        h3 {
            font-size: 1.375rem;
            font-weight: 500;
            line-height: 1.4;
            margin-bottom: 0.75rem;
        }
        .stButton>button {
            background-color: #4F46E5;
            color: #F5F7FA;
            border-radius: 8px;
            padding: 10px 20px;
            font-size: 1rem;
            font-weight: 500;
            transition: all 0.3s ease;
        }
        .stButton>button:hover, .stButton>button:focus {
            background-color: #EC4899;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.3);
            outline: none;
        }
        .stSelectbox, .stSlider, .stMultiSelect {
            background-color: #2D3B55;
            color: #F5F7FA;
            border-radius: 8px;
            padding: 8px;
            margin-bottom: 16px;
            font-size: 1rem;
        }
        .tooltip {
            position: relative;
            display: inline-flex;
            align-items: center;
            margin-left: 8px;
        }
        .tooltip .tooltiptext {
            visibility: hidden;
            width: 300px;
            background-color: #2D3B55;
            color: #F5F7FA;
            text-align: left;
            border-radius: 8px;
            padding: 16px;
            position: absolute;
            z-index: 10;
            top: 100%;
            left: 50%;
            margin-left: -150px;
            opacity: 0;
            transition: opacity 0.3s;
            box-shadow: 0 4px 8px rgba(0,0,0,0.3);
            font-size: 0.875rem;
            line-height: 1.5;
        }
        .tooltip:hover .tooltiptext, .tooltip:focus .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
        [data-testid="stSidebar"] {
            background-color: #2D3B55;
            color: #F5F7FA;
            padding: 24px;
            border-right: 1px solid #4B5EAA;
            font-size: 1rem;
        }
        [data-testid="stSidebar"] .stButton>button {
            background-color: #10B981;
        }
        [data-testid="stSidebar"] .stButton>button:hover, [data-testid="stSidebar"] .stButton>button:focus {
            background-color: #EC4899;
        }
        .stMetric {
            background-color: #2D3B55;
            border-radius: 8px;
            padding: 20px;
            margin: 16px 0;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            font-size: 1.125rem;
        }
        .stExpander {
            background-color: #2D3B55;
            border-radius: 8px;
            margin: 16px 0;
            border: 1px solid #4B5EAA;
        }
        .recommendation {
            color: #FBBF24;
            font-size: 0.875rem;
            margin-top: 12px;
            font-style: italic;
        }
        .stTabs [data-baseweb="tab-list"] {
            background-color: #2D3B55;
            border-radius: 8px;
            padding: 10px;
            display: flex;
            justify-content: center;
            gap: 10px;
        }
        .stTabs [data-baseweb="tab"] {
            color: #D1D5DB;
            padding: 12px 24px;
            border-radius: 8px;
            font-weight: 500;
            font-size: 1rem;
            transition: all 0.3s ease;
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
            margin: 16px 0;
            background-color: #2D3B55;
            border-radius: 8px;
            padding: 16px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        /* Enhanced Summary Cards */
        .summary-card {
            background-color: #2D3B55;
            border-radius: 8px;
            padding: 20px;
            margin: 16px 0;
            display: flex;
            flex-direction: row;
            align-items: center;
            gap: 16px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            transition: transform 0.2s ease;
        }
        .summary-card:hover {
            transform: translateY(-4px);
        }
        .summary-card h4 {
            font-size: 1.25rem;
            font-weight: 600;
            margin: 0;
        }
        .summary-card p {
            font-size: 1.75rem;
            font-weight: 700;
            color: #FBBF24;
            margin: 0;
        }
        /* Plot Container */
        .plot-container {
            background-color: #2D3B55;
            border-radius: 8px;
            padding: 16px;
            margin: 16px 0;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        /* Data Table */
        .stDataFrame {
            background-color: #2D3B55;
            border-radius: 8px;
            padding: 16px;
            font-size: 0.875rem;
        }
        /* Responsive Design */
        @media (max-width: 768px) {
            .main { padding: 16px; }
            h1 { font-size: 2rem; }
            h2 { font-size: 1.5rem; }
            h3 { font-size: 1.25rem; }
            .stColumn {
                width: 100% !important;
                margin-bottom: 1.5rem;
            }
            .stPlotlyChart {
                height: 360px !important;
            }
            .summary-card {
                flex-direction: column;
                text-align: center;
                gap: 12px;
            }
            .stTabs [data-baseweb="tab"] {
                padding: 10px 16px;
                font-size: 0.875rem;
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
            line-height: 1.6;
            margin-bottom: 16px;
        }
    </style>
""", unsafe_allow_html=True)

# Function to display a loading spinner
def display_loading(message):
    with st.container():
        st.markdown(f'<div class="spinner"></div><p style="text-align: center; color: #F5F7FA;">{message}</p>', unsafe_allow_html=True)

# Sidebar for settings with fixed nesting issue
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

        # Export Options (moved out of nested expander)
        with st.expander("📊 Export Options"):
            if 'simulation_results' in st.session_state:
                if st.button("Export Plots as PNG", key="export_png"):
                    st.info("Exporting plots as PNG is handled within each plot.")
                if st.button("Export Data as CSV", key="export_csv"):
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
                    st.download_button(
                        label="Download Summary CSV",
                        data=summary_df.to_csv(index=False),
                        file_name="workplace_summary.csv",
                        mime="text/csv"
                    )
            else:
                st.info("Run a simulation to enable export options.", icon="ℹ️")

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
                        <li><-nanodisabled>Sidebar</nanodisabled>: Adjust simulation settings and navigate sections.</li>
                        <li><nanodisabled>Tabs</nanodisabled>: View metrics, worker insights, and more.</li>
                        <li><nanodisabled>Charts</nanodisabled>: Hover for details, use sliders to filter, export as PNG.</li>
                        <li><nanodisabled>Export</nanodisabled>: Download data as CSV or generate PDF reports.</li>
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
                        <li><nanodisabled>Overview</nanodisabled>: High-level metrics with insights.</li>
                        <li><nanodisabled>Operational Metrics</nanodisabled>: Trends for performance.</li>
                        <li><nanodisabled>Worker Insights</nanodisabled>: Well-being and safety data.</li>
                        <li><nanodisabled>Downtime</nanodisabled>: Analyze downtime trends.</li>
                        <li><nanodisabled>Glossary</nanodisabled>: Metric definitions.</li>
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
                '<div class="tooltip">Key Metrics<span class="tooltiptext">Summary of Task Compliance, Collaboration Proximity, Worker Well-Being, and Downtime with actionable insights.</span></div>',
                unsafe_allow_html=True
            )
            if st.session_state.simulation_results:
                (team_positions_df, task_compliance, collaboration_proximity, operational_recovery,
                 efficiency_metrics_df, productivity_loss, worker_wellbeing, psychological_safety,
                 feedback_impact, downtime_minutes, task_completion_rate) = st.session_state.simulation_results
                
                # Validate and compute means with fallbacks
                inputs = {
                    "compliance_mean": np.mean(task_compliance['data']) if task_compliance['data'] and not np.all(np.isnan(task_compliance['data'])) else 0.0,
                    "proximity_mean": np.mean(collaboration_proximity['data']) if collaboration_proximity['data'] and not np.all(np.isnan(collaboration_proximity['data'])) else 0.0,
                    "wellbeing_mean": np.mean(worker_wellbeing['scores']) if worker_wellbeing['scores'] and not np.all(np.isnan(worker_wellbeing['scores'])) else 0.0,
                    "total_downtime": np.sum(downtime_minutes) if downtime_minutes and not np.all(np.isnan(downtime_minutes)) else 0.0
                }

                # Log inputs and validate
                for name, value in inputs.items():
                    if not isinstance(value, (int, float)) or np.isnan(value):
                        logger.error(f"Invalid input in main.py: {name}={value}, type={type(value)}", extra={'user_action': 'Render Overview Metrics'})
                        inputs[name] = 0.0  # Fallback to 0.0
                    logger.info(f"Input in main.py: {name}={value}, type={type(value)}", extra={'user_action': 'Render Overview Metrics'})

                compliance_mean = inputs["compliance_mean"]
                proximity_mean = inputs["proximity_mean"]
                wellbeing_mean = inputs["wellbeing_mean"]
                total_downtime = inputs["total_downtime"]

                # Call plot_key_metrics_summary
                summary_figs = plot_key_metrics_summary(compliance_mean, proximity_mean, wellbeing_mean, total_downtime)
                
                # Enhanced Metrics Display
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Task Compliance", f"{compliance_mean:.1f}%", delta=f"{compliance_mean - 75:.1f}%", delta_color="normal")
                with col2:
                    st.metric("Collaboration", f"{proximity_mean:.1f}%", delta=f"{proximity_mean - 60:.1f}%", delta_color="normal")
                with col3:
                    st.metric("Well-Being", f"{wellbeing_mean:.1f}%", delta=f"{wellbeing_mean - 70:.1f}%", delta_color="normal")
                with col4:
                    st.metric("Downtime", f"{total_downtime:.1f} min", delta=f"{total_downtime - 30:.1f} min", delta_color="inverse")
                
                # Data Table
                with st.expander("View Detailed Data", expanded=False):
                    summary_df = pd.DataFrame({
                        'Time (min)': [i * 2 for i in range(len(task_compliance['data']))],
                        'Task Compliance (%)': task_compliance['data'],
                        'Collaboration (%)': collaboration_proximity['data'],
                        'Well-Being (%)': worker_wellbeing['scores'],
                        'Downtime (min)': downtime_minutes
                    })
                    st.dataframe(summary_df, use_container_width=True, height=300)
                
                # Gauge Charts
                col1, col2 = st.columns(2)
                for i, fig in enumerate(summary_figs):
                    with st.container(border=True):
                        st.markdown(f'<div class="plot-container">', unsafe_allow_html=True)
                        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True, 'toImageButtonOptions': {'format': 'png'}})
                        st.markdown('</div>', unsafe_allow_html=True)
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
                try:
                    filtered_compliance = task_compliance['data'][time_indices[0]:time_indices[1]]
                    filtered_z_scores = task_compliance['z_scores'][time_indices[0]:time_indices[1]]
                    filtered_forecast = task_compliance['forecast'][time_indices[0]:time_indices[1]] if task_compliance['forecast'] is not None else None
                    filtered_disruptions = [t for t in DEFAULT_CONFIG['DISRUPTION_INTERVALS'] if time_indices[0] <= t < time_indices[1]]
                    
                    if not filtered_compliance or not filtered_z_scores:
                        logger.error(
                            f"Empty input data: compliance={len(filtered_compliance)}, z_scores={len(filtered_z_scores)}",
                            extra={'user_action': 'Render Operational Metrics'}
                        )
                        st.error("No data available for the selected time range.")
                    elif len(filtered_compliance) != len(filtered_z_scores) or (filtered_forecast is not None and len(filtered_forecast) != len(filtered_compliance)):
                        logger.error(
                            f"Input length mismatch: compliance={len(filtered_compliance)}, "
                            f"z_scores={len(filtered_z_scores)}, forecast={'None' if filtered_forecast is None else len(filtered_forecast)}",
                            extra={'user_action': 'Render Operational Metrics'}
                        )
                        st.error("Input data lengths do not match.")
                    else:
                        with st.container(border=True):
                            st.markdown(f'<div class="plot-container">', unsafe_allow_html=True)
                            compliance_fig = plot_task_compliance_score(filtered_compliance, filtered_disruptions, filtered_forecast, filtered_z_scores)
                            st.plotly_chart(compliance_fig, use_container_width=True, config={'displayModeBar': True, 'toImageButtonOptions': {'format': 'png'}})
                            st.markdown('</div>', unsafe_allow_html=True)
                except Exception as e:
                    logger.error(
                        f"Failed to render task compliance chart: {str(e)}",
                        extra={'user_action': 'Render Operational Metrics'}
                    )
                    st.error(f"Error rendering task compliance chart: {str(e)}.")
                
                try:
                    filtered_collab = collaboration_proximity['data'][time_indices[0]:time_indices[1]]
                    filtered_forecast = collaboration_proximity['forecast'][time_indices[0]:time_indices[1]] if collaboration_proximity['forecast'] is not None else None
                    if not filtered_collab:
                        logger.error(
                            f"Empty collaboration data: length={len(filtered_collab)}",
                            extra={'user_action': 'Render Operational Metrics'}
                        )
                        st.error("No collaboration data available for the selected time range.")
                    else:
                        with st.container(border=True):
                            st.markdown(f'<div class="plot-container">', unsafe_allow_html=True)
                            collaboration_fig = plot_collaboration_proximity_index(filtered_collab, filtered_disruptions, filtered_forecast)
                            st.plotly_chart(collaboration_fig, use_container_width=True, config={'displayModeBar': True, 'toImageButtonOptions': {'format': 'png'}})
                            st.markdown('</div>', unsafe_allow_html=True)
                except Exception as e:
                    logger.error(
                        f"Failed to render collaboration chart: {str(e)}",
                        extra={'user_action': 'Render Operational Metrics'}
                    )
                    st.error(f"Error rendering collaboration chart: {str(e)}.")
                
                with st.expander("Additional Metrics"):
                    try:
                        filtered_recovery = operational_recovery[time_indices[0]:time_indices[1]]
                        filtered_loss = productivity_loss[time_indices[0]:time_indices[1]]
                        if not filtered_recovery or not filtered_loss:
                            logger.error(
                                f"Empty additional metrics: recovery={len(filtered_recovery)}, loss={len(filtered_loss)}",
                                extra={'user_action': 'Render Operational Metrics'}
                            )
                            st.error("No data available for additional metrics.")
                        else:
                            with st.container(border=True):
                                st.markdown(f'<div class="plot-container">', unsafe_allow_html=True)
                                resilience_fig = plot_operational_recovery(filtered_recovery, filtered_loss)
                                st.plotly_chart(resilience_fig, use_container_width=True, config={'displayModeBar': True, 'toImageButtonOptions': {'format': 'png'}})
                                st.markdown('</div>', unsafe_allow_html=True)
                    except Exception as e:
                        logger.error(
                            f"Failed to render operational recovery chart: {str(e)}",
                            extra={'user_action': 'Render Operational Metrics'}
                        )
                        st.error(f"Error rendering operational recovery chart: {str(e)}.")
                    
                    try:
                        selected_metrics = st.multiselect(
                            "Efficiency Metrics",
                            options=['uptime', 'throughput', 'quality', 'oee'],
                            default=['uptime', 'throughput', 'quality', 'oee'],
                            key="efficiency_metrics_op"
                        )
                        filtered_df = efficiency_metrics_df.iloc[time_indices[0]:time_indices[1]]
                        if filtered_df.empty:
                            logger.error(
                                f"Empty efficiency data: rows={len(filtered_df)}",
                                extra={'user_action': 'Render Operational Metrics'}
                            )
                            st.error("No efficiency data available for the selected time range.")
                        else:
                            with st.container(border=True):
                                st.markdown(f'<div class="plot-container">', unsafe_allow_html=True)
                                efficiency_fig = plot_operational_efficiency(filtered_df, selected_metrics)
                                st.plotly_chart(efficiency_fig, use_container_width=True, config={'displayModeBar': True, 'toImageButtonOptions': {'format': 'png'}})
                                st.markdown('</div>', unsafe_allow_html=True)
                    except Exception as e:
                        logger.error(
                            f"Failed to render efficiency chart: {str(e)}",
                            extra={'user_action': 'Render Operational Metrics'}
                        )
                        st.error(f"Error rendering efficiency chart: {str(e)}.")
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
                            with st.container(border=True):
                                st.markdown(f'<div class="plot-container">', unsafe_allow_html=True)
                                distribution_fig = plot_worker_distribution(
                                    filtered_df, DEFAULT_CONFIG['FACILITY_SIZE'], DEFAULT_CONFIG, use_3d=use_3d_distribution,
                                    selected_step=selected_step, show_entry_exit=show_entry_exit, show_production_lines=show_production_lines
                                )
                                st.plotly_chart(distribution_fig, use_container_width=True, config={'displayModeBar': True, 'toImageButtonOptions': {'format': 'png'}})
                                st.markdown('</div>', unsafe_allow_html=True)
                        except Exception as e:
                            logger.error(f"Failed to plot worker distribution: {str(e)}", extra={'user_action': 'Render Worker Insights'})
                            st.error(f"Error rendering worker distribution: {str(e)}. Check debug mode for details.")
                    with col_dist2:
                        st.markdown("### Density Heatmap")
                        try:
                            with st.container(border=True):
                                st.markdown(f'<div class="plot-container">', unsafe_allow_html=True)
                                heatmap_fig = plot_worker_density_heatmap(
                                    filtered_df, DEFAULT_CONFIG['FACILITY_SIZE'], DEFAULT_CONFIG,
                                    show_entry_exit=show_entry_exit, show_production_lines=show_production_lines
                                )
                                st.plotly_chart(heatmap_fig, use_container_width=True, config={'displayModeBar': True, 'toImageButtonOptions': {'format': 'png'}})
                                st.markdown('</div>', unsafe_allow_html=True)
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
                        with st.container(border=True):
                            st.markdown(f'<div class="plot-container">', unsafe_allow_html=True)
                            wellbeing_fig = plot_worker_wellbeing(filtered_scores, filtered_triggers)
                            st.plotly_chart(wellbeing_fig, use_container_width=True, config={'displayModeBar': True, 'toImageButtonOptions': {'format': 'png'}})
                            st.markdown('</div>', unsafe_allow_html=True)
                    with col_well2:
                        st.markdown("### Psychological Safety")
                        filtered_safety = psychological_safety[time_indices[0]:time_indices[1]]
                        with st.container(border=True):
                            st.markdown(f'<div class="plot-container">', unsafe_allow_html=True)
                            safety_fig = plot_psychological_safety(filtered_safety)
                            st.plotly_chart(safety_fig, use_container_width=True, config={'displayModeBar': True, 'toImageButtonOptions': {'format': 'png'}})
                            st.markdown('</div>', unsafe_allow_html=True)
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
                with st.container(border=True):
                    st.markdown(f'<div class="plot-container">', unsafe_allow_html=True)
                    downtime_fig = plot_downtime_trend(filtered_downtime, DEFAULT_CONFIG['DOWNTIME_THRESHOLD'])
                    st.plotly_chart(downtime_fig, use_container_width=True, config={'displayModeBar': True, 'toImageButtonOptions': {'format': 'png'}})
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("Run a simulation or load data to view analysis.", icon="ℹ️")

    # Glossary Tab
    with tabs[4]:
        with st.container():
            st.header("Glossary", divider="grey")
            st.markdown("""
                ### Metric Definitions
                - <nanodisabled>Task Compliance Score</nanodisabled>: Percentage of tasks completed correctly and on time (0–100%). Measures adherence to operational protocols.
                - <nanodisabled>Collaboration Proximity Index</nanodisabled>: Percentage of workers within 5 meters of colleagues (0–100%). Indicates teamwork and communication opportunities.
                - <nanodisabled>Operational Recovery Score</nanodisabled>: Ability to maintain output after disruptions (0–100%). Reflects resilience to unexpected events.
                - <nanodisabled>Worker Well-Being Index</nanodisabled>: Composite score of fatigue, stress, and satisfaction (0–100%). Tracks worker health and morale.
                - <nanodisabled>Psychological Safety Score</nanodisabled>: Comfort level in reporting issues or suggesting improvements (0–100%). Indicates a supportive work environment.
                - <nanodisabled>Uptime</nanodisabled>: Percentage of time equipment is operational (0–100%). Measures equipment reliability.
                - <nanodisabled>Throughput</nanodisabled>: Percentage of maximum production rate achieved (0–100%). Indicates production efficiency.
                - <nanodisabled>Quality</nanodisabled>: Percentage of products meeting quality standards (0–100%). Reflects output consistency.
                - <nanodisabled>OEE (Overall Equipment Effectiveness)</nanodisabled>: Combined score of uptime, throughput, and quality (0–100%). Holistic measure of operational performance.
                - <nanodisabled>Productivity Loss</nanodisabled>: Percentage of potential output lost due to inefficiencies or disruptions (0–100%).
                - <nanodisabled>Downtime</nanodisabled>: Total minutes of unplanned operational stops. Tracks interruptions to workflow.
                - <nanodisabled>Task Completion Rate</nanodisabled>: Percentage of tasks completed per time interval (0–100%). Measures task efficiency over time.
            """)

if __name__ == "__main__":
    main()
