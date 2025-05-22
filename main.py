"""
main.py
Streamlit dashboard for the Workplace Shift Monitoring Dashboard.
Consolidated into a single file with sidebar navigation for improved display quality and no overlapping content.
"""

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

# Configure logging with more detailed context
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

# Custom CSS for enhanced UX and display quality
st.markdown("""
    <style>
        /* Base Styles */
        .main { 
            background-color: #1A252F; 
            color: #E6ECEF; 
            font-family: 'Arial', sans-serif;
            padding: 20px;
        }
        h1 { 
            color: #E6ECEF; 
            font-size: 2.5rem; 
            margin-bottom: 1.5rem; 
        }
        h2 { 
            color: #E6ECEF; 
            font-size: 1.8rem; 
            margin-top: 1.5rem; 
            margin-bottom: 1rem; 
        }
        .stButton>button { 
            background-color: #3B82F6; 
            color: #E6ECEF; 
            border-radius: 8px; 
            border: 2px solid #E6ECEF; 
            padding: 8px 16px; 
            transition: background-color 0.3s; 
        }
        .stButton>button:hover, .stButton>button:focus { 
            background-color: #EC4899; 
            border-color: #EC4899; 
            outline: none; 
        }
        .stSelectbox, .stSlider, .stMultiSelect { 
            background-color: #2D3748; 
            color: #E6ECEF; 
            border-radius: 8px; 
            padding: 5px; 
        }
        .tooltip {
            position: relative;
            display: inline-block;
            cursor: pointer;
        }
        .tooltip .tooltiptext {
            visibility: hidden;
            width: 250px;
            background-color: #2D3748;
            color: #E6ECEF;
            text-align: center;
            border-radius: 8px;
            padding: 10px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            margin-left: -125px;
            opacity: 0;
            transition: opacity 0.3s;
        }
        .tooltip:hover .tooltiptext, .tooltip:focus .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
        [data-testid="stSidebar"] { 
            background-color: #2D3748; 
            color: #E6ECEF; 
            padding: 10px;
        }
        [data-testid="stSidebar"] .stButton>button { 
            background-color: #10B981; 
            border-color: #E6ECEF; 
        }
        [data-testid="stSidebar"] .stButton>button:hover, [data-testid="stSidebar"] .stButton>button:focus { 
            background-color: #EC4899; 
            border-color: #EC4899; 
        }
        .stMetric { 
            background-color: #2D3748; 
            border-radius: 8px; 
            padding: 10px; 
            margin: 10px 0;
        }
        .stExpander { 
            background-color: #2D3748; 
            border-radius: 8px; 
            margin: 10px 0;
        }
        .recommendation {
            color: #FBBF24;
            font-size: 14px;
            margin-top: 5px;
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
            h1 { font-size: 2rem; }
            h2 { font-size: 1.5rem; }
        }
        /* Loading Spinner */
        .spinner {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100px;
        }
        .spinner::after {
            content: '';
            width: 40px;
            height: 40px;
            border: 4px solid #E6ECEF;
            border-top: 4px solid #3B82F6;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
""", unsafe_allow_html=True)

# Function to display a loading spinner
def display_loading(message):
    with st.container():
        st.markdown(f'<div class="spinner"></div><p style="text-align: center; color: #E6ECEF;">{message}</p>', unsafe_allow_html=True)

# Sidebar for settings and navigation
def render_settings_sidebar():
    with st.sidebar:
        st.header("Dashboard Settings")
        
        # Company logo with accessibility
        st.markdown(
            f'<img src="{LEAN_LOGO_BASE64}" width="150" alt="Lean 2.0 Institute Logo" aria-label="Lean 2.0 Institute Logo">',
            unsafe_allow_html=True
        )
        
        with st.expander("Simulation Controls", expanded=True):
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
                help="Times (minutes) when disruptions occur.",
                key="disruption_intervals"
            )
            
            team_initiative = st.selectbox(
                "Team Initiative",
                options=["More frequent breaks", "Team recognition"],
                index=0,
                help="Strategy to improve well-being and psychological safety.",
                key="team_initiative"
            )
            
            run_simulation = st.button(
                "Run Simulation", 
                key="run_simulation", 
                help="Start a new simulation with the selected parameters.",
                type="primary"
            )
        
        with st.expander("Visualization Settings"):
            high_contrast = st.checkbox(
                "High Contrast Mode", 
                help="Enable high-contrast colors for accessibility.",
                key="high_contrast"
            )
            use_3d_distribution = st.checkbox(
                "3D Team Distribution", 
                help="Use 3D scatter plot with time slider for team distribution.",
                key="use_3d_distribution"
            )
        
        with st.expander("Data Management"):
            load_data = st.button(
                "Load Saved Data", 
                key="load_data", 
                help="Load previously saved simulation data."
            )
            if st.button(
                "Download PDF Report", 
                key="download_report", 
                help="Generate and download a PDF report of the current data."
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
                    st.error(f"Failed to generate report: {str(e)}. See dashboard.log for details.")
        
        # Navigation
        st.header("Navigation")
        page = st.radio(
            "Go to Section",
            options=["Key Metrics Summary", "Operational Metrics", "Worker Distribution", "Worker Well-Being & Safety", "Downtime Analysis", "Glossary"],
            index=0,
            key="page_selection",
            help="Select a section to view."
        )
        
        # First-Time User Tour
        if st.button(
            "Take a Tour", 
            key="tour_button", 
            help="Take a guided tour of the dashboard features."
        ):
            st.session_state.show_tour = True
        
        # Help Section
        if st.button(
            "Help", 
            key="help_button", 
            help="View detailed help and documentation."
        ):
            st.markdown("""
                ### Help
                Navigate through different sections to monitor workplace performance:
                - **Key Metrics Summary**: Gauge charts for key metrics.
                - **Operational Metrics**: Trends for performance metrics.
                - **Worker Distribution**: Worker positions and density heatmap.
                - **Worker Well-Being & Safety**: Well-Being and Safety scores.
                - **Downtime Analysis**: Downtime trends.
                - **Glossary**: Definitions of all metrics and terms.
                
                Use the sidebar to adjust parameters, load saved data, or download a PDF report.
                Contact support@xai.com for assistance.
            """, unsafe_allow_html=True)

    return team_size, shift_duration, disruption_intervals, team_initiative, run_simulation, load_data, high_contrast, use_3d_distribution, page

# Function to handle simulation and data loading
@st.cache_data
def run_simulation_logic(team_size, shift_duration, disruption_intervals, team_initiative):
    config = DEFAULT_CONFIG.copy()
    config['TEAM_SIZE'] = team_size
    config['SHIFT_DURATION_MINUTES'] = shift_duration
    config['SHIFT_DURATION_INTERVALS'] = shift_duration // 2
    config['DISRUPTION_INTERVALS'] = [t // 2 for t in disruption_intervals]
    
    # Update WORK_AREAS worker counts to match new TEAM_SIZE
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
    if 'show_tour' not in st.session_state:
        st.session_state.show_tour = False

    # Render settings sidebar and get user inputs
    team_size, shift_duration, disruption_intervals, team_initiative, run_simulation, load_data, high_contrast, use_3d_distribution, page = render_settings_sidebar()

    # Precompute minutes for efficiency
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
                st.success("Simulation completed successfully! Select a section to view results.")
            except Exception as e:
                logger.error(f"Simulation failed: {str(e)}", extra={'user_action': 'Run Simulation'})
                st.error(f"Simulation failed: {str(e)}. Check dashboard.log for details.")

    if load_data:
        with st.spinner("Loading saved data..."):
            try:
                st.session_state.simulation_results = load_simulation_data()
                st.success("Loaded saved simulation data successfully!")
            except Exception as e:
                logger.error(f"Failed to load data: {str(e)}", extra={'user_action': 'Load Data'})
                st.error(f"Failed to load data: {str(e)}. Check dashboard.log for details.")

    # Display guided tour
    if st.session_state.show_tour:
        st.markdown("""
            <script>
                alert("Welcome to the Workplace Shift Monitoring Dashboard Tour!\\n\\n" +
                      "1. **Settings Sidebar**: Adjust simulation parameters here.\\n" +
                      "2. **Navigation**: Use the 'Go to Section' dropdown to switch between sections.\\n" +
                      "3. **Interactive Charts**: Hover over charts for details and use sliders to filter data.\\n\\n" +
                      "Click OK to start exploring!");
            </script>
        """, unsafe_allow_html=True)
        st.session_state.show_tour = False

    # Render selected page
    with st.container():
        if st.session_state.simulation_results is None:
            st.info("Run a simulation or load saved data using the sidebar controls to view results.")
        else:
            (team_positions_df, task_compliance, collaboration_proximity, operational_recovery,
             efficiency_metrics_df, productivity_loss, worker_wellbeing, psychological_safety,
             feedback_impact, downtime_minutes, task_completion_rate) = st.session_state.simulation_results

            if page == "Key Metrics Summary":
                st.header("Key Metrics Summary")
                st.markdown(
                    '<div class="tooltip">Overview<span class="tooltiptext">Key metrics for Task Compliance, Collaboration Proximity, Worker Well-Being, and Downtime with actionable recommendations.</span></div>',
                    unsafe_allow_html=True
                )
                compliance_mean = np.mean(task_compliance['data'])
                proximity_mean = np.mean(collaboration_proximity['data'])
                wellbeing_mean = np.mean(worker_wellbeing['scores']) if worker_wellbeing['scores'] else 0
                total_downtime = np.sum(downtime_minutes)
                col1, col2, col3, col4 = st.columns(4)
                summary_figs = plot_key_metrics_summary(compliance_mean, proximity_mean, wellbeing_mean, total_downtime)
                with col1:
                    st.plotly_chart(summary_figs[0], use_container_width=True)
                with col2:
                    st.plotly_chart(summary_figs[1], use_container_width=True)
                with col3:
                    st.plotly_chart(summary_figs[2], use_container_width=True)
                    if wellbeing_mean < DEFAULT_CONFIG['WELLBEING_THRESHOLD'] * 100:
                        if st.button("Suggest Break Schedule", key="break_schedule", help="Get a recommended break schedule to improve well-being."):
                            st.info("Suggested break schedule: Add 10-minute breaks every 60 minutes.")
                with col4:
                    st.plotly_chart(summary_figs[3], use_container_width=True)

            elif page == "Operational Metrics":
                st.header("Operational Metrics")
                st.markdown(
                    '<div class="tooltip">Performance Trends<span class="tooltiptext">Trends for task compliance, collaboration, operational recovery, and efficiency metrics.</span></div>',
                    unsafe_allow_html=True
                )
                time_range = st.slider(
                    "Select Time Range (minutes)",
                    min_value=0,
                    max_value=DEFAULT_CONFIG['SHIFT_DURATION_MINUTES'] - 2,
                    value=(0, DEFAULT_CONFIG['SHIFT_DURATION_MINUTES'] - 2),
                    step=2,
                    key="time_range_op",
                    help="Filter data by selecting a time range."
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
                filtered_recovery = operational_recovery[time_indices[0]:time_indices[1]]
                filtered_loss = productivity_loss[time_indices[0]:time_indices[1]]
                resilience_fig = plot_operational_recovery(filtered_recovery, filtered_loss)
                st.plotly_chart(resilience_fig, use_container_width=True)
                selected_metrics = st.multiselect(
                    "Select Efficiency Metrics",
                    options=['uptime', 'throughput', "quality", 'oee'],
                    default=['uptime', 'throughput', "quality", 'oee'],
                    key="efficiency_metrics_op",
                    help="Choose which efficiency metrics to display."
                )
                filtered_df = efficiency_metrics_df.iloc[time_indices[0]:time_indices[1]]
                efficiency_fig = plot_operational_efficiency(filtered_df, selected_metrics)
                st.plotly_chart(efficiency_fig, use_container_width=True)

            elif page == "Worker Distribution":
                st.header("Worker Distribution")
                st.markdown(
                    '<div class="tooltip">Worker Positions<span class="tooltiptext">2D layout or 3D scatter with time slider showing worker locations in meters, with entry/exit points and production lines.</span></div>',
                    unsafe_allow_html=True
                )
                time_range = st.slider(
                    "Select Time Range (minutes)",
                    min_value=0,
                    max_value=DEFAULT_CONFIG['SHIFT_DURATION_MINUTES'] - 2,
                    value=(0, DEFAULT_CONFIG['SHIFT_DURATION_MINUTES'] - 2),
                    step=2,
                    key="time_range_dist",
                    help="Filter data by selecting a time range."
                )
                time_indices = (time_range[0] // 2, time_range[1] // 2 + 1)
                zone_filter = st.selectbox(
                    "Filter by Zone", 
                    options=["All"] + list(DEFAULT_CONFIG['WORK_AREAS'].keys()),
                    help="Filter workers by specific zones."
                )
                filtered_df = team_positions_df if zone_filter == "All" else team_positions_df[team_positions_df['zone'] == zone_filter]
                filtered_df = filtered_df[(filtered_df['step'] >= time_indices[0]) & (filtered_df['step'] < time_indices[1])]
                show_entry_exit = st.checkbox(
                    "Show Entry/Exit Points", 
                    value=True, 
                    key="show_entry_exit_dist", 
                    help="Toggle visibility of entry and exit points."
                )
                show_production_lines = st.checkbox(
                    "Show Production Lines", 
                    value=True, 
                    key="show_production_lines_dist", 
                    help="Toggle visibility of production lines."
                )
                col_dist1, col_dist2 = st.columns(2)
                with col_dist1:
                    if use_3d_distribution:
                        selected_step = st.slider(
                            "Select Time Step (3D)",
                            min_value=int(time_indices[0]),
                            max_value=int(time_indices[1] - 1),
                            value=int(time_indices[0]),
                            key="team_distribution_step_3d",
                            help="Select a specific time step to view the 3D distribution."
                        )
                        distribution_fig_3d = plot_worker_distribution(
                            filtered_df, DEFAULT_CONFIG['FACILITY_SIZE'], DEFAULT_CONFIG, use_3d=True,
                            selected_step=selected_step, show_entry_exit=show_entry_exit, show_production_lines=show_production_lines
                        )
                        st.plotly_chart(distribution_fig_3d, use_container_width=True)
                    else:
                        selected_step = st.slider(
                            "Select Time Step (2D)",
                            min_value=int(time_indices[0]),
                            max_value=int(time_indices[1] - 1),
                            value=int(time_indices[0]),
                            key="team_distribution_step_2d",
                            help="Select a specific time step to view the 2D distribution."
                        )
                        distribution_fig_2d = plot_worker_distribution(
                            filtered_df, DEFAULT_CONFIG['FACILITY_SIZE'], DEFAULT_CONFIG, use_3d=False,
                            selected_step=selected_step, show_entry_exit=show_entry_exit, show_production_lines=show_production_lines
                        )
                        st.plotly_chart(distribution_fig_2d, use_container_width=True)
                with col_dist2:
                    heatmap_fig = plot_worker_density_heatmap(
                        filtered_df, DEFAULT_CONFIG['FACILITY_SIZE'], DEFAULT_CONFIG,
                        show_entry_exit=show_entry_exit, show_production_lines=show_production_lines
                    )
                    st.plotly_chart(heatmap_fig, use_container_width=True)

            elif page == "Worker Well-Being & Safety":
                st.header("Worker Well-Being & Safety")
                st.markdown(
                    '<div class="tooltip">Well-Being & Safety<span class="tooltiptext">Worker Well-Being Index and Psychological Safety Score with actionable alerts.</span></div>',
                    unsafe_allow_html=True
                )
                time_range = st.slider(
                    "Select Time Range (minutes)",
                    min_value=0,
                    max_value=DEFAULT_CONFIG['SHIFT_DURATION_MINUTES'] - 2,
                    value=(0, DEFAULT_CONFIG['SHIFT_DURATION_MINUTES'] - 2),
                    step=2,
                    key="time_range_well",
                    help="Filter data by selecting a time range."
                )
                time_indices = (time_range[0] // 2, time_range[1] // 2 + 1)
                col_well1, col_well2 = st.columns(2)
                with col_well1:
                    filtered_scores = worker_wellbeing['scores'][time_indices[0]:time_indices[1]]
                    filtered_triggers = {
                        'threshold': [t for t in worker_wellbeing['triggers']['threshold'] if time_indices[0] <= t < time_indices[1]],
                        'trend': [t for t in worker_wellbeing['triggers']['trend'] if time_indices[0] <= t < time_indices[1]],
                        'work_area': {k: [t for t in v if time_indices[0] <= t < time_indices[1]] for k, v in worker_wellbeing['triggers']['work_area'].items()},
                        'disruption': [t for t in worker_wellbeing['triggers']['disruption'] if time_indices[0] <= t < time_indices[1]]
                    }
                    wellbeing_fig = plot_worker_wellbeing(filtered_scores, filtered_triggers)
                    st.plotly_chart(wellbeing_fig, use_container_width=True)
                    with st.expander("Well-Being Triggers"):
                        st.write(f"**Threshold Alerts (Score < {DEFAULT_CONFIG['WELLBEING_THRESHOLD']*100}%):** {filtered_triggers['threshold']}")
                        st.write(f"**Trend Alerts (Declining):** {filtered_triggers['trend']}")
                        st.write("**Work Area Alerts:**")
                        for zone, triggers in filtered_triggers['work_area'].items():
                            st.write(f"{zone}: {triggers}")
                        st.write(f"**Disruption Alerts:** {filtered_triggers['disruption']}")
                with col_well2:
                    filtered_safety = psychological_safety[time_indices[0]:time_indices[1]]
                    safety_fig = plot_psychological_safety(filtered_safety)
                    st.plotly_chart(safety_fig, use_container_width=True)

            elif page == "Downtime Analysis":
                st.header("Downtime Analysis")
                st.markdown(
                    '<div class="tooltip">Downtime<span class="tooltiptext">Downtime in minutes with alerts for high values.</span></div>',
                    unsafe_allow_html=True
                )
                time_range = st.slider(
                    "Select Time Range (minutes)",
                    min_value=0,
                    max_value=DEFAULT_CONFIG['SHIFT_DURATION_MINUTES'] - 2,
                    value=(0, DEFAULT_CONFIG['SHIFT_DURATION_MINUTES'] - 2),
                    step=2,
                    key="time_range_down",
                    help="Filter data by selecting a time range."
                )
                time_indices = (time_range[0] // 2, time_range[1] // 2 + 1)
                filtered_downtime = downtime_minutes[time_indices[0]:time_indices[1]]
                downtime_fig = plot_downtime_trend(filtered_downtime, DEFAULT_CONFIG['DOWNTIME_THRESHOLD'])
                st.plotly_chart(downtime_fig, use_container_width=True)

            elif page == "Glossary":
                st.header("Glossary")
                st.markdown("""
                    ### Metric Definitions
                    - **Task Compliance Score**: Percentage of tasks completed correctly (0–100%).
                    - **Collaboration Proximity Index**: Percentage of workers within 5 meters of colleagues (0–100%).
                    - **Operational Recovery Score**: Ability to maintain production output after disruptions (0–100%).
                    - **Worker Well-Being Index**: Measure of worker fatigue, stress, and satisfaction (0–100%).
                    - **Psychological Safety Score**: Workers’ comfort in reporting issues (0–100%).
                    - **Uptime**: Percentage of time equipment is operational (0–100%).
                    - **Throughput**: Percentage of maximum production rate achieved (0–100%).
                    - **Quality**: Percentage of products meeting quality standards (0–100%).
                    - **OEE**: Combined metric of uptime, throughput, and quality (0–100%).
                    - **Productivity Loss**: Percentage of potential output lost due to disruptions (0–100%).
                    - **Downtime**: Total minutes of unplanned equipment or process stops.
                    - **Task Completion Rate**: Percentage of assigned tasks completed per interval (0–100%).
                    - **Feedback Impact**: Estimated improvement in well-being or team cohesion from initiatives.

                    ### Terms
                    - **Disruption**: An event (e.g., equipment failure) causing a temporary drop in performance.
                    - **Team Initiative**: Strategies like "More frequent breaks" or "Team recognition".
                    - **Anomaly**: A statistically significant deviation (z-score > 2.0) in metrics.
                """)

# High-contrast mode with WCAG compliance
def apply_high_contrast_mode():
    st.markdown("""
        <style>
            .main { 
                background-color: #000000; 
                color: #FFFFFF; 
            }
            h1, h2 { 
                color: #FFFFFF; 
            }
            .stButton>button { 
                background-color: #FFFFFF; 
                color: #000000; 
                border: 2px solid #FFFFFF; 
            }
            .stButton>button:hover, .stButton>button:focus { 
                background-color: #FFFF00; 
                color: #000000; 
            }
            .stSelectbox, .stSlider, .stMultiSelect { 
                background-color: #333333; 
                color: #FFFFFF; 
            }
            [data-testid="stSidebar"] { 
                background-color: #111111; 
                color: #FFFFFF; 
            }
            .stMetric, .stExpander { 
                background-color: #333333; 
            }
            .recommendation {
                color: #FFFF00;
            }
            .tooltip .tooltiptext {
                background-color: #333333;
                color: #FFFFFF;
            }
        </style>
    """, unsafe_allow_html=True)

# Developer Guide
def developer_guide():
    """
    ### Developer Guide
    This dashboard is built with Streamlit and Plotly for interactive visualizations. Key components:

    - **Structure**:
      - `main()`: Manages the main content with sidebar navigation.
      - `render_settings_sidebar()`: Handles sidebar controls and page selection.
      - `run_simulation_logic()`: Executes the simulation with caching.
      - `apply_high_contrast_mode()`: Applies WCAG-compliant high-contrast styling.

    - **Key Files**:
      - `visualizations.py`: Contains all Plotly chart functions.
      - `simulation.py`: Handles simulation logic.
      - `utils.py`: Utilities for saving/loading data and generating PDF reports.
      - `config.py`: Configuration settings.

    - **Extending the Dashboard**:
      1. Add new sections to the `page` radio options and corresponding `if` blocks in `main()`.
      2. Update `DEFAULT_CONFIG` in `config.py` for new parameters.
      3. Enhance accessibility with ARIA labels.
      4. Use the `logger` for debugging; logs are saved to `dashboard.log`.

    - **Performance Tips**:
      - Use `@st.cache_data` for expensive computations.
      - Minimize re-renders with session state.
      - Optimize Plotly charts with efficient layouts.

    - **Testing**:
      - Run simulations with various parameters.
      - Test high-contrast mode and responsiveness.
      - Check `dashboard.log` for errors (last error at 09:37 PM PDT; current time is 09:40 PM PDT).
    """

if __name__ == "__main__":
    main()
    if st.session_state.get('high_contrast', False):
        apply_high_contrast_mode()
    developer_guide()
