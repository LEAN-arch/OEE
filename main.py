"""
main.py
Streamlit dashboard for the Workplace Shift Monitoring Dashboard.
Provides a professional, interactive UI with actionable visualizations and clear metrics.
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

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='dashboard.log'
)

# Streamlit page config
st.set_page_config(
    page_title="Workplace Shift Monitoring Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme and improved styling
st.markdown("""
    <style>
        .main { 
            background-color: #1A252F; 
            color: #E6ECEF; 
        }
        .stButton>button { 
            background-color: #3B82F6; 
            color: #E6ECEF; 
            border-radius: 8px; 
            border: 2px solid #E6ECEF; 
            padding: 8px 16px; 
            transition: background-color 0.3s; 
        }
        .stButton>button:hover { 
            background-color: #EC4899; 
            border-color: #EC4899; 
        }
        .stSelectbox, .stSlider, .stMultiSelect { 
            background-color: #2D3748; 
            color: #E6ECEF; 
            border-radius: 8px; 
            padding: 5px; 
        }
        h1, h2, h3 { 
            color: #E6ECEF; 
            font-weight: 700; 
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
        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
        [data-testid="stSidebar"] { 
            background-color: #2D3748; 
            color: #E6ECEF; 
        }
        [data-testid="stSidebar"] .stButton>button { 
            background-color: #10B981; 
            border-color: #E6ECEF; 
        }
        [data-testid="stSidebar"] .stButton>button:hover { 
            background-color: #EC4899; 
            border-color: #EC4899; 
        }
        .stMetric { 
            background-color: #2D3748; 
            border-radius: 8px; 
            padding: 10px; 
        }
        .stExpander { 
            background-color: #2D3748; 
            border-radius: 8px; 
        }
        .recommendation {
            color: #FBBF24;
            font-size: 14px;
            margin-top: 5px;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("Workplace Shift Monitoring Dashboard")
    
    # Company logo
    st.markdown(
        f'<img src="{LEAN_LOGO_BASE64}" width="150" alt="Lean 2.0 Institute Logo">',
        unsafe_allow_html=True
    )
    
    with st.expander("Simulation Controls", expanded=True):
        team_size = st.slider(
            "Team Size",
            min_value=10, max_value=100, value=DEFAULT_CONFIG['TEAM_SIZE'],
            help="Number of workers in the simulation."
        )
        
        shift_duration = st.slider(
            "Shift Duration (minutes)",
            min_value=200, max_value=2000, value=DEFAULT_CONFIG['SHIFT_DURATION_MINUTES'], step=2,
            help="Shift duration in minutes (2-minute intervals)."
        )
        
        disruption_intervals = st.multiselect(
            "Disruption Times (minutes)",
            options=[i * 2 for i in range(shift_duration // 2)],
            default=[i * 2 for i in DEFAULT_CONFIG['DISRUPTION_INTERVALS']],
            help="Times (minutes) when disruptions occur."
        )
        
        team_initiative = st.selectbox(
            "Team Initiative",
            options=["More frequent breaks", "Team recognition"],
            index=0,
            help="Strategy to improve well-being and psychological safety."
        )
        
        run_simulation = st.button("Run Simulation", key="run_simulation")
    
    with st.expander("Visualization Settings"):
        high_contrast = st.checkbox("High Contrast Mode", help="Enable high-contrast colors for accessibility.")
        use_3d_distribution = st.checkbox("3D Team Distribution", help="Use 3D scatter plot with time slider for team distribution.")
    
    with st.expander("Data Management"):
        load_data = st.button("Load Saved Data", key="load_data")
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
                st.success("PDF report generated as 'workplace_report.tex'. Compile with LaTeX to view.")
            except Exception as e:
                logger.error(f"Failed to generate report: {str(e)}")
                st.error(f"Failed to generate report: {str(e)}")
    
    if st.button("Help", key="help_button"):
        st.markdown("""
            ### Help
            Monitor workplace performance with professional visualizations:
            - **Key Metrics Summary**: Gauge charts for Task Compliance, Collaboration Proximity, Worker Well-Being, and Downtime.
            - **Operational Metrics**: Trends for task compliance, collaboration, operational recovery, and efficiency.
            - **Worker Distribution**: 2D/3D worker positions and density heatmap.
            - **Worker Well-Being & Safety**: Well-Being Index and Psychological Safety Score.
            - **Downtime Analysis**: Downtime trends with investigation prompts.
            - **Glossary**: Definitions of all metrics and terms.
            
            Use the sidebar to adjust parameters, load saved data, or download a PDF report.
            Contact support@xai.com for assistance.
        """, unsafe_allow_html=True)

# Main content
st.title("Workplace Shift Monitoring Dashboard")

# Initialize session state
if 'simulation_results' not in st.session_state:
    st.session_state.simulation_results = None

# Precompute minutes for efficiency
if st.session_state.simulation_results:
    num_steps = len(st.session_state.simulation_results[0]['step'].unique())
    minutes = [i * 2 for i in range(num_steps)]
else:
    minutes = [i * 2 for i in range(DEFAULT_CONFIG['SHIFT_DURATION_INTERVALS'])]

# Run simulation
if run_simulation:
    try:
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
            # Adjust for rounding errors
            current_sum = sum(zone['workers'] for zone in config['WORK_AREAS'].values())
            if current_sum != team_size:
                diff = team_size - current_sum
                config['WORK_AREAS']['Assembly Line']['workers'] += diff
        
        validate_config(config)
        logger.info("Running simulation with team_size=%d, shift_duration=%d min", team_size, shift_duration)
        simulation_results = simulate_workplace_operations(
            num_team_members=team_size,
            num_steps=shift_duration // 2,
            disruption_intervals=[t // 2 for t in disruption_intervals],
            team_initiative=team_initiative,
            config=config
        )
        
        st.session_state.simulation_results = simulation_results
        save_simulation_data(*simulation_results)
        st.success("Simulation completed successfully!")
    except Exception as e:
        logger.error(f"Simulation failed: {str(e)}")
        st.error(f"Simulation failed: {str(e)}. Check dashboard.log.")

# Load saved data
if load_data:
    try:
        st.session_state.simulation_results = load_simulation_data()
        st.success("Loaded saved simulation data!")
    except Exception as e:
        logger.error(f"Failed to load data: {str(e)}")
        st.error(f"Failed to load data: {str(e)}. Check dashboard.log.")

# Display results
if st.session_state.simulation_results:
    (team_positions_df, task_compliance, collaboration_proximity, operational_recovery,
     efficiency_metrics_df, productivity_loss, worker_wellbeing, psychological_safety,
     feedback_impact, downtime_minutes, task_completion_rate) = st.session_state.simulation_results
    
    # Time range slider for overall trends
    time_range = st.slider(
        "Select Time Range (minutes)",
        min_value=0,
        max_value=DEFAULT_CONFIG['SHIFT_DURATION_MINUTES'] - 2,
        value=(0, DEFAULT_CONFIG['SHIFT_DURATION_MINUTES'] - 2),
        step=2,
        key="time_range"
    )
    time_indices = (time_range[0] // 2, time_range[1] // 2 + 1)

    # Section 1: Key Metrics Summary (Permanent, at the top)
    st.header("Key Metrics Summary")
    st.markdown('<div class="tooltip">Overview<span class="tooltiptext">Key metrics for Task Compliance, Collaboration Proximity, Worker Well-Being, and Downtime with actionable recommendations.</span></div>', unsafe_allow_html=True)
    
    # Calculate averages for gauges
    compliance_mean = np.mean(task_compliance['data'])
    proximity_mean = np.mean(collaboration_proximity['data'])
    wellbeing_mean = np.mean(worker_wellbeing['scores']) if worker_wellbeing['scores'] else 0
    total_downtime = np.sum(downtime_minutes)
    
    # Display 2x2 grid of gauges
    col1, col2, col3, col4 = st.columns(4)
    summary_figs = plot_key_metrics_summary(compliance_mean, proximity_mean, wellbeing_mean, total_downtime)
    
    with col1:
        st.plotly_chart(summary_figs[0], use_container_width=True)
    with col2:
        st.plotly_chart(summary_figs[1], use_container_width=True)
    with col3:
        st.plotly_chart(summary_figs[2], use_container_width=True)
        if wellbeing_mean < DEFAULT_CONFIG['WELLBEING_THRESHOLD'] * 100:
            if st.button("Suggest Break Schedule", key="break_schedule"):
                st.info("Suggested break schedule: Add 10-minute breaks every 60 minutes.")
    with col4:
        st.plotly_chart(summary_figs[3], use_container_width=True)

    # Section 2: Operational Metrics
    st.header("Operational Metrics")
    st.markdown('<div class="tooltip">Performance Trends<span class="tooltiptext">Trends for task compliance, collaboration, operational recovery, and efficiency metrics.</span></div>', unsafe_allow_html=True)
    
    # Task Compliance
    filtered_compliance = task_compliance['data'][time_indices[0]:time_indices[1]]
    filtered_z_scores = task_compliance['z_scores'][time_indices[0]:time_indices[1]]
    filtered_forecast = task_compliance['forecast'][time_indices[0]:time_indices[1]] if task_compliance['forecast'] is not None else None
    filtered_disruptions = [t for t in DEFAULT_CONFIG['DISRUPTION_INTERVALS'] if time_indices[0] <= t < time_indices[1]]
    compliance_fig = plot_task_compliance_score(filtered_compliance, filtered_disruptions, filtered_forecast, filtered_z_scores)
    st.plotly_chart(compliance_fig, use_container_width=True)

    # Collaboration Proximity
    filtered_collab = collaboration_proximity['data'][time_indices[0]:time_indices[1]]
    filtered_forecast = collaboration_proximity['forecast'][time_indices[0]:time_indices[1]] if collaboration_proximity['forecast'] is not None else None
    collaboration_fig = plot_collaboration_proximity_index(filtered_collab, filtered_disruptions, filtered_forecast)
    st.plotly_chart(collaboration_fig, use_container_width=True)

    # Operational Recovery
    filtered_recovery = operational_recovery[time_indices[0]:time_indices[1]]
    filtered_loss = productivity_loss[time_indices[0]:time_indices[1]]
    resilience_fig = plot_operational_recovery(filtered_recovery, filtered_loss)
    st.plotly_chart(resilience_fig, use_container_width=True)

    # Operational Efficiency
    selected_metrics = st.multiselect(
        "Select Efficiency Metrics",
        options=['uptime', 'throughput', "quality", 'oee'],
        default=['uptime', 'throughput', "quality", 'oee'],
        key="efficiency_metrics"
    )
    filtered_df = efficiency_metrics_df.iloc[time_indices[0]:time_indices[1]]
    efficiency_fig = plot_operational_efficiency(filtered_df, selected_metrics)
    st.plotly_chart(efficiency_fig, use_container_width=True)

    # Section 3: Worker Distribution
    st.header("Worker Distribution")
    st.markdown('<div class="tooltip">Worker Positions<span class="tooltiptext">2D layout or 3D scatter with time slider showing worker locations in meters, color-coded by workload status, with entry/exit points and production lines.</span></div>', unsafe_allow_html=True)
    
    zone_filter = st.selectbox("Filter by Zone", options=["All"] + list(DEFAULT_CONFIG['WORK_AREAS'].keys()))
    filtered_df = team_positions_df if zone_filter == "All" else team_positions_df[team_positions_df['zone'] == zone_filter]
    filtered_df = filtered_df[(filtered_df['step'] >= time_indices[0]) & (filtered_df['step'] < time_indices[1])]
    
    show_entry_exit = st.checkbox("Show Entry/Exit Points", value=True, key="show_entry_exit")
    show_production_lines = st.checkbox("Show Production Lines", value=True, key="show_production_lines")
    
    col_dist1, col_dist2 = st.columns(2)
    with col_dist1:
        if use_3d_distribution:
            selected_step = st.slider(
                "Select Time Step (3D)",
                min_value=int(time_indices[0]),
                max_value=int(time_indices[1] - 1),
                value=int(time_indices[0]),
                key="team_distribution_step_3d"
            )
            distribution_fig_3d = plot_worker_distribution(
                filtered_df, 
                DEFAULT_CONFIG['FACILITY_SIZE'], 
                DEFAULT_CONFIG, 
                use_3d=True, 
                selected_step=selected_step,
                show_entry_exit=show_entry_exit,
                show_production_lines=show_production_lines
            )
            st.plotly_chart(distribution_fig_3d, use_container_width=True)
        else:
            selected_step = st.slider(
                "Select Time Step (2D)",
                min_value=int(time_indices[0]),
                max_value=int(time_indices[1] - 1),
                value=int(time_indices[0]),
                key="team_distribution_step_2d"
            )
            distribution_fig_2d = plot_worker_distribution(
                filtered_df, 
                DEFAULT_CONFIG['FACILITY_SIZE'], 
                DEFAULT_CONFIG, 
                use_3d=False, 
                selected_step=selected_step,
                show_entry_exit=show_entry_exit,
                show_production_lines=show_production_lines
            )
            st.plotly_chart(distribution_fig_2d, use_container_width=True)
    
    with col_dist2:
        heatmap_fig = plot_worker_density_heatmap(
            filtered_df, 
            DEFAULT_CONFIG['FACILITY_SIZE'], 
            DEFAULT_CONFIG,
            show_entry_exit=show_entry_exit,
            show_production_lines=show_production_lines
        )
        st.plotly_chart(heatmap_fig, use_container_width=True)

    # Section 4: Worker Well-Being and Safety
    st.header("Worker Well-Being and Safety")
    st.markdown('<div class="tooltip">Well-Being & Safety<span class="tooltiptext">Worker Well-Being Index and Psychological Safety Score with actionable alerts.</span></div>', unsafe_allow_html=True)
    
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

    # Section 5: Downtime Analysis
    st.header("Downtime Analysis")
    st.markdown('<div class="tooltip">Downtime<span class="tooltiptext">Downtime in minutes with alerts for high values.</span></div>', unsafe_allow_html=True)
    filtered_downtime = downtime_minutes[time_indices[0]:time_indices[1]]
    downtime_fig = plot_downtime_trend(filtered_downtime, DEFAULT_CONFIG['DOWNTIME_THRESHOLD'])
    st.plotly_chart(downtime_fig, use_container_width=True)

    # Section 6: Glossary
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

else:
    st.info("Run a simulation or load saved data using the sidebar controls to view results.")

# High-contrast mode
if high_contrast:
    st.markdown("""
        <style>
            .main { 
                background-color: #000000; 
                color: #FFFFFF; 
            }
            h1, h2, h3 { 
                color: #FFFFFF; 
            }
            .stButton>button { 
                background-color: #FFFFFF; 
                color: #000000; 
                border: 2px solid #FFFFFF; 
            }
            .stButton>button:hover { 
                background-color: #19D3F3; 
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
                color: #19D3F3;
            }
        </style>
    """, unsafe_allow_html=True)
