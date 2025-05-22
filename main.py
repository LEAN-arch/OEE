"""
main.py
Streamlit dashboard for the Workplace Shift Monitoring Dashboard.
Provides a professional, interactive UI with actionable visualizations and clear metrics.
"""

import logging
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from config import DEFAULT_CONFIG, validate_config
from visualizations import (
    plot_gauge_chart,
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
        .stTabs [data-baseweb="tab"] { 
            color: #E6ECEF; 
            background-color: #2D3748; 
            border-radius: 8px; 
            padding: 8px 16px; 
        }
        .stTabs [data-baseweb="tab"]:hover { 
            background-color: #3B82F6; 
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
            - **Overview**: Gauge charts for OEE, Well-Being, Safety, and Downtime with recommendations.
            - **Efficiency**: Uptime, throughput, quality, and OEE trends.
            - **Team Distribution**: Worker positions (2D with layout or 3D with time slider) or density heatmap.
            - **Well-Being**: Worker Well-Being Index with actionable alerts.
            - **Safety**: Psychological Safety Score with training recommendations.
            - **Compliance**: Task Compliance Score with anomaly detection.
            - **Collaboration**: Collaboration Proximity Index showing teamwork.
            - **Resilience**: Operational Recovery vs. productivity loss.
            - **Downtime**: Downtime trends with investigation prompts.
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
    
    # Tabbed navigation
    tabs = st.tabs([
        "Overview", "Efficiency", "Team Distribution", "Well-Being", "Safety",
        "Compliance", "Collaboration", "Resilience", "Downtime", "Glossary"
    ])
    
    with tabs[0]:
        st.header("Dashboard Overview")
        st.markdown('<div class="tooltip">Key Metrics<span class="tooltiptext">Gauge charts for OEE, Worker Well-Being, Psychological Safety, and Downtime with actionable recommendations.</span></div>', unsafe_allow_html=True)
        
        # Composite line chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=minutes[time_indices[0]:time_indices[1]],
            y=efficiency_metrics_df['oee'][time_indices[0]:time_indices[1]],
            mode='lines',
            name='OEE',
            line=dict(color='#3B82F6', width=3),
            hovertemplate='Time: %{x} min<br>OEE: %{y:.1f}%'
        ))
        fig.add_trace(go.Scatter(
            x=minutes[time_indices[0]:time_indices[1]],
            y=worker_wellbeing['scores'][time_indices[0]:time_indices[1]],
            mode='lines',
            name='Well-Being',
            line=dict(color='#10B981', width=3),
            hovertemplate='Time: %{x} min<br>Well-Being: %{y:.1f}%'
        ))
        fig.add_trace(go.Scatter(
            x=minutes[time_indices[0]:time_indices[1]],
            y=psychological_safety[time_indices[0]:time_indices[1]],
            mode='lines',
            name='Psychological Safety',
            line=dict(color='#EC4899', width=3),
            hovertemplate='Time: %{x} min<br>Safety: %{y:.1f}%'
        ))
        
        annotations = []
        y_offset = 10
        wellbeing_slice = worker_wellbeing['scores'][time_indices[0]:time_indices[1]]
        if wellbeing_slice:
            if np.mean(wellbeing_slice) < DEFAULT_CONFIG['WELLBEING_THRESHOLD'] * 100:
                annotations.append(dict(
                    x=minutes[time_indices[0]], 
                    y=worker_wellbeing['scores'][time_indices[0]] + y_offset,
                    text="Low well-being<br>Recommend breaks",
                    showarrow=True, arrowhead=1, ax=20, ay=-30,
                    font=dict(color='#EF4444')
                ))
                y_offset += 5
        safety_slice = psychological_safety[time_indices[0]:time_indices[1]]
        if safety_slice:
            if np.mean(safety_slice) < DEFAULT_CONFIG['SAFETY_THRESHOLD'] * 100:
                annotations.append(dict(
                    x=minutes[time_indices[0]], 
                    y=psychological_safety[time_indices[0]] + y_offset,
                    text="Low safety<br>Enhance training",
                    showarrow=True, arrowhead=1, ax=20, ay=-30,
                    font=dict(color='#EF4444')
                ))
        
        fig.update_layout(
            title=dict(text='Key Performance Metrics Trend', x=0.5, font_size=22),
            xaxis_title='Time (minutes)',
            yaxis_title='Score (%)',
            font=dict(color='#E6ECEF', size=14),
            template='plotly_dark',
            hovermode='x unified',
            plot_bgcolor='#1A252F',
            paper_bgcolor='#1A252F',
            legend=dict(orientation='h', yanchor='top', y=1.1, xanchor='right', x=1),
            annotations=annotations[:2],
            xaxis=dict(tickangle=45, nticks=len(minutes[time_indices[0]:time_indices[1]])//10 + 1)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("Key Metrics Summary"):
            col1, col2, col3, col4 = st.columns(4)
            
            # OEE Gauge
            with col1:
                oee_mean = efficiency_metrics_df['oee'].mean()
                oee_fig, oee_rec = plot_gauge_chart(
                    value=oee_mean,
                    title="Average OEE",
                    threshold=75,
                    recommendation="Optimize processes to improve OEE."
                )
                st.plotly_chart(oee_fig, use_container_width=True)
                if oee_mean < 75:
                    st.markdown(f'<div class="recommendation">{oee_rec}</div>', unsafe_allow_html=True)
            
            # Well-Being Gauge
            with col2:
                wellbeing_mean = np.mean(worker_wellbeing['scores']) if worker_wellbeing['scores'] else 0
                wellbeing_fig, wellbeing_rec = plot_gauge_chart(
                    value=wellbeing_mean,
                    title="Average Well-Being",
                    threshold=DEFAULT_CONFIG['WELLBEING_THRESHOLD'] * 100,
                    recommendation="Schedule a break to improve well-being."
                )
                st.plotly_chart(wellbeing_fig, use_container_width=True)
                if wellbeing_mean < DEFAULT_CONFIG['WELLBEING_THRESHOLD'] * 100:
                    st.markdown(f'<div class="recommendation">{wellbeing_rec}</div>', unsafe_allow_html=True)
                    if st.button("Suggest Break Schedule", key="break_schedule"):
                        st.info("Suggested break schedule: Add 10-minute breaks every 60 minutes.")
            
            # Safety Gauge
            with col3:
                safety_mean = np.mean(psychological_safety) if psychological_safety else 0
                safety_fig, safety_rec = plot_gauge_chart(
                    value=safety_mean,
                    title="Average Safety",
                    threshold=DEFAULT_CONFIG['SAFETY_THRESHOLD'] * 100,
                    recommendation="Implement team-building exercises."
                )
                st.plotly_chart(safety_fig, use_container_width=True)
                if safety_mean < DEFAULT_CONFIG['SAFETY_THRESHOLD'] * 100:
                    st.markdown(f'<div class="recommendation">{safety_rec}</div>', unsafe_allow_html=True)
            
            # Downtime Gauge
            with col4:
                total_downtime = np.sum(downtime_minutes)
                downtime_fig, downtime_rec = plot_gauge_chart(
                    value=total_downtime,
                    title="Total Downtime",
                    threshold=DEFAULT_CONFIG['DOWNTIME_THRESHOLD'] * 5,  # Scale threshold for total
                    max_value=DEFAULT_CONFIG['DOWNTIME_THRESHOLD'] * 10,
                    recommendation="Investigate equipment or process issues."
                )
                downtime_fig.update_traces(number_suffix=" min")
                st.plotly_chart(downtime_fig, use_container_width=True)
                if total_downtime > DEFAULT_CONFIG['DOWNTIME_THRESHOLD'] * 5:
                    st.markdown(f'<div class="recommendation">{downtime_rec}</div>', unsafe_allow_html=True)
    
    with tabs[1]:
        st.header("Operational Efficiency")
        st.markdown('<div class="tooltip">Efficiency Metrics<span class="tooltiptext">Trends for uptime, throughput, quality, and OEE with rolling averages.</span></div>', unsafe_allow_html=True)
        selected_metrics = st.multiselect(
            "Select Metrics",
            options=['uptime', 'throughput', "quality", 'oee'],
            default=['uptime', 'throughput', "quality", 'oee'],
            key="efficiency_metrics"
        )
        filtered_df = efficiency_metrics_df.iloc[time_indices[0]:time_indices[1]]
        efficiency_fig = plot_operational_efficiency(filtered_df, selected_metrics)
        st.plotly_chart(efficiency_fig, use_container_width=True)
    
    with tabs[2]:
        st.header("Team Distribution")
        st.markdown('<div class="tooltip">Worker Positions<span class="tooltiptext">2D layout or 3D scatter with time slider showing worker locations in meters, color-coded by workload status.</span></div>', unsafe_allow_html=True)
        zone_filter = st.selectbox("Filter by Zone", options=["All"] + list(DEFAULT_CONFIG['WORK_AREAS'].keys()))
        filtered_df = team_positions_df if zone_filter == "All" else team_positions_df[team_positions_df['zone'] == zone_filter]
        filtered_df = filtered_df[(filtered_df['step'] >= time_indices[0]) & (filtered_df['step'] < time_indices[1])]
        
        if use_3d_distribution:
            selected_step = st.slider(
                "Select Time Step",
                min_value=int(time_indices[0]),
                max_value=int(time_indices[1] - 1),
                value=int(time_indices[0]),
                key="team_distribution_step"
            )
            distribution_fig = plot_worker_distribution(filtered_df, DEFAULT_CONFIG['FACILITY_SIZE'], DEFAULT_CONFIG, use_3d=True, selected_step=selected_step)
            st.plotly_chart(distribution_fig, use_container_width=True)
        else:
            if st.checkbox("Show Density Heatmap"):
                heatmap_fig = plot_worker_density_heatmap(filtered_df, DEFAULT_CONFIG['FACILITY_SIZE'], DEFAULT_CONFIG)
                st.plotly_chart(heatmap_fig, use_container_width=True)
            else:
                selected_step = st.slider(
                    "Select Time Step",
                    min_value=int(time_indices[0]),
                    max_value=int(time_indices[1] - 1),
                    value=int(time_indices[0]),
                    key="team_distribution_step_2d"
                )
                distribution_fig = plot_worker_distribution(filtered_df, DEFAULT_CONFIG['FACILITY_SIZE'], DEFAULT_CONFIG, use_3d=False, selected_step=selected_step)
                st.plotly_chart(distribution_fig, use_container_width=True)
    
    with tabs[3]:
        st.header("Worker Well-Being")
        st.markdown('<div class="tooltip">Well-Being Index<span class="tooltiptext">Worker Well-Being Index (0–100%) with alerts for low scores or trends.</span></div>', unsafe_allow_html=True)
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
    
    with tabs[4]:
        st.header("Psychological Safety")
        st.markdown('<div class="tooltip">Safety Score<span class="tooltiptext">Psychological Safety Score (0–100%) reflecting comfort in reporting issues.</span></div>', unsafe_allow_html=True)
        filtered_safety = psychological_safety[time_indices[0]:time_indices[1]]
        safety_fig = plot_psychological_safety(filtered_safety)
        st.plotly_chart(safety_fig, use_container_width=True)
    
    with tabs[5]:
        st.header("Task Compliance")
        st.markdown('<div class="tooltip">Compliance Score<span class="tooltiptext">Task Compliance Score (0–100%) with anomaly detection and recommendations.</span></div>', unsafe_allow_html=True)
        filtered_compliance = task_compliance['data'][time_indices[0]:time_indices[1]]
        filtered_z_scores = task_compliance['z_scores'][time_indices[0]:time_indices[1]]
        filtered_forecast = task_compliance['forecast'][time_indices[0]:time_indices[1]] if task_compliance['forecast'] is not None else None
        filtered_disruptions = [t for t in DEFAULT_CONFIG['DISRUPTION_INTERVALS'] if time_indices[0] <= t < time_indices[1]]
        compliance_fig = plot_task_compliance_score(filtered_compliance, filtered_disruptions, filtered_forecast, filtered_z_scores)
        st.plotly_chart(compliance_fig, use_container_width=True)
    
    with tabs[6]:
        st.header("Collaboration Proximity")
        st.markdown('<div class="tooltip">Proximity Index<span class="tooltiptext">Percentage of workers within 5 meters of colleagues, indicating teamwork.</span></div>', unsafe_allow_html=True)
        filtered_collab = collaboration_proximity['data'][time_indices[0]:time_indices[1]]
        filtered_forecast = collaboration_proximity['forecast'][time_indices[0]:time_indices[1]] if collaboration_proximity['forecast'] is not None else None
        filtered_disruptions = [t for t in DEFAULT_CONFIG['DISRUPTION_INTERVALS'] if time_indices[0] <= t < time_indices[1]]
        collaboration_fig = plot_collaboration_proximity_index(filtered_collab, filtered_disruptions, filtered_forecast)
        st.plotly_chart(collaboration_fig, use_container_width=True)
    
    with tabs[7]:
        st.header("Operational Resilience")
        st.markdown('<div class="tooltip">Recovery Score<span class="tooltiptext">Ability to maintain output post-disruption vs. productivity loss.</span></div>', unsafe_allow_html=True)
        filtered_recovery = operational_recovery[time_indices[0]:time_indices[1]]
        filtered_loss = productivity_loss[time_indices[0]:time_indices[1]]
        resilience_fig = plot_operational_recovery(filtered_recovery, filtered_loss)
        st.plotly_chart(resilience_fig, use_container_width=True)
    
    with tabs[8]:
        st.header("Downtime Analysis")
        st.markdown('<div class="tooltip">Downtime<span class="tooltiptext">Downtime in minutes with alerts for high values.</span></div>', unsafe_allow_html=True)
        filtered_downtime = downtime_minutes[time_indices[0]:time_indices[1]]
        downtime_fig = plot_downtime_trend(filtered_downtime, DEFAULT_CONFIG['DOWNTIME_THRESHOLD'])
        st.plotly_chart(downtime_fig, use_container_width=True)
    
    with tabs[9]:
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
            - **Productivity Loss**: Percentage of potential output
