try:
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import sys
import logging
from industrial_workplace_simulation import (
    simulate_workplace_operations,
    plot_task_compliance_trend,
    plot_worker_collaboration_trend,
    plot_operational_resilience,
    plot_operational_efficiency,
    plot_worker_distribution,
    plot_worker_wellbeing,
    plot_psychological_safety
)

# Configure logging for operational diagnostics
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Default configuration (same as simulation)
DEFAULT_CONFIG = {
    'WORKSTATIONS': {
        'production_line': {'center': [50, 50], 'label': 'Production Line'},
        'assembly_zone': {'center': [150, 50], 'label': 'Assembly Zone'},
        'quality_control': {'center': [50, 150], 'label': 'Quality Control'},
        'logistics_hub': {'center': [150, 150], 'label': 'Logistics Hub'}
    },
    'COMPLIANCE_THRESHOLD': 0.7,
    'COMPLIANCE_TREND_WINDOW': 5,
    'DISRUPTION_IMPACT_WINDOW': 3,
    'BREAK_SCHEDULE_INTERVAL': 10,
    'WORKLOAD_LIMIT_STEPS': 5,
    'NUM_WORKERS': 20,
    'NUM_SHIFTS': 50,
    'FACILITY_SIZE': 200,
    'COMPLIANCE_ADJUSTMENT_RATE': 0.1,
    'SUPERVISOR_IMPACT_FACTOR': 0.2,
    'DISRUPTION_SHIFTS': [10, 30],
    'ANOMALY_THRESHOLD': 2.0  # Z-score threshold for anomalies
}

def validate_config(config):
    """Validate configuration parameters for the dashboard."""
    try:
        if config['NUM_WORKERS'] < 1 or config['NUM_SHIFTS'] < 1 or config['FACILITY_SIZE'] <= 0:
            raise ValueError("Invalid config: NUM_WORKERS, NUM_SHIFTS, and FACILITY_SIZE must be positive.")
        if not config['WORKSTATIONS'] or not all(
            "center" in ws and "label" in ws for ws in config['WORKSTATIONS'].values()
        ):
            raise ValueError("Invalid WORKSTATIONS: Must define center coordinates and labels.")
        for name, ws in config['WORKSTATIONS'].items():
            x, y = ws["center"]
            if not (0 <= x <= config['FACILITY_SIZE'] and 0 <= y <= config['FACILITY_SIZE']):
                raise ValueError(
                    f"Invalid center for {name}: Coordinates must be within [0, {config['FACILITY_SIZE']}]."
                )
    except ValueError as e:
        st.error(f"Configuration error: {str(e)}. Verify configuration settings.")
        st.stop()

def main():
    """Industrial Workplace Operations Dashboard."""
    # Warn about Python version compatibility
    if sys.version_info >= (3, 13):
        st.warning("Python 3.13 detected. Use Python 3.10 for optimal compatibility with Streamlit Cloud.")

    st.title("Industrial Workplace Operations Dashboard", anchor="dashboard-title")

    # Transparency statement
    st.info("""
    **Transparency Notice**: This dashboard uses aggregated, synthetic data to monitor workstation performance, worker well-being, and operational efficiency. Visualizations optimize facility layouts and schedules to enhance productivity and worker health, not to track individuals. Interventions like breaks and ergonomic adjustments improve team conditions. Worker feedback shapes a supportive workplace environment.
    """, icon="ℹ️")

    # Load configuration
    config = DEFAULT_CONFIG
    validate_config(config)

    # Sidebar controls
    st.sidebar.header("Operational Controls")
    show_forecast = st.sidebar.checkbox(
        "Show Predictive Trends", value=True, key="forecast-checkbox",
        help="Display forecasted trends for compliance and collaboration."
    )
    export_data = st.sidebar.button("Export Operational Data", key="export-button")

    # Collapsible worker feedback section
    with st.sidebar.expander("Worker Feedback & Initiatives"):
        worker_feedback = st.text_area(
            "Submit feedback to improve workplace conditions",
            placeholder="E.g., request more breaks, ergonomic equipment, or wellness programs...",
            key="feedback-input"
        )
        initiative = st.selectbox(
            "Worker Initiative Priority",
            ["More frequent breaks", "Task reduction", "Wellness programs", "Team recognition"],
            key="initiative-select",
            help="Select the primary worker-driven initiative for this shift."
        )

    # Generate operational data
    try:
        (
            worker_positions_df,
            compliance_variability,
            collaboration_index,
            operational_resilience,
            efficiency_metrics_df,
            productivity_loss,
            worker_wellbeing,
            safety_scores,
            worker_feedback_impact
        ) = simulate_workplace_operations(
            num_workers=config['NUM_WORKERS'],
            num_shifts=config['NUM_SHIFTS'],
            facility_size=config['FACILITY_SIZE'],
            compliance_adjustment_rate=config['COMPLIANCE_ADJUSTMENT_RATE'],
            supervisor_impact_factor=config['SUPERVISOR_IMPACT_FACTOR'],
            disruption_shifts=config['DISRUPTION_SHIFTS'],
            worker_initiative=initiative,
            config=config
        )
    except Exception as e:
        st.error(f"Data generation failed: {str(e)}.")
        st.error("Possible cause: Invalid data or configuration. Check logs or simulation settings.")
        try:
            (
                worker_positions_df,
                compliance_variability,
                collaboration_index,
                operational_resilience,
                efficiency_metrics_df,
                productivity_loss,
                worker_wellbeing,
                safety_scores,
                worker_feedback_impact
            ) = simulate_workplace_operations(
                num_workers=config['NUM_WORKERS'],
                num_shifts=config['NUM_SHIFTS'],
                facility_size=config['FACILITY_SIZE'],
                compliance_adjustment_rate=config['COMPLIANCE_ADJUSTMENT_RATE'],
                supervisor_impact_factor=config['SUPERVISOR_IMPACT_FACTOR'],
                disruption_shifts=config['DISRUPTION_SHIFTS'],
                worker_initiative=initiative,
                skip_forecast=True,
                config=config
            )
            show_forecast = False
            st.warning("Running in fallback mode without predictive trends due to data issues.")
        except Exception as e2:
            st.error(f"Fallback mode failed: {str(e2)}. Contact support or verify configuration.")
            st.stop()

    # Summary section
    st.subheader("Shift Performance Summary", anchor="summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Productivity Loss (Disruptions)",
            f"{np.mean([loss for i, loss in enumerate(productivity_loss) if i in config['DISRUPTION_SHIFTS']]):.1f}%",
            help="Average productivity loss during disruptions (e.g., equipment failures)."
        )
    with col2:
        st.metric(
            "Well-Being Improvement",
            f"{worker_feedback_impact['wellbeing']:.2%}",
            help="Improvement in worker well-being from feedback-driven initiatives."
        )
    with col3:
        st.metric(
            "Team Cohesion Improvement",
            f"{worker_feedback_impact['cohesion']:.2%}",
            help="Improvement in team collaboration from worker initiatives."
        )

    # Tabbed navigation for metrics
    tab1, tab2, tab3 = st.tabs(["Operational Performance", "Worker Health & Collaboration", "Facility Operations"])

    with tab1:
        st.subheader("Operational Performance Metrics")
        # Task Compliance
        with st.expander("Action: Improve Task Compliance"):
            st.markdown("""
            - **High variability**: Implement standardized operating procedures (SOPs) or additional training.
            - **Post-disruption dips**: Schedule supervisor oversight or process reviews.
            """)
        try:
            st.markdown('<div aria-label="Line plot of task compliance variability over shift intervals">', unsafe_allow_html=True)
            st.pyplot(plot_task_compliance_trend(
                compliance_variability['data'],
                config['DISRUPTION_SHIFTS'],
                compliance_variability['forecast'] if show_forecast else None
            ))
            st.caption("Line plot showing task compliance consistency over shift intervals. Lower values indicate uniform compliance across workers.")
            st.markdown('</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Failed to render task compliance plot: {str(e)}. Check simulation data.")

        # Operational Efficiency (OEE)
        with st.expander("Action: Enhance Operational Efficiency"):
            st.markdown("""
            - **Target**: Uptime >90%, Throughput >85%, Quality >97%.
            - **Low OEE**: Review equipment maintenance, worker training, or process bottlenecks.
            """)
        try:
            st.markdown('<div aria-label="Line plot of operational efficiency metrics (OEE) over shift intervals">', unsafe_allow_html=True)
            st.pyplot(plot_operational_efficiency(efficiency_metrics_df))
            st.caption("Line plot showing operational efficiency metrics (uptime, throughput, quality, OEE) over shift intervals.")
            st.markdown('</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Failed to render OEE plot: {str(e)}. Verify efficiency data.")

        # Operational Resilience
        with st.expander("Action: Strengthen Operational Resilience"):
            st.markdown("""
            - **Low resilience**: Provide resources (e.g., backup equipment) or cross-training.
            - **Post-disruption recovery**: Implement rapid response protocols.
            """)
        try:
            st.markdown('<div aria-label="Line plot of operational resilience over shift intervals">', unsafe_allow_html=True)
            st.pyplot(plot_operational_resilience(operational_resilience))
            st.caption("Line plot showing operational resilience over shift intervals, with 1 indicating full recovery from disruptions.")
            st.markdown('</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Failed to render resilience plot: {str(e)}. Check resilience data.")

    with tab2:
        st.subheader("Worker Health & Collaboration Metrics")
        col1, col2 = st.columns(2)
        # Worker Well-Being
        with col1:
            with st.expander("Action: Improve Worker Well-Being"):
                st.markdown("""
                - **Low scores**: Increase break frequency, provide ergonomic equipment, or offer wellness programs.
                - **Sustained declines**: Review workload or shift schedules.
                """)
            try:
                st.markdown('<div aria-label="Line plot of worker well-being over shift intervals">', unsafe_allow_html=True)
                st.pyplot(plot_worker_wellbeing(worker_wellbeing['scores']))
                st.caption("Line plot showing worker well-being trends over shift intervals, with 1 indicating optimal health and morale.")
                st.markdown('</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Failed to render well-being plot: {str(e)}. Verify well-being data.")

        # Psychological Safety
        with col2:
            with st.expander("Action: Enhance Psychological Safety"):
                st.markdown("""
                - **Low trust**: Encourage open communication through feedback sessions.
                - **Team recognition**: Implement reward programs to boost morale.
                """)
            try:
                st.markdown('<div aria-label="Line plot of psychological safety over shift intervals">', unsafe_allow_html=True)
                st.pyplot(plot_psychological_safety(safety_scores))
                st.caption("Line plot showing psychological safety trends over shift intervals, with 1 indicating high trust and communication.")
                st.markdown('</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Failed to render psychological safety plot: {str(e)}. Check safety data.")

        # Worker Collaboration
        with st.expander("Action: Boost Worker Collaboration"):
            st.markdown("""
            - **Low collaboration**: Organize team-building activities or cross-workstation training.
            - **High distances**: Optimize workstation layouts for better interaction.
            """)
        try:
            st.markdown('<div aria-label="Line plot of worker collaboration strength over shift intervals">', unsafe_allow_html=True)
            st.pyplot(plot_worker_collaboration_trend(
                collaboration_index['data'],
                collaboration_index['forecast'] if show_forecast else None
            ))
            st.caption("Line plot showing worker collaboration strength over shift intervals, with higher values indicating stronger teamwork.")
            st.markdown('</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Failed to render collaboration plot: {str(e)}. Verify collaboration data.")

    with tab3:
        st.subheader("Facility Operations Insights")
        # Worker Distribution
        with st.expander("Action: Optimize Facility Layout"):
            st.markdown("""
            - **High density areas**: Reconfigure workstations or reassign tasks to reduce congestion.
            - **Low activity zones**: Investigate underutilization or equipment issues.
            """)
        try:
            plot_mode = st.selectbox(
                "Visualization Mode", ["Hexbin + Scatter", "Hexbin Only"], key="plot-mode",
                help="Choose between combined hexbin and scatter plot or hexbin only."
            )
            use_scatter = plot_mode == "Hexbin + Scatter"
            fig = plot_worker_distribution(worker_positions_df, config['FACILITY_SIZE'], config=config, use_plotly=True)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Interactive plot showing worker distribution and density in the facility. Use the dropdown to filter by workstation, "
                "the slider to animate across shift intervals, and the reset zoom button to restore the view."
            )
        except Exception as e:
            st.warning(f"Interactive visualization failed: {str(e)}. Using Matplotlib fallback.")
            try:
                st.markdown('<div aria-label="Hexbin plot of worker distribution and density in facility">', unsafe_allow_html=True)
                st.pyplot(plot_worker_distribution(worker_positions_df, config['FACILITY_SIZE'], config=config, use_plotly=False))
                st.caption("Hexbin plot with scatter points showing worker distribution and density, colored by workstation with worker counts.")
                st.markdown('</div>', unsafe_allow_html=True)
            except Exception as e2:
                st.error(f"Failed to render distribution plot: {str(e2)}. Check worker position data.")

    # Worker well-being opportunities
    st.subheader("Worker Well-Being Opportunities", anchor="wellbeing-opportunities")
    triggers = worker_wellbeing['triggers']
    if triggers['threshold']:
        shifts = triggers['threshold']
        st.info(
            f"💡 Address low well-being in {len(shifts)} shift intervals (Shifts: {', '.join(map(str, shifts[:3]))}"
            f"{', ...' if len(shifts) > 3 else ''}). **Actions**: Increase break frequency, provide ergonomic equipment, or launch wellness programs."
        )
    if triggers['trend']:
        shifts = triggers['trend']
        st.info(
            f"💡 Address declining well-being trends in {len(shifts)} shift intervals (Shifts: {', '.join(map(str, shifts[:3]))}"
            f"{', ...' if len(shifts) > 3 else ''}). **Actions**: Reduce task loads, offer stress management workshops."
        )
    if triggers['workstation']:
        for ws, shifts in triggers['workstation'].items():
            st.info(
                f"💡 Support {config['WORKSTATIONS'][ws]['label']} in {len(shifts)} shift intervals (Shifts: {', '.join(map(str, shifts[:3]))}"
                f"{', ...' if len(shifts) > 3 else ''}). **Actions**: Assign additional supervisors, improve workstation ergonomics."
            )
    if triggers['disruption']:
        shifts = triggers['disruption']
        st.info(
            f"💡 Support recovery post-disruptions in {len(shifts)} shift intervals (Shifts: {', '.join(map(str, shifts[:3]))}"
            f"{', ...' if len(shifts) > 3 else ''}). **Actions**: Recognize team efforts, schedule additional breaks."
        )
    if not any(triggers.values()):
        st.success("Worker well-being is strong! Maintain with regular breaks, ergonomic support, and recognition programs.")

    # Performance anomalies
    anomalies = [
        (i, e, c) for i, (e, c) in enumerate(zip(compliance_variability['z_scores'], collaboration_index['z_scores']))
        if abs(e) > config['ANOMALY_THRESHOLD'] or abs(c) > config['ANOMALY_THRESHOLD']
    ]
    if anomalies:
        st.info(
            f"💡 Address performance or collaboration anomalies in {len(anomalies)} shift intervals. "
            "**Actions**: Provide targeted training, enhance team-building activities."
        )

    # Data export
    if export_data:
        if not worker_positions_df.empty:
            csv = worker_positions_df.to_csv(index=False)
            st.download_button(
                label="Download Worker Position Data",
                data=csv,
                file_name='worker_position_data.csv',
                mime='text/csv',
                key="download-button"
            )
        else:
            st.error("No worker position data available to export.")

    st.success("Operations dashboard loaded successfully. Use insights to optimize performance and worker well-being.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Dashboard initialization failed: {str(e)}. Ensure dependencies are installed and configuration is valid.")
        st.stop()
```
