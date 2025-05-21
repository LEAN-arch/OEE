try:
    import streamlit as st
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    from simulation import generate_synthetic_data, plot_compliance_variability, plot_team_clustering, plot_resilience, plot_oee, plot_worker_density, plot_wellbeing, plot_psychological_safety
    from config import NUM_OPERATORS, NUM_STEPS, FACTORY_SIZE, ADAPTATION_RATE, SUPERVISOR_INFLUENCE, DISRUPTION_STEPS, ANOMALY_THRESHOLD, WELLBEING_THRESHOLD, WELLBEING_TREND_LENGTH, WELLBEING_DISRUPTION_WINDOW, BREAK_INTERVAL, WORKLOAD_CAP_STEPS
except ImportError as e:
    st.error(f"Failed to import required libraries: {str(e)}. Ensure all dependencies are installed using 'uv pip install -r requirements.txt'.")
    st.error("If deploying on Streamlit Cloud, use Python 3.10 and verify requirements.txt.")
    st.stop()

def main():
    """Factory Operations Dashboard for Shift Monitoring."""
    # Warn about Python version
    import sys
    if sys.version_info >= (3, 13):
        st.warning("Python 3.13 detected. This version may not be fully supported. Use Python 3.10 for best results.")

    st.title("Factory Shift Monitoring Dashboard", anchor="dashboard-title")

    # Transparency statement
    st.info("""
    **Transparency Notice**: This dashboard uses aggregated, synthetic data to monitor zone-level performance and well-being, ensuring worker privacy. The activity and congestion visualization helps optimize factory layouts and task assignments to improve working conditions, not to track individuals. Actions like breaks and workload adjustments aim to enhance team health and collaboration. Worker suggestions are valued and simulated to shape a supportive workplace.
    """, icon="ℹ️")

    # Sidebar controls
    st.sidebar.header("Shift Controls")
    show_forecast = st.sidebar.checkbox("Show Predictive Trends", value=True, key="forecast-checkbox")
    export_data = st.sidebar.button("Export Shift Data", key="export-button")

    # Worker feedback and priorities
    st.sidebar.subheader("Worker Suggestions")
    worker_feedback = st.sidebar.text_area(
        "Share ideas to improve conditions (placeholder for real-world input)",
        placeholder="E.g., more breaks, ergonomic tools, training...",
        key="feedback-input"
    )
    st.sidebar.subheader("Worker Priorities")
    priority = st.sidebar.selectbox(
        "Simulated worker priority for this shift",
        ["More frequent breaks", "Task reduction", "Wellness resources", "Team recognition"],
        key="priority-select",
        help="Reflects team preferences for improving conditions."
    )

    # Generate synthetic data
    try:
        history_df, compliance_entropy, clustering_index, resilience_scores, oee_history, productivity_loss, wellbeing_data, safety_scores, feedback_impact = generate_synthetic_data(
            NUM_OPERATORS, NUM_STEPS, FACTORY_SIZE, ADAPTATION_RATE, SUPERVISOR_INFLUENCE, DISRUPTION_STEPS, priority
        )
        oee_df = pd.DataFrame(oee_history)
        wellbeing_scores = wellbeing_data['scores']
        wellbeing_triggers = wellbeing_data['triggers']
    except Exception as e:
        st.error(f"Failed to generate data: {str(e)}. Check logs or verify dependencies with 'uv pip install -r requirements.txt'.")
        st.error("If on Streamlit Cloud, ensure Python 3.10 is set and rebuild the app.")
        st.stop()

    # Feedback impact
    st.subheader("Worker Feedback Impact", anchor="feedback-impact")
    st.metric("Well-Being Boost from Suggestions", f"{feedback_impact['wellbeing']:.2%}")
    st.metric("Collaboration Boost from Suggestions", f"{feedback_impact['cohesion']:.2%}")
    st.caption("Shows how worker suggestions improve team health and teamwork.")

    # Well-being opportunities
    st.subheader("Team Well-Being Opportunities", anchor="wellbeing-opportunities")
    if wellbeing_triggers['threshold']:
        steps = wellbeing_triggers['threshold']
        st.info(f"💡 Opportunity: Support teams in {len(steps)} intervals with low well-being (Steps: {', '.join(map(str, steps[:3]))}{', ...' if len(steps) > 3 else ''}). "
                "Actions: Extend breaks by 15 minutes, provide ergonomic tools.")
    if wellbeing_triggers['trend']:
        steps = wellbeing_triggers['trend']
        st.info(f"💡 Opportunity: Address sustained well-being decline in {len(steps)} intervals (Steps: {', '.join(map(str, steps[:3]))}{', ...' if len(steps) > 3 else ''}). "
                "Actions: Reduce tasks by 10%, offer stress management workshops.")
    if wellbeing_triggers['zone']:
        for zone, steps in wellbeing_triggers['zone'].items():
            st.info(f"💡 Opportunity: Support {zone.capitalize()} zone in {len(steps)} intervals (Steps: {', '.join(map(str, steps[:3]))}{', ...' if len(steps) > 3 else ''}). "
                    f"Actions: Add supervisors, implement ergonomic adjustments.")
    if wellbeing_triggers['disruption']:
        steps = wellbeing_triggers['disruption']
        st.info(f"💡 Opportunity: Support teams near disruptions in {len(steps)} intervals (Steps: {', '.join(map(str, steps[:3]))}{', ...' if len(steps) > 3 else ''}). "
                "Actions: Recognize resilient teams, schedule post-disruption breaks.")
    if not any(wellbeing_triggers.values()):
        st.success("Team well-being is strong! Maintain support with regular breaks and recognition.")

    # Performance and collaboration opportunities
    anomalies = [(i, e, c) for i, (e, c) in enumerate(zip(compliance_entropy['z_scores'], clustering_index['z_scores']))
                 if abs(e) > ANOMALY_THRESHOLD or abs(c) > ANOMALY_THRESHOLD]
    if anomalies:
        st.info(f"💡 Opportunity: Enhance team performance or collaboration in {len(anomalies)} intervals. "
                "Actions: Offer training resources, foster team-building activities.")

    # Productivity loss summary
    st.metric("Average Productivity Loss During Disruptions", f"{np.mean([loss for i, loss in enumerate(productivity_loss) if i in DISRUPTION_STEPS]):.1f}%")

    # Plots
    st.subheader("Team SOP Compliance Consistency", anchor="compliance-plot")
    try:
        st.pyplot(plot_compliance_variability(compliance_entropy['data'], DISRUPTION_STEPS, compliance_entropy['forecast'] if show_forecast else None))
        st.caption("Inconsistent compliance may suggest opportunities for clearer procedures or additional training support.")
    except Exception as e:
        st.error(f"Failed to render compliance plot: {str(e)}. Ensure matplotlib is installed.")

    st.subheader("Team Collaboration Strength", anchor="cohesion-plot")
    try:
        st.pyplot(plot_team_clustering(clustering_index['data'], clustering_index['forecast'] if show_forecast else None))
        st.caption("Strong collaboration supports efficient teamwork. Foster with team-building or recognition programs.")
    except Exception as e:
        st.error(f"Failed to render collaboration plot: {str(e)}.")

    st.subheader("Team Recovery After Disruptions", anchor="resilience-plot")
    try:
        st.pyplot(plot_resilience(resilience_scores))
        st.caption("Strong recovery reflects team resilience. Support with resources or supervisor guidance.")
    except Exception as e:
        st.error(f"Failed to render resilience plot: {str(e)}.")

    st.subheader("Equipment Efficiency (OEE)", anchor="oee-plot")
    try:
        st.pyplot(plot_oee(oee_df))
        st.caption("High OEE indicates efficient operations. Target: Availability >90%, Performance >85%, Quality >97%. Support with maintenance or training.")
    except Exception as e:
        st.error(f"Failed to render OEE plot: {str(e)}.")

    st.subheader("Factory Floor Activity and Congestion", anchor="density-plot")
    try:
        st.pyplot(plot_worker_density(history_df, FACTORY_SIZE))
        st.caption("High congestion areas may need layout adjustments or task reassignments to reduce bottlenecks and improve workflow.")
    except Exception as e:
        st.error(f"Failed to render activity plot: {str(e)}.")

    st.subheader("Team Well-Being Trends", anchor="wellbeing-plot")
    try:
        st.pyplot(plot_wellbeing(wellbeing_scores))
        st.caption("Strong well-being supports team health. Enhance with regular breaks, ergonomic tools, or wellness programs.")
    except Exception as e:
        st.error(f"Failed to render well-being plot: {str(e)}.")

    st.subheader("Team Psychological Safety Trends", anchor="safety-plot")
    try:
        st.pyplot(plot_psychological_safety(safety_scores))
        st.caption("High psychological safety fosters trust and collaboration. Encourage open feedback and team recognition.")
    except Exception as e:
        st.error(f"Failed to render safety plot: {str(e)}.")

    # Data export
    if export_data:
        if not history_df.empty:
            csv = history_df.to_csv(index=False)
            st.download_button(
                label="Download Aggregated Shift Data",
                data=csv,
                file_name='factory_shift_data.csv',
                mime='text/csv',
                key="download-button"
            )
        else:
            st.error("No shift data available to export.")

    st.success("Shift dashboard loaded — use insights to create a healthier, more collaborative workplace.")

if __name__ == "__main__":
    main()
    st.success("Shift dashboard loaded — use insights to create a healthier, more supportive workplace.")

if __name__ == "__main__":
    main()
