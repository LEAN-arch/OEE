try:
    import streamlit as st
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    from simulation import generate_synthetic_data, plot_compliance_variability, plot_team_clustering, plot_resilience, plot_oee, plot_worker_density, plot_wellbeing
    from config import NUM_OPERATORS, NUM_STEPS, FACTORY_SIZE, ADAPTATION_RATE, SUPERVISOR_INFLUENCE, DISRUPTION_STEPS, ANOMALY_THRESHOLD, WELLBEING_THRESHOLD
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

    st.title("Factory Shift Monitoring Dashboard")

    # Transparency statement
    st.info("""
    **Transparency Notice**: This dashboard uses aggregated, synthetic data to monitor team and zone performance, prioritizing worker privacy. Data is used to improve operations, training, and well-being, not to penalize individuals. For real-world use, workers are encouraged to provide feedback to ensure fair and supportive outcomes.
    """)

    # Sidebar controls
    st.sidebar.header("Shift Controls")
    show_forecast = st.sidebar.checkbox("Show Predictive Trends", value=True)
    export_data = st.sidebar.button("Export Shift Data")

    # Worker feedback placeholder
    st.sidebar.subheader("Worker Feedback")
    st.sidebar.text_area("Share concerns or suggestions (placeholder for real-world implementation)", placeholder="E.g., workload concerns, training needs...")

    # Generate synthetic data
    try:
        history_df, compliance_entropy, clustering_index, resilience_scores, oee_history, productivity_loss, wellbeing_scores = generate_synthetic_data(
            NUM_OPERATORS, NUM_STEPS, FACTORY_SIZE, ADAPTATION_RATE, SUPERVISOR_INFLUENCE, DISRUPTION_STEPS
        )
        oee_df = pd.DataFrame(oee_history)
    except Exception as e:
        st.error(f"Data generation failed: {str(e)}. Check logs or verify dependency installation with 'uv pip install -r requirements.txt'.")
        st.error("If on Streamlit Cloud, ensure Python 3.10 is set and rebuild the app.")
        st.stop()

    # Anomaly detection with supportive alerts
    anomalies = [(i, e, c) for i, (e, c) in enumerate(zip(compliance_entropy['z_scores'], clustering_index['z_scores']))
                 if abs(e) > ANOMALY_THRESHOLD or abs(c) > ANOMALY_THRESHOLD]
    if anomalies:
        st.warning(f"⚠️ NOTICE: Detected {len(anomalies)} variations in team performance or cohesion. "
                   "Suggestions: Provide additional training resources, adjust workloads, or recognize high-performing teams.")

    # Well-being alerts
    low_wellbeing_steps = [i for i, score in enumerate(wellbeing_scores) if score < WELLBEING_THRESHOLD]
    if low_wellbeing_steps:
        st.warning(f"⚠️ NOTICE: Low team well-being detected in {len(low_wellbeing_steps)} time intervals. "
                   "Suggestions: Increase break frequency, review workload distribution, or offer wellness support.")

    # Productivity loss summary
    avg_loss = np.mean([loss for i, loss in enumerate(productivity_loss) if i in DISRUPTION_STEPS])
    st.metric("Average Productivity Loss During Disruptions", f"{avg_loss:.1f}%", delta=-avg_loss)

    # Plots
    st.subheader("Team SOP Compliance Variability")
    try:
        st.pyplot(plot_compliance_variability(compliance_entropy['data'], DISRUPTION_STEPS, compliance_entropy['forecast'] if show_forecast else None))
        st.caption("High variability may indicate need for clearer SOPs or additional team support.")
    except Exception as e:
        st.error(f"Failed to render compliance plot: {str(e)}. Ensure matplotlib is installed correctly.")

    st.subheader("Team Cohesion Across Zones")
    try:
        st.pyplot(plot_team_clustering(clustering_index['data'], clustering_index['forecast'] if show_forecast else None))
        st.caption("Lower clustering suggests strong team collaboration. Consider team-building initiatives to maintain cohesion.")
    except Exception as e:
        st.error(f"Failed to render cohesion plot: {str(e)}.")

    st.subheader("Production Resilience After Disruptions")
    try:
        st.pyplot(plot_resilience(resilience_scores))
        st.caption("Scores close to 1 indicate strong team recovery. Support teams with resources to sustain resilience.")
    except Exception as e:
        st.error(f"Failed to render resilience plot: {str(e)}.")

    st.subheader("Overall Equipment Effectiveness (OEE)")
    try:
        st.pyplot(plot_oee(oee_df))
        st.caption("OEE reflects team efficiency. Target: Availability >90%, Performance >85%, Quality >97%. Support teams to maintain high OEE.")
    except Exception as e:
        st.error(f"Failed to render OEE plot: {str(e)}.")

    st.subheader("Team Density on Factory Floor")
    try:
        st.pyplot(plot_worker_density(history_df, FACTORY_SIZE))
        st.caption("High-density areas may indicate collaboration or bottlenecks. Adjust layouts or schedules as needed.")
    except Exception as e:
        st.error(f"Failed to render density plot: {str(e)}.")

    st.subheader("Team Well-Being Trends")
    try:
        st.pyplot(plot_wellbeing(wellbeing_scores))
        st.caption("Lower scores may indicate high workload or stress. Prioritize breaks and support to improve well-being.")
    except Exception as e:
        st.error(f"Failed to render well-being plot: {str(e)}.")

    # Data export
    if export_data:
        if not history_df.empty:
            csv = history_df.to_csv(index=False)
            st.download_button(
                label="Download Aggregated Shift Data",
                data=csv,
                file_name='factory_shift_data.csv',
                mime='text/csv'
            )
        else:
            st.error("No shift data available to export.")

    st.success("Shift dashboard loaded — use insights to support team performance, well-being, and operational efficiency.")

if __name__ == "__main__":
    main()
