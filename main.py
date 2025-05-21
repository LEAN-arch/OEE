import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from simulation import run_simulation, plot_compliance_variability, plot_team_clustering, plot_resilience, plot_oee, plot_worker_density
from config import NUM_OPERATORS, NUM_STEPS, FACTORY_SIZE, ADAPTATION_RATE, SUPERVISOR_INFLUENCE, DISRUPTION_STEPS, ANOMALY_THRESHOLD

def main():
    """Factory Operations Dashboard for Shift Monitoring."""
    st.title("Factory Shift Monitoring Dashboard")

    # Sidebar controls
    st.sidebar.header("Shift Controls")
    show_forecast = st.sidebar.checkbox("Show Predictive Trends", value=True)
    export_data = st.sidebar.button("Export Shift Data")

    # Run simulation
    try:
        history_df, compliance_entropy, clustering_index, resilience_scores, oee_history, productivity_loss = run_simulation(
            NUM_OPERATORS, NUM_STEPS, FACTORY_SIZE, ADAPTATION_RATE, SUPERVISOR_INFLUENCE, DISRUPTION_STEPS
        )
        oee_df = pd.DataFrame(oee_history)
    except Exception as e:
        st.error(f"Simulation failed: {str(e)}")
        return

    # Anomaly detection with actionable alerts
    anomalies = [(i, e, c) for i, (e, c) in enumerate(zip(compliance_entropy['z_scores'], clustering_index['z_scores']))
                 if abs(e) > ANOMALY_THRESHOLD or abs(c) > ANOMALY_THRESHOLD]
    if anomalies:
        st.error(f"⚠️ ALERT: Detected {len(anomalies)} anomalies in SOP compliance or team cohesion. "
                 "Recommended actions: Schedule SOP retraining and review supervisor assignments.")

    # Productivity loss summary
    avg_loss = np.mean([loss for i, loss in enumerate(productivity_loss) if i in DISRUPTION_STEPS])
    st.metric("Average Productivity Loss During Disruptions", f"{avg_loss:.1f}%", delta=-avg_loss)

    # Plots
    st.subheader("Operator SOP Compliance Variability")
    st.pyplot(plot_compliance_variability(compliance_entropy['data'], DISRUPTION_STEPS, compliance_entropy['forecast'] if show_forecast else None))
    st.caption("High variability indicates inconsistent SOP adherence. Consider targeted training.")

    st.subheader("Team Cohesion on Factory Floor")
    st.pyplot(plot_team_clustering(clustering_index['data'], clustering_index['forecast'] if show_forecast else None))
    st.caption("Lower clustering index indicates tighter team collaboration.")

    st.subheader("Production Resilience After Disruptions")
    st.pyplot(plot_resilience(resilience_scores))
    st.caption("Scores close to 1 indicate strong recovery post-disruption.")

    st.subheader("Overall Equipment Effectiveness (OEE)")
    st.pyplot(plot_oee(oee_df))
    st.caption("OEE reflects production efficiency. Target: Availability >90%, Performance >85%, Quality >97%.")

    st.subheader("Worker Density on Factory Floor")
    st.pyplot(plot_worker_density(history_df, FACTORY_SIZE))
    st.caption("High-density areas may indicate bottlenecks or congestion.")

    # Data export
    if export_data:
        if not history_df.empty:
            csv = history_df.to_csv(index=False)
            st.download_button(
                label="Download Shift Data",
                data=csv,
                file_name='factory_shift_data.csv',
                mime='text/csv'
            )
        else:
            st.error("No shift data available to export.")

    st.success("Shift dashboard loaded — use insights to optimize operator performance and production efficiency.")

if __name__ == "__main__":
    main()