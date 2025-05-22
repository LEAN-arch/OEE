"""
utils.py
Utility functions for saving/loading simulation data and generating reports.
"""

import pandas as pd
import numpy as np
import logging

# Configure logging
logger = logging.getLogger(__name__)

def save_simulation_data(team_positions_df, task_compliance, collaboration_proximity, operational_recovery,
                         efficiency_metrics_df, productivity_loss, worker_wellbeing, psychological_safety,
                         feedback_impact, downtime_minutes, task_completion_rate):
    """
    Save simulation results to CSV files.
    """
    try:
        team_positions_df.to_csv("team_positions.csv", index=False)
        efficiency_metrics_df.to_csv("efficiency_metrics.csv", index=False)
        
        task_compliance_df = pd.DataFrame({
            'task_compliance': task_compliance['data'],
            'z_scores': task_compliance['z_scores'],
            'forecast': task_compliance['forecast'] if task_compliance['forecast'] is not None else [None] * len(task_compliance['data'])
        })
        task_compliance_df.to_csv("task_compliance.csv", index=False)
        
        collaboration_proximity_df = pd.DataFrame({
            'collaboration_proximity': collaboration_proximity['data'],
            'forecast': collaboration_proximity['forecast'] if collaboration_proximity['forecast'] is not None else [None] * len(collaboration_proximity['data'])
        })
        collaboration_proximity_df.to_csv("collaboration_proximity.csv", index=False)
        
        worker_wellbeing_df = pd.DataFrame({'scores': worker_wellbeing['scores']})
        worker_wellbeing_df.to_csv("worker_wellbeing.csv", index=False)
        
        operational_recovery_df = pd.DataFrame({'operational_recovery': operational_recovery})
        operational_recovery_df.to_csv("operational_recovery.csv", index=False)
        
        productivity_loss_df = pd.DataFrame({'productivity_loss': productivity_loss})
        productivity_loss_df.to_csv("productivity_loss.csv", index=False)
        
        psychological_safety_df = pd.DataFrame({'psychological_safety': psychological_safety})
        psychological_safety_df.to_csv("psychological_safety.csv", index=False)
        
        downtime_minutes_df = pd.DataFrame({'downtime_minutes': downtime_minutes})
        downtime_minutes_df.to_csv("downtime_minutes.csv", index=False)
        
        task_completion_rate_df = pd.DataFrame({'task_completion_rate': task_completion_rate})
        task_completion_rate_df.to_csv("task_completion_rate.csv", index=False)
        
        feedback_impact_df = pd.DataFrame({'feedback_impact': [feedback_impact]})
        feedback_impact_df.to_csv("feedback_impact.csv", index=False)
        
        logger.info("Simulation data saved successfully.")
    except Exception as e:
        logger.error(f"Failed to save simulation data: {str(e)}")
        raise

def load_simulation_data():
    """
    Load simulation results from CSV files.
    """
    try:
        team_positions_df = pd.read_csv("team_positions.csv")
        task_compliance = {
            'data': pd.read_csv("task_compliance.csv")['task_compliance'].tolist(),
            'z_scores': pd.read_csv("task_compliance.csv")['z_scores'].tolist(),
            'forecast': pd.read_csv("task_compliance.csv")['forecast'].tolist()
        }
        collaboration_proximity = {
            'data': pd.read_csv("collaboration_proximity.csv")['collaboration_proximity'].tolist(),
            'forecast': pd.read_csv("collaboration_proximity.csv")['forecast'].tolist()
        }
        operational_recovery = pd.read_csv("operational_recovery.csv")['operational_recovery'].tolist()
        efficiency_metrics_df = pd.read_csv("efficiency_metrics.csv")
        productivity_loss = pd.read_csv("productivity_loss.csv")['productivity_loss'].tolist()
        worker_wellbeing = {
            'scores': pd.read_csv("worker_wellbeing.csv")['scores'].tolist(),
            'triggers': {}
        }
        psychological_safety = pd.read_csv("psychological_safety.csv")['psychological_safety'].tolist()
        feedback_impact = pd.read_csv("feedback_impact.csv")['feedback_impact'][0]
        downtime_minutes = pd.read_csv("downtime_minutes.csv")['downtime_minutes'].tolist()
        task_completion_rate = pd.read_csv("task_completion_rate.csv")['task_completion_rate'].tolist()
        
        logger.info("Simulation data loaded successfully.")
        return (team_positions_df, task_compliance, collaboration_proximity, operational_recovery,
                efficiency_metrics_df, productivity_loss, worker_wellbeing, psychological_safety,
                feedback_impact, downtime_minutes, task_completion_rate)
    except Exception as e:
        logger.error(f"Failed to load simulation data: {str(e)}")
        raise

def generate_pdf_report(summary_df):
    """
    Generate a LaTeX-based PDF report from simulation summary.
    """
    try:
        with open("workplace_report.tex", "w") as f:
            f.write("\\documentclass{article}\n")
            f.write("\\usepackage{booktabs}\n")
            f.write("\\begin{document}\n")
            f.write("\\title{Workplace Shift Monitoring Report}\n")
            f.write("\\author{Generated by Dashboard}\n")
            f.write("\\date{May 2025}\n")
            f.write("\\maketitle\n")
            f.write("\\section{Summary Statistics}\n")
            
            summary_stats = summary_df.describe().to_latex()
            f.write(summary_stats)
            
            f.write("\\end{document}\n")
        logger.info("PDF report generated successfully.")
    except Exception as e:
        logger.error(f"Failed to generate PDF report: {str(e)}")
        raise
