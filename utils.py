"""
utils.py
Utility functions for saving/loading simulation data and generating reports.
"""

import pandas as pd
import numpy as np
import logging
import os

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s - [User Action: %(user_action)s]',
    filename='dashboard.log'
)

def sanitize_latex(text):
    """
    Escape special characters for LaTeX compatibility.

    Args:
        text (str): Input text to sanitize.

    Returns:
        str: Sanitized text.
    """
    if not isinstance(text, str):
        text = str(text)
    replacements = {
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}',
        '\\': r'\textbackslash{}'
    }
    for char, escape in replacements.items():
        text = text.replace(char, escape)
    return text

def save_simulation_data(team_positions_df, task_compliance, collaboration_proximity, operational_recovery,
                         efficiency_metrics_df, productivity_loss, worker_wellbeing, psychological_safety,
                         feedback_impact, downtime_minutes, task_completion_rate):
    """
    Save simulation results to CSV files.

    Args:
        team_positions_df (pd.DataFrame): Worker positions and workload status.
        task_compliance (dict): Task compliance data, z-scores, and forecast.
        collaboration_proximity (dict): Collaboration proximity data and forecast.
        operational_recovery (list): Operational recovery scores.
        efficiency_metrics_df (pd.DataFrame): Efficiency metrics (uptime, throughput, quality, oee).
        productivity_loss (list): Productivity loss percentages.
        worker_wellbeing (dict): Worker well-being scores and triggers.
        psychological_safety (list): Psychological safety scores.
        feedback_impact (float): Feedback impact score.
        downtime_minutes (list): Downtime values in minutes.
        task_completion_rate (list): Task completion rates.
    """
    try:
        # Handle NaNs
        team_positions_df = team_positions_df.fillna(0.0)
        efficiency_metrics_df = efficiency_metrics_df.fillna(0.0)
        
        # Save DataFrames
        team_positions_df.to_csv("team_positions.csv", index=False)
        efficiency_metrics_df.to_csv("efficiency_metrics.csv", index=False)
        
        # Save task compliance
        task_compliance_df = pd.DataFrame({
            'task_compliance': np.nan_to_num(task_compliance['data'], nan=0.0),
            'z_scores': np.nan_to_num(task_compliance['z_scores'], nan=0.0),
            'forecast': task_compliance['forecast'] if task_compliance['forecast'] is not None else [0.0] * len(task_compliance['data'])
        })
        task_compliance_df.to_csv("task_compliance.csv", index=False)
        
        # Save collaboration proximity
        collaboration_proximity_df = pd.DataFrame({
            'collaboration_proximity': np.nan_to_num(collaboration_proximity['data'], nan=0.0),
            'forecast': collaboration_proximity['forecast'] if collaboration_proximity['forecast'] is not None else [0.0] * len(collaboration_proximity['data'])
        })
        collaboration_proximity_df.to_csv("collaboration_proximity.csv", index=False)
        
        # Save worker well-being
        worker_wellbeing_df = pd.DataFrame({'scores': np.nan_to_num(worker_wellbeing['scores'], nan=0.0)})
        worker_wellbeing_df.to_csv("worker_wellbeing.csv", index=False)
        # Save triggers separately
        triggers_df = pd.DataFrame({
            'threshold': pd.Series(worker_wellbeing['triggers']['threshold']),
            'trend': pd.Series(worker_wellbeing['triggers']['trend']),
            'disruption': pd.Series(worker_wellbeing['triggers']['disruption'])
        })
        for zone in worker_wellbeing['triggers']['work_area']:
            triggers_df[f'work_area_{zone}'] = pd.Series(worker_wellbeing['triggers']['work_area'][zone])
        triggers_df.to_csv("worker_wellbeing_triggers.csv", index=False)
        
        # Save other metrics
        operational_recovery_df = pd.DataFrame({'operational_recovery': np.nan_to_num(operational_recovery, nan=0.0)})
        operational_recovery_df.to_csv("operational_recovery.csv", index=False)
        
        productivity_loss_df = pd.DataFrame({'productivity_loss': np.nan_to_num(productivity_loss, nan=0.0)})
        productivity_loss_df.to_csv("productivity_loss.csv", index=False)
        
        psychological_safety_df = pd.DataFrame({'psychological_safety': np.nan_to_num(psychological_safety, nan=0.0)})
        psychological_safety_df.to_csv("psychological_safety.csv", index=False)
        
        downtime_minutes_df = pd.DataFrame({'downtime_minutes': np.nan_to_num(downtime_minutes, nan=0.0)})
        downtime_minutes_df.to_csv("downtime_minutes.csv", index=False)
        
        task_completion_rate_df = pd.DataFrame({'task_completion_rate': np.nan_to_num(task_completion_rate, nan=0.0)})
        task_completion_rate_df.to_csv("task_completion_rate.csv", index=False)
        
        feedback_impact_df = pd.DataFrame({'feedback_impact': [feedback_impact]})
        feedback_impact_df.to_csv("feedback_impact.csv", index=False)
        
        logger.info("Simulation data saved successfully", extra={'user_action': 'Save Simulation Data'})
    except Exception as e:
        logger.error(f"Failed to save simulation data: {str(e)}", extra={'user_action': 'Save Simulation Data'})
        raise

def load_simulation_data():
    """
    Load simulation results from CSV files.

    Returns:
        tuple: Simulation results including worker positions with workload status.
    """
    try:
        # Check if files exist
        required_files = [
            "team_positions.csv", "task_compliance.csv", "collaboration_proximity.csv",
            "operational_recovery.csv", "efficiency_metrics.csv", "productivity_loss.csv",
            "worker_wellbeing.csv", "worker_wellbeing_triggers.csv", "psychological_safety.csv",
            "feedback_impact.csv", "downtime_minutes.csv", "task_completion_rate.csv"
        ]
        for file in required_files:
            if not os.path.exists(file):
                raise FileNotFoundError(f"Required file {file} not found")

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
        # Load triggers
        triggers_df = pd.read_csv("worker_wellbeing_triggers.csv")
        worker_wellbeing['triggers'] = {
            'threshold': triggers_df['threshold'].dropna().astype(int).tolist(),
            'trend': triggers_df['trend'].dropna().astype(int).tolist(),
            'disruption': triggers_df['disruption'].dropna().astype(int).tolist(),
            'work_area': {}
        }
        for col in triggers_df.columns:
            if col.startswith('work_area_'):
                zone = col.replace('work_area_', '')
                worker_wellbeing['triggers']['work_area'][zone] = triggers_df[col].dropna().astype(int).tolist()
        
        psychological_safety = pd.read_csv("psychological_safety.csv")['psychological_safety'].tolist()
        feedback_impact = pd.read_csv("feedback_impact.csv")['feedback_impact'][0]
        downtime_minutes = pd.read_csv("downtime_minutes.csv")['downtime_minutes'].tolist()
        task_completion_rate = pd.read_csv("task_completion_rate.csv")['task_completion_rate'].tolist()
        
        logger.info("Simulation data loaded successfully", extra={'user_action': 'Load Simulation Data'})
        return (team_positions_df, task_compliance, collaboration_proximity, operational_recovery,
                efficiency_metrics_df, productivity_loss, worker_wellbeing, psychological_safety,
                feedback_impact, downtime_minutes, task_completion_rate)
    except Exception as e:
        logger.error(f"Failed to load simulation data: {str(e)}", extra={'user_action': 'Load Simulation Data'})
        raise

def generate_pdf_report(summary_df):
    """
    Generate a LaTeX-based PDF report from simulation summary.

    Args:
        summary_df (pd.DataFrame): Summary DataFrame with simulation metrics.
    """
    try:
        # Handle NaNs in DataFrame
        summary_df = summary_df.fillna(0.0)
        
        # Generate LaTeX table
        table_content = "\\begin{tabular}{l" + "r" * (len(summary_df.columns) - 1) + "}\n\\toprule\n"
        headers = [sanitize_latex(col) for col in summary_df.columns]
        table_content += " & ".join(headers) + " \\\\\n\\midrule\n"
        for _, row in summary_df.head(10).iterrows():
            row_values = [sanitize_latex(f"{val:.1f}") for val in row]
            table_content += " & ".join(row_values) + " \\\\\n"
        table_content += "\\bottomrule\n\\end{tabular}\n"
        
        # Write LaTeX file
        with open("workplace_report.tex", "w", encoding='utf-8') as f:
            f.write("\\documentclass[a4paper,12pt]{article}\n")
            f.write("\\usepackage[utf8]{inputenc}\n")
            f.write("\\usepackage[T1]{fontenc}\n")
            f.write("\\usepackage{lmodern}\n")
            f.write("\\usepackage[margin=1in]{geometry}\n")
            f.write("\\usepackage{booktabs}\n")
            f.write("\\usepackage{parskip}\n")
            f.write("\\usepackage{noto}\n")
            f.write("\\begin{document}\n")
            f.write("\\title{Workplace Shift Monitoring Report}\n")
            f.write("\\author{Generated by Dashboard}\n")
            f.write("\\date{May 2025}\n")
            f.write("\\maketitle\n")
            f.write("\\section{Summary Statistics}\n")
            f.write(table_content)
            f.write("\\end{document}\n")
        logger.info("PDF report generated successfully at 'workplace_report.tex'", extra={'user_action': 'Generate PDF Report'})
    except Exception as e:
        logger.error(f"Failed to generate PDF report: {str(e)}", extra={'user_action': 'Generate PDF Report'})
        raise
