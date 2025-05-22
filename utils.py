"""
utils.py
Utility functions for the Industrial Workplace Shift Monitoring Dashboard.
Handles data saving, PDF reporting, and data loading.
"""

import logging
import pandas as pd
from config import DEFAULT_CONFIG

logger = logging.getLogger(__name__)

def save_simulation_data(
    team_positions_df: pd.DataFrame,
    efficiency_metrics_df: pd.DataFrame,
    task_compliance: dict,
    collaboration_proximity: dict,
    operational_recovery: list,
    worker_wellbeing: dict,
    psychological_safety: list,
    productivity_loss: list,
    feedback_impact: dict,
    downtime_minutes: list,
    task_completion_rate: list
) -> None:
    """
    Save simulation outputs to CSV files.

    Args:
        team_positions_df (pd.DataFrame): Worker positions (meters).
        efficiency_metrics_df (pd.DataFrame): Efficiency metrics (%).
        task_compliance (dict): Task compliance scores (%).
        collaboration_proximity (dict): Collaboration proximity index (%).
        operational_recovery (list): Recovery scores (%).
        worker_wellbeing (dict): Well-being index (%).
        psychological_safety (list): Safety scores (%).
        productivity_loss (list): Productivity loss (%).
        feedback_impact (dict): Feedback impact (%).
        downtime_minutes (list): Downtime (minutes).
        task_completion_rate (list): Task completion rate (%).
    """
    try:
        logger.info("Saving team_positions.csv")
        team_positions_df.to_csv('team_positions.csv', index=False)
        logger.info("Saving efficiency_metrics.csv")
        efficiency_metrics_df.to_csv('efficiency_metrics.csv', index=False)
        logger.info("Creating summary_metrics DataFrame")
        summary_df = pd.DataFrame({
            'step': range(DEFAULT_CONFIG['SHIFT_DURATION_INTERVALS']),
            'time_minutes': [i * 2 for i in range(DEFAULT_CONFIG['SHIFT_DURATION_INTERVALS'])],
            'task_compliance': task_compliance['data'],
            'collaboration_proximity': collaboration_proximity['data'],
            'operational_recovery': operational_recovery,
            'worker_wellbeing': worker_wellbeing['scores'],
            'psychological_safety': psychological_safety,
            'productivity_loss': productivity_loss,
            'downtime_minutes': downtime_minutes,
            'task_completion_rate': task_completion_rate,
            'feedback_wellbeing': [feedback_impact['wellbeing']] * DEFAULT_CONFIG['SHIFT_DURATION_INTERVALS'],
            'feedback_cohesion': [feedback_impact['cohesion']] * DEFAULT_CONFIG['SHIFT_DURATION_INTERVALS']
        })
        logger.info("Saving summary_metrics.csv")
        summary_df.to_csv('summary_metrics.csv', index=False)
        logger.info("Simulation data saved successfully")
    except Exception as e:
        logger.error(f"Failed to save simulation data: {str(e)}")
        raise

def load_simulation_data() -> tuple:
    """
    Load simulation data from CSV files.

    Returns:
        tuple: Same structure as simulate_workplace_operations output.
    """
    try:
        team_positions_df = pd.read_csv('team_positions.csv')
        efficiency_metrics_df = pd.read_csv('efficiency_metrics.csv')
        summary_df = pd.read_csv('summary_metrics.csv')
        
        task_compliance = {
            'data': summary_df['task_compliance'].values,
            'z_scores': (summary_df['task_compliance'] - summary_df['task_compliance'].mean()) / summary_df['task_compliance'].std(),
            'forecast': None
        }
        collaboration_proximity = {
            'data': summary_df['collaboration_proximity'].values,
            'z_scores': (summary_df['collaboration_proximity'] - summary_df['collaboration_proximity'].mean()) / summary_df['collaboration_proximity'].std(),
            'forecast': None
        }
        operational_recovery = summary_df['operational_recovery'].values
        worker_wellbeing = {
            'scores': summary_df['worker_wellbeing'].values,
            'triggers': {
                'threshold': summary_df[summary_df['worker_wellbeing'] < DEFAULT_CONFIG['WELLBEING_THRESHOLD'] * 100]['step'].tolist(),
                'trend': [],
                'work_area': {},
                'disruption': []
            }
        }
        psychological_safety = summary_df['psychological_safety'].values
        productivity_loss = summary_df['productivity_loss'].values
        downtime_minutes = summary_df['downtime_minutes'].values
        task_completion_rate = summary_df['task_completion_rate'].values
        feedback_impact = {
            'wellbeing': summary_df['feedback_wellbeing'].iloc[0],
            'cohesion': summary_df['feedback_cohesion'].iloc[0]
        }
        
        return (
            team_positions_df,
            task_compliance,
            collaboration_proximity,
            operational_recovery,
            efficiency_metrics_df,
            productivity_loss,
            worker_wellbeing,
            psychological_safety,
            feedback_impact,
            downtime_minutes,
            task_completion_rate
        )
    except Exception as e:
        logger.error(f"Failed to load simulation data: {str(e)}")
        raise

def generate_pdf_report(summary_df: pd.DataFrame, output_file: str = 'workplace_report.tex') -> None:
    """
    Generate a LaTeX-based PDF report of key metrics.

    Args:
        summary_df (pd.DataFrame): Summary metrics DataFrame.
        output_file (str): Output LaTeX file path.
    """
    try:
        latex_content = r"""
\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{geometry}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{amsmath}
\geometry{a4paper, margin=1in}
\begin{document}
\begin{center}
    \textbf{\Large Workplace Shift Monitoring Report} \\
    \vspace{0.5cm}
    \includegraphics[width=0.3\textwidth]{logo.png} \\
    \vspace{0.5cm}
    Manufacturing Facility, Shift Duration: 960 minutes
\end{center}

\section*{Summary Metrics}
\begin{table}[h]
\centering
\begin{tabular}{lr}
\toprule
Metric & Value \\
\midrule
Average Task Compliance (\%) & """ + f"{summary_df['task_compliance'].mean():.1f}" + r""" \\
Average Collaboration Proximity (\%) & """ + f"{summary_df['collaboration_proximity'].mean():.1f}" + r""" \\
Average Operational Recovery (\%) & """ + f"{summary_df['operational_recovery'].mean():.1f}" + r""" \\
Average Worker Well-Being (\%) & """ + f"{summary_df['worker_wellbeing'].mean():.1f}" + r""" \\
Average Psychological Safety (\%) & """ + f"{summary_df['psychological_safety'].mean():.1f}" + r""" \\
Total Productivity Loss (\%) & """ + f"{summary_df['productivity_loss'].sum():.1f}" + r""" \\
Total Downtime (minutes) & """ + f"{summary_df['downtime_minutes'].sum():.1f}" + r""" \\
Average Task Completion Rate (\%) & """ + f"{summary_df['task_completion_rate'].mean():.1f}" + r""" \\
\bottomrule
\end{tabular}
\caption{Summary of key performance metrics.}
\end{table}

\section*{Recommendations}
\begin{itemize}
    \item """ + ("Increase break frequency to improve well-being." if summary_df['worker_wellbeing'].mean() < DEFAULT_CONFIG['WELLBEING_THRESHOLD'] * 100 else "Maintain current break schedule.") + r"""
    \item """ + ("Enhance safety training." if summary_df['psychological_safety'].mean() < DEFAULT_CONFIG['SAFETY_THRESHOLD'] * 100 else "Continue safety initiatives.") + r"""
    \item """ + ("Investigate downtime causes." if summary_df['downtime_minutes'].sum() > DEFAULT_CONFIG['DOWNTIME_THRESHOLD'] * DEFAULT_CONFIG['SHIFT_DURATION_INTERVALS'] / 2 else "Downtime within acceptable limits.") + r"""
\end{itemize}

\end{document}
"""
        with open(output_file, 'w') as f:
            f.write(latex_content)
        logger.info(f"Generated LaTeX report: {output_file}")
    except Exception as e:
        logger.error(f"Failed to generate PDF report: {str(e)}")
        raise
