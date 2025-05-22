# utils.py
import pickle
import pandas as pd
import numpy as np 
import logging

logger = logging.getLogger(__name__)
SAVE_FILE_PATH = "simulation_data_cache.pkl"

def save_simulation_data(simulation_results_dict: dict):
    try:
        with open(SAVE_FILE_PATH, "wb") as f:
            pickle.dump(simulation_results_dict, f)
        logger.info(f"Simulation data successfully saved to {SAVE_FILE_PATH}", extra={'user_action': 'Save Data - Success'})
    except Exception as e:
        logger.error(f"Error saving simulation data to {SAVE_FILE_PATH}: {e}", exc_info=True, extra={'user_action': 'Save Data - Error'})

def load_simulation_data():
    try:
        with open(SAVE_FILE_PATH, "rb") as f:
            simulation_results_dict = pickle.load(f)
        logger.info(f"Simulation data successfully loaded from {SAVE_FILE_PATH}", extra={'user_action': 'Load Data - Success'})
        return simulation_results_dict
    except FileNotFoundError:
        logger.warning(f"Save file {SAVE_FILE_PATH} not found. No data loaded.", extra={'user_action': 'Load Data - File Not Found'})
        return None
    except pickle.UnpicklingError as e:
        logger.error(f"Error unpickling data from {SAVE_FILE_PATH}. File may be corrupted or from an incompatible version: {e}", exc_info=True, extra={'user_action': 'Load Data - Unpickling Error'})
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading data from {SAVE_FILE_PATH}: {e}", exc_info=True, extra={'user_action': 'Load Data - Error'})
        return None

def generate_pdf_report(summary_df: pd.DataFrame):
    if not isinstance(summary_df, pd.DataFrame):
        logger.error("generate_pdf_report expects a Pandas DataFrame as input.", extra={'user_action': 'Generate PDF - Invalid Input'})
        return
    logger.info("Attempting to generate LaTeX report from summary DataFrame.", extra={'user_action': 'Generate PDF - Start'})
    try:
        df_for_latex = summary_df.copy()
        for col in df_for_latex.select_dtypes(include=np.number).columns:
            if not pd.api.types.is_integer_dtype(df_for_latex[col].dropna()): df_for_latex[col] = df_for_latex[col].round(2)
        df_for_latex.columns = [col.replace('_', ' ').title() for col in df_for_latex.columns]
        num_cols = len(df_for_latex.columns)
        col_fmt = '|' + 'l|' * num_cols if num_cols > 0 else '|l|' 
        
        latex_table = df_for_latex.to_latex(
            index=False, 
            escape=True, 
            column_format=col_fmt, 
            header=True, 
            longtable=True, 
            na_rep='-',      
            caption='Summary of Simulation Metrics Over Time.',
            label='tab:simsummary',
        )
        
        # Corrected LaTeX comment
        latex_document = f"""
\\documentclass[10pt,a4paper]{{article}}
\\usepackage[utf8]{{inputenc}}
\\usepackage{{amsmath}}
\\usepackage{{amsfonts}}
\\usepackage{{amssymb}}
\\usepackage{{graphicx}}
\\usepackage{{booktabs}} 
\\usepackage{{longtable}} 
\\usepackage[margin=1in]{{geometry}} 
\\usepackage{{fancyhdr}} 

\\pagestyle{{fancy}}
\\fancyhf{{}} 
\\fancyhead[C]{{Workplace Shift Monitoring Report}}
\\fancyfoot[C]{{\\thepage}}
\\renewcommand{{\\headrulewidth}}{{0.4pt}} 
\\renewcommand{{\\footrulewidth}}{{0.4pt}} 

\\title{{Workplace Shift Monitoring Report}}
\\author{{Automated Simulation System}}
\\date{{\\today}}

\\begin{{document}}
\\maketitle
\\thispagestyle{{empty}} 
\\clearpage
\\pagenumbering{{arabic}} 

\\section*{{Simulation Data Summary}}
The data from the dashboard provides an overview of key operational metrics from the
latest simulation run. The table below presents a time-series summary.

{latex_table}

% Further analysis and visualizations could be included here, 
% potentially by saving plots as images and including them 
% with \\includegraphics{{plot.png}}.

\\end{{document}}
"""
        report_filename = "workplace_report.tex"
        with open(report_filename, "w", encoding="utf-8") as f: f.write(latex_document)
        logger.info(f"LaTeX report '{report_filename}' generated successfully.", extra={'user_action': 'Generate PDF - Success'})
    except Exception as e:
        logger.error(f"Failed to generate LaTeX report: {e}", exc_info=True, extra={'user_action': 'Generate PDF - Error'})
