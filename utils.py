# utils.py
import pickle
import pandas as pd
import numpy as np 
import logging

logger = logging.getLogger(__name__)
SAVE_FILE_PATH = "simulation_data_cache.pkl" # Consistent naming

def save_simulation_data(simulation_results_dict: dict):
    """Saves the entire simulation results dictionary to a file using pickle."""
    try:
        with open(SAVE_FILE_PATH, "wb") as f:
            pickle.dump(simulation_results_dict, f)
        logger.info(f"Simulation data successfully saved to {SAVE_FILE_PATH}", extra={'user_action': 'Save Data - Success'})
    except Exception as e:
        logger.error(f"Error saving simulation data to {SAVE_FILE_PATH}: {e}", exc_info=True, extra={'user_action': 'Save Data - Error'})
        # Depending on requirements, you might want to re-raise or handle differently in Streamlit
        # For now, logging the error is the primary action.

def load_simulation_data():
    """
    Loads the simulation results dictionary from a file using pickle.
    Returns the loaded dictionary or None if an error occurs or file not found.
    """
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
    """
    Generates a LaTeX (.tex) file as a placeholder for a PDF report.
    This requires a LaTeX distribution (e.g., MiKTeX, TeX Live, Overleaf) to compile into a PDF.
    """
    if not isinstance(summary_df, pd.DataFrame):
        logger.error("generate_pdf_report expects a Pandas DataFrame as input.", extra={'user_action': 'Generate PDF - Invalid Input'})
        return

    logger.info("Attempting to generate LaTeX report from summary DataFrame.", extra={'user_action': 'Generate PDF - Start'})
    
    try:
        # Ensure numeric columns are rounded for better presentation in LaTeX
        df_for_latex = summary_df.copy()
        for col in df_for_latex.select_dtypes(include=np.number).columns:
            # Check if column has non-integer floats before rounding to avoid ".0"
            if not pd.api.types.is_integer_dtype(df_for_latex[col].dropna()):
                 df_for_latex[col] = df_for_latex[col].round(2)
        
        # Handle potential MultiIndex columns by flattening if necessary, or choose specific columns
        # For simplicity, assuming single-level columns or that to_latex handles it.
        
        # Replace underscores with spaces and capitalize for column headers for better readability
        df_for_latex.columns = [col.replace('_', ' ').title() for col in df_for_latex.columns]

        latex_table = df_for_latex.to_latex(
            index=False, 
            escape=True, # Escapes special LaTeX characters
            column_format='|' + 'l|' * len(df_for_latex.columns), # Left-align columns
            header=True,
            longtable=False, # Use longtable if your table spans multiple pages
            na_rep='-',      # Representation for NaN values
            caption='Summary of Simulation Metrics Over Time.',
            label='tab:simsummary',
            position='!htbp' # LaTeX float position suggestion
        )
        
        latex_document = f"""
\\documentclass[10pt,a4paper]{{article}}
\\usepackage[utf8]{{inputenc}}
\\usepackage{{amsmath}}
\\usepackage{{amsfonts}}
\\usepackage{{amssymb}}
\\usepackage{{graphicx}}
\\usepackage{{booktabs}} % For professional tables (midrule, toprule, bottomrule)
\\usepackage{{longtable}} % If tables span pages
\\usepackage[margin=1in]{{geometry}} % Adjust margins
\\usepackage{{fancyhdr}} % For headers/footers

\\pagestyle{{fancy}}
\\fancyhf{{}} % Clear header/footer
\\fancyhead[C]{{Workplace Shift Monitoring Report}}
\\fancyfoot[C]{{\\thepage}}

\\title{{Workplace Shift Monitoring Report}}
\\author{{Automated Simulation System}}
\\date{{\\today}}

\\begin{{document}}
\\maketitle
\\thispagestyle{{empty}} % No header/footer on title page
\\clearpage
\\pagenumbering{{arabic}} % Start page numbering

\\section*{{Simulation Data Summary}}
\\ Gendatafrom the dashboard provides an overview of key operational metrics from the
\\ latest simulation run. The table below presents a time-series summary.

{latex_table}

\\ Gendfurther analysis and visualizations could be included here, potentially by
\\ saving plots as images and including them with \\includegraphics{{plot.png}}.

\\end{{document}}
"""
        report_filename = "workplace_report.tex"
        with open(report_filename, "w", encoding="utf-8") as f:
            f.write(latex_document)
        
        logger.info(f"LaTeX report '{report_filename}' generated successfully. Compile with a LaTeX distribution to produce PDF.", extra={'user_action': 'Generate PDF - Success'})
    except Exception as e:
        logger.error(f"Failed to generate LaTeX report: {e}", exc_info=True, extra={'user_action': 'Generate PDF - Error'})
