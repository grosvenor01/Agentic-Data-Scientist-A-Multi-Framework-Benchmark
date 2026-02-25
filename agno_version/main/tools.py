import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys, io , traceback

def EDA(file_path:str , output_dir:str="figs/"):
    """
    A tool to perform automated exploratory data analysis (EDA) on a CSV dataset.
    Generates summary statistics, missing values, unique counts, and visualizations.
    """
    df = pd.read_csv(file_path)

    col_info = df.dtypes.to_dict()
    missing_values = df.isna().sum().to_dict()
    unique_counts = df.nunique().to_dict()
    stats = df.describe(include='all').to_dict()

    # Separate numeric and categorical columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=['number']).columns.tolist()

    # Visualizations
    charts = []

    # Histograms for numeric columns
    for col in numeric_cols:
        plt.figure(figsize=(6,4))
        sns.histplot(df[col], kde=True)
        chart_path = os.path.join(output_dir, f"{col}_hist.png")
        plt.savefig(chart_path)
        plt.close()
        charts.append(chart_path)

    # Barplots for categorical columns
    for col in categorical_cols:
        plt.figure(figsize=(6,4))
        counts = df[col].value_counts()
        sns.barplot(x=counts.index, y=counts.values)
        plt.xticks(rotation=45)
        chart_path = os.path.join(output_dir, f"{col}_bar.png")
        plt.savefig(chart_path)
        plt.close()
        charts.append(chart_path)

        # Correlation heatmap (if numeric columns exist)
        if len(numeric_cols) > 1:
            plt.figure(figsize=(8,6))
            sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm")
            heatmap_path = os.path.join(output_dir, "correlation_heatmap.png")
            plt.savefig(heatmap_path)
            plt.close()
            charts.append(heatmap_path)

        # Construct report
        report = {
            "columns_info": col_info,
            "missing_values": missing_values,
            "unique_counts": unique_counts,
            "statistics": stats,
            "charts": charts,
            "report": "Dataset structure, missing values, key patterns, correlations, and visualizations."
        }
        return report
analysis_tools = [EDA]

import subprocess
import traceback

import subprocess
import traceback

def run_python_script(code: str) -> str:
    filename = "excutables/code_to_run.py"

    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(code)

        result = subprocess.run(
            [sys.executable, filename],
            capture_output=True,
            text=True
        )

        stdout = result.stdout.strip()
        stderr = result.stderr.strip()

        if result.returncode == 0:
            return stdout or "Execution successful (no output)."

        return f"Execution failed:\n{stderr}"

    except Exception:
        return f"Unexpected error:\n{traceback.format_exc()}"
    
preprocessing_tools = [run_python_script]


training_tools = []
evaluation_tools = []