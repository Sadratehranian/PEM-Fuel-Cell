# Data3040 Analysis

This repository contains a Python script and a dataset for analyzing anomalies in cell voltage readings. The goal is to identify patterns, detect anomalies, and visualize results for improved analysis.

## Contents

- **`Main.py`**: Python script for loading, cleaning, and analyzing the dataset.
- **`data3040b.xlsx`**: Dataset containing voltage readings for multiple cells.
- **`README.md`**: Documentation for this project.

## Features

1. Data Cleaning:
   - Removes redundant rows (e.g., empty rows or units).
   - Ensures proper formatting for analysis.

2. Feature Engineering:
   - Calculates rolling statistics (mean, standard deviation, etc.).
   - Detects anomalies in cell voltage readings.

3. Anomaly Detection Techniques:
   - **Isolation Forest**
   - **DBSCAN**
   - **K-Means Clustering**
   - **One-Class SVM**

4. Visualization:
   - Distribution histograms
   - Time-series plots
   - Scatter plots for relationships
   - Heatmaps for correlation

## Setup and Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/Data3040_Analysis.git
