# Pulse Rate Analysis

Ayurvedic pulse analysis for multi-sensor CSV data. The project loads patient pulse files, cleans and filters the signals, extracts numerical features, clusters patients into three pulse types, and saves model artifacts and analysis outputs for reuse.

## Overview

The pipeline classifies each patient into one of three Ayurvedic pulse groups:

- Vata: Snake
- Pitta: Crow/Frog
- Kapha: Swan

There are two main entry points:

- `pluse.py`: runs the full batch analysis pipeline and saves artifacts
- `app.py`: launches the Streamlit UI for training, prediction, and result review

## Project Layout

```text
Pulse_Rate/
├── app.py
├── pluse.py
├── QUICK_START.py
├── README.md
├── requirements.txt
├── models/
│   ├── scaler.pkl
│   ├── pca.pkl
│   ├── kmeans.pkl
│   └── model_summary.json
├── outputs/
│   ├── cluster_results.csv
│   ├── eda_summary.csv
│   └── plots/
└── Pulse_CSV/
    ├── deep.csv
    ├── komal Gadge.csv
    ├── krishna.csv
    ├── prerna1.csv
    └── shreya kherde.csv
```

## Requirements

Install the dependencies with:

```bash
pip install -r requirements.txt
```

The project uses pandas, numpy, scipy, scikit-learn, matplotlib, joblib, and streamlit.

## Input Data

Put patient CSV files in `Pulse_CSV/`. The workflow expects these canonical columns after normalization:

- `Time`
- `S1`
- `S2`
- `S3`

The loader also accepts common aliases such as `timestamp`, `sample`, `sensor1`, `channel1`, `pulse1`, and similar variants.

Example format:

```csv
Time,S1,S2,S3
0,100,95,102
1,101,96,103
2,102,97,104
```

## Running The Pipeline

Run the full analysis from the command line:

```bash
python pluse.py
```

This performs the full workflow:

1. Load all CSV files in `Pulse_CSV/`
2. Inspect and clean the data
3. Apply pulse signal filtering
4. Extract statistical, frequency, HRV, and shape-based features
5. Run EDA and save a correlation plot
6. Scale features and apply PCA
7. Cluster patients with K-Means using 3 clusters
8. Save model artifacts and result files

## Streamlit App

Launch the UI with:

```bash
streamlit run app.py
```

The app includes pages for:

- Home and project summary
- Training and analysis
- Predicting a new patient CSV
- Viewing saved outputs and model information

## Outputs

After a successful run, the project writes these files:

- `outputs/cluster_results.csv`: patient cluster assignments and pulse labels
- `outputs/eda_summary.csv`: descriptive statistics for extracted features
- `outputs/plots/eda_correlation_top20.png`: top feature correlation heatmap
- `models/scaler.pkl`: fitted `StandardScaler`
- `models/pca.pkl`: fitted PCA transformer
- `models/kmeans.pkl`: fitted K-Means model
- `models/model_summary.json`: feature list, cluster mapping, and metrics

## Model Results

The saved cluster mapping uses the project’s Ayurvedic labels:

- Vata: Snake
- Pitta: Crow/Frog
- Kapha: Swan

The model summary includes:

- silhouette score
- Davies-Bouldin score
- Calinski-Harabasz score
- feature count
- saved feature column order

## Predict A New CSV

Once the model artifacts exist in `models/`, you can predict a new file from Python:

```python
from pluse import PulseAnalysisWorkflow

workflow = PulseAnalysisWorkflow()
result = workflow.predict_new_csv("path/to/new_patient.csv", model_dir="models")

print(result)
```

Expected output includes the patient ID, predicted cluster, and pulse type.

## Reusable Components

The main reusable classes in `pluse.py` are:

- `PulseDataCollector`: loads and normalizes CSV files
- `PulseDataCleaner`: removes invalid rows and reduces noise
- `PulseSignalProcessor`: filters signals and detects peaks
- `PulseFeatureExtractor`: builds the feature table
- `PulseAnalysisWorkflow`: runs the end-to-end pipeline
- `ModelPersistence`: saves and loads trained artifacts

## Configuration

Default paths are defined in `pluse.py`:

```python
DEFAULT_DATA_FOLDER = r"C:\Users\ndpac\OneDrive\Desktop\Pulse_CSV"
MODEL_DIR = r"models"
OUTPUT_DIR = r"outputs"
PLOTS_DIR = r"outputs\plots"
```

If your data lives somewhere else, update `DEFAULT_DATA_FOLDER` or pass a custom folder to `analyze_pulse_data()`.

## Quick Start Script

`QUICK_START.py` contains helper examples for:

- running the full analysis pipeline
- loading a trained model
- predicting a new patient file
- extracting features only

## Notes

- The repository already contains sample CSV files in `Pulse_CSV/`.
- Generated model and output files are expected to be recreated as you rerun the pipeline.
- If a CSV uses different column names, the loader will try to map common aliases automatically.
