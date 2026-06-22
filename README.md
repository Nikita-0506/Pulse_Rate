# Pulse Rate Analysis

Ayurvedic pulse analysis system for multi-sensor CSV data. This project cleans pulse signals, extracts features, performs K-Means clustering into three pulse groups, and provides both a Python pipeline and a Streamlit interface.

## Pulse Types

- Vata -> Snake (Vata)
- Pitta -> Crow/Frog (Pitta)
- Kapha -> Swan (Kapha)

## Main Entry Points

- `pluse.py` -> Full end-to-end workflow (data loading to model saving)
- `app.py` -> Streamlit UI for training, prediction, and result exploration
- `QUICK_START.py` -> Helper script with usage examples

## Project Structure

```text
Pulse_Rate/
|-- app.py
|-- pluse.py
|-- QUICK_START.py
|-- README.md
|-- requirements.txt
|-- SETUP_SUMMARY.md
|-- models/
|   |-- scaler.pkl
|   |-- pca.pkl
|   |-- kmeans.pkl
|   `-- model_summary.json
|-- outputs/
|   |-- cluster_results.csv
|   |-- eda_summary.csv
|   `-- plots/
|       `-- eda_correlation_top20.png
`-- Pulse_CSV/
    |-- deep.csv
    |-- komal Gadge.csv
    |-- krishna.csv
    |-- prerna1.csv
    `-- shreya kherde.csv
```

## Installation

```bash
pip install -r requirements.txt
```

Dependencies include:

- pandas
- numpy
- scipy
- scikit-learn
- matplotlib
- joblib
- streamlit

## Input CSV Requirements

Place patient CSV files in `Pulse_CSV/`.

Canonical columns used by the pipeline:

- `Time`
- `S1`
- `S2`
- `S3`

The loader can auto-map common aliases such as `timestamp`, `sample`, `sensor1`, `channel1`, `pulse1`, etc.

Example:

```csv
Time,S1,S2,S3
0,100,95,102
1,101,96,103
2,102,97,104
```

## Run The Full Pipeline

```bash
python pluse.py
```

Pipeline stages:

1. Collect and merge all patient CSV files
2. Data understanding and quality checks
3. Cleaning (invalid rows, duplicates, outlier capping)
4. Signal filtering and peak detection
5. Feature extraction (statistical, frequency, variability, pulse-shape)
6. EDA export (summary + correlation dashboard plot)
7. Feature scaling and PCA
8. K-Means clustering (K=3)
9. Cluster-to-pulse mapping (Vata/Pitta/Kapha)
10. Save models and output reports

## Launch Streamlit App

```bash
streamlit run app.py
```

UI pages:

- Home
- Train and Analyze
- Predict New Patient
- View Results

## Generated Artifacts

After running training, these files are generated/updated:

- `models/scaler.pkl`
- `models/pca.pkl`
- `models/kmeans.pkl`
- `models/model_summary.json`
- `outputs/cluster_results.csv`
- `outputs/eda_summary.csv`
- `outputs/plots/eda_correlation_top20.png`

## Predict A New CSV From Python

```python
from pluse import PulseAnalysisWorkflow

workflow = PulseAnalysisWorkflow()
result = workflow.predict_new_csv(
    file_path="path/to/new_patient.csv",
    model_dir="models/"
)

print(result)
```

Expected fields in prediction result:

- `patient_id`
- `cluster`
- `pulse_type`

## Reusable Classes

Key classes in `pluse.py`:

- `PulseDataCollector`
- `PulseDataUnderstanding`
- `PulseDataCleaner`
- `PulseSignalProcessor`
- `PulseFeatureExtractor`
- `FeaturePreprocessor`
- `PulseClusterer`
- `ModelPersistence`
- `PulseAnalysisWorkflow`

## Configuration Notes

Current defaults in `pluse.py`:

```python
DEFAULT_DATA_FOLDER = r"C:\Users\ndpac\OneDrive\Desktop\Pulse_CSV"
MODEL_DIR = r"models"
OUTPUT_DIR = r"outputs"
PLOTS_DIR = r"outputs\plots"
```

If your CSV files are in this repository folder (`Pulse_Rate/Pulse_CSV`), pass a custom folder path when running from Python, for example:

```python
from pluse import analyze_pulse_data

results, artifacts = analyze_pulse_data(folder_path="Pulse_CSV")
```

## Quick Start Helper

You can also run:

```bash
python QUICK_START.py
```

It demonstrates:

- Full workflow run
- Loading saved model artifacts
- Predicting a new patient file
- Extracting features only
