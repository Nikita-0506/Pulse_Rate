# Pulse Rate Analysis

Ayurvedic pulse analysis system for multi-sensor CSV data.

This project:
- Loads patient CSV files
- Cleans and filters pulse signals
- Extracts feature vectors
- Trains a 3-cluster KMeans model
- Maps clusters to pulse types: Vata, Pitta, Kapha
- Exports model artifacts and analysis reports
- Provides a Streamlit UI for training, prediction, and result viewing

## Pulse Type Mapping

- Vata -> Snake
- Pitta -> Crow/Frog
- Kapha -> Swan

## Entry Points

- `pluse.py`: end-to-end pipeline, model training, export
- `app.py`: Streamlit app
- `QUICK_START.py`: scripted usage examples

## Current Project Structure

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

## Input Data Requirements

Place patient files in `Pulse_CSV/` (or provide another folder path).

Canonical columns used by the pipeline:
- `Time`
- `S1`
- `S2`
- `S3`

The loader can auto-map common aliases like `timestamp`, `sample`, `sensor1`, `channel1`, `pulse1`, and similar names.

Example CSV:

```csv
Time,S1,S2,S3
0,100,95,102
1,101,96,103
2,102,97,104
```

## Run The Pipeline

```bash
python pluse.py
```

Main stages:
1. Data collection and schema normalization
2. Data quality checks
3. Cleaning (missing values, duplicate patient-time rows, outlier capping)
4. Signal filtering and peak detection
5. Feature extraction
6. EDA export
7. Scaling and PCA
8. KMeans clustering (`k=3`)
9. Cluster-to-pulse mapping
10. Save model and output reports

## Launch Streamlit UI

```bash
streamlit run app.py
```

App pages:
- Home
- Train & Analyze
- Predict New Patient
- View Results

## Generated Artifacts

After training, the following are created or updated:

- `models/scaler.pkl`
- `models/pca.pkl`
- `models/kmeans.pkl`
- `models/model_summary.json`
- `outputs/cluster_results.csv`
- `outputs/eda_summary.csv`
- `outputs/plots/eda_correlation_top20.png`

## Predict A New Patient In Python

```python
from pluse import PulseAnalysisWorkflow

workflow = PulseAnalysisWorkflow()
result = workflow.predict_new_csv(
    file_path="path/to/new_patient.csv",
    model_dir="models"
)

print(result)
```

Typical prediction keys:
- `patient_id`
- `cluster`
- `pulse_type`
- `description`

## Programmatic Full Run

```python
from pluse import analyze_pulse_data

# Use repository folder data
results, artifacts = analyze_pulse_data(folder_path="Pulse_CSV")
```

## Important Configuration Note

Default data folder in `pluse.py` is:

```python
DEFAULT_DATA_FOLDER = r"C:\Users\ndpac\OneDrive\Desktop\Pulse_CSV"
```

If your CSV files are inside this repository (`Pulse_Rate/Pulse_CSV`), run with `folder_path="Pulse_CSV"` (as shown above) or set the folder path in the Streamlit sidebar.

## Quick Helper Script

```bash
python QUICK_START.py
```

It demonstrates:
- Full workflow execution
- Loading saved model artifacts
- Predicting on a new CSV
- Feature extraction flow
