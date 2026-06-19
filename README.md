# Pulse Rate Analysis: Ayurvedic Pulse Classification

A professional end-to-end machine learning pipeline for analyzing pulse sensor data and classifying pulse patterns into three Ayurvedic pulse types: **Snake (Vata)**, **Crow/Frog (Pitta)**, and **Swan (Kapha)**.

---

## 📋 Project Overview

This system processes multi-sensor pulse data from patient CSV files, performs comprehensive signal processing and feature extraction, and uses unsupervised K-Means clustering to identify distinct pulse patterns corresponding to Ayurvedic classifications.

### Architecture Flow

```
Patient CSV Folder (Pulse_CSV/)
        ↓
Load All CSV Files
        ↓
Data Understanding
        ↓
Data Cleaning
        ↓
Signal Processing (Bandpass Filter)
        ↓
Feature Extraction (Statistical, Frequency, HRV, Shape)
        ↓
Feature Dataset
        ↓
EDA (Correlation Analysis)
        ↓
Feature Scaling (StandardScaler)
        ↓
PCA (Dimensionality Reduction)
        ↓
K-Means Clustering (K=3)
        ↓
Cluster Generation & Pattern Identification
        ↓
Validation (Silhouette, Davies-Bouldin, Calinski-Harabasz)
        ↓
Save Model & Results
        ↓
Future CSV Prediction
```

---

## 📁 Project Structure

```
PULSE_RATE/
├── .vscode/                    # VS Code settings
├── models/                     # Trained model artifacts
│   ├── scaler.pkl             # StandardScaler
│   ├── pca.pkl                # PCA transformer
│   ├── kmeans.pkl             # KMeans clustering model
│   └── model_summary.json     # Model metadata & metrics
├── outputs/                    # Analysis results
│   ├── cluster_results.csv    # Patient cluster assignments
│   ├── eda_summary.csv        # Feature statistics
│   └── plots/
│       └── eda_correlation_top20.png   # Correlation heatmap
├── Pulse_CSV/                 # Input patient data (CSV files)
│   ├── patient1.csv
│   ├── patient2.csv
│   └── ...
├── pluse.py                   # Main analysis pipeline
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Input Data

Place your patient CSV files in the `Pulse_CSV/` folder. Each CSV must contain these required columns:
- **Time**: Timestamp or sample index (numeric)
- **S1**: Sensor 1 readings (numeric)
- **S2**: Sensor 2 readings (numeric)
- **S3**: Sensor 3 readings (numeric)

Example CSV structure:
```
Time,S1,S2,S3
0,100,95,102
1,101,96,103
2,102,97,104
...
```

### 3. Run Analysis

```bash
python pluse.py
```

### 4. View Results

After execution, check these outputs:
- **Cluster assignments**: `outputs/cluster_results.csv`
- **Feature summary**: `outputs/eda_summary.csv`
- **Visualizations**: `outputs/plots/eda_correlation_top20.png`
- **Trained models**: `models/scaler.pkl`, `models/pca.pkl`, `models/kmeans.pkl`

---

## 📊 Output Files

### cluster_results.csv
Contains patient-to-cluster mapping with pulse type classifications:
```
patient_id,cluster_id,pulse_type,description,confidence_score,cluster_size
patient1,0,Pitta,Crow/Frog (Pitta),0.65,12
patient2,2,Kapha,Swan (Kapha),0.58,14
patient3,1,Vata,Snake (Vata),0.72,10
```

### eda_summary.csv
Statistical summary of extracted features (mean, std, min, max, 25%, 50%, 75%)

### Model Files
- **scaler.pkl**: Standardizes features to zero mean and unit variance
- **pca.pkl**: Reduces dimensionality while preserving 90% variance
- **kmeans.pkl**: Clustering model with 3 centers
- **model_summary.json**: Metadata, cluster mapping, and validation metrics

---

## 🔬 Feature Extraction

### Statistical Features (per sensor S1, S2, S3)
- Mean, Std, Median, Min, Max, Range
- Coefficient of Variation (CV)
- Skewness, Kurtosis
- Percentiles (25th, 75th, 90th)

### Frequency Features
- FFT mean, std, max
- Dominant frequency
- Spectral entropy

### Heart Rate Variability (HRV)
- Heart rate (bpm) statistics
- RR interval statistics
- RMSSD (root mean square of successive differences)
- PNN50 (percentage of RR intervals > 50ms)
- Stress index
- Total power

### Pulse Shape Features
- Average peak height
- Peak height coefficient of variation
- Number of peaks
- Average peak interval
- Peak interval variability
- Total pulse duration

---

## 🎯 Clustering Output

### Pulse Type Mapping

| Cluster | Pulse Type | Description | Characteristics |
|---------|-----------|-------------|-----------------|
| Vata | Snake 🐍 | Fast, feeble, irregular | High variability, low amplitude, irregular rhythm |
| Pitta | Crow/Frog 🐸 | Strong, high amplitude, forceful | High amplitude, strong, regular pulse |
| Kapha | Swan 🦢 | Slow, deep, smooth | Low variability, stable, smooth rhythm |

### Validation Metrics

- **Silhouette Score**: Measures cluster cohesion (-1 to 1, higher is better)
- **Davies-Bouldin Score**: Measures cluster separation (lower is better)
- **Calinski-Harabasz Score**: Ratio of between-cluster to within-cluster variance (higher is better)

---

## 🔮 Future Predictions

To predict pulse type for a new patient:

```python
from pluse import PulseAnalysisWorkflow

workflow = PulseAnalysisWorkflow()
prediction = workflow.predict_new_csv("path/to/new_patient.csv", model_dir="models/")

print(prediction)
# Output: {
#     'patient_id': 'new_patient',
#     'cluster': 1,
#     'pulse_type': 'Snake (Vata)'
# }
```

---

## 📈 Signal Processing Details

### Bandpass Filter
- **Frequency Range**: 0.5 - 5.0 Hz (typical pulse frequency)
- **Filter Type**: Butterworth (order 4)
- **Purpose**: Remove noise and extract pulse signal

### Data Cleaning
1. Remove all-zero records
2. Remove duplicate timestamps
3. Clip outliers (1st and 99th percentiles ± 1.5×IQR)
4. Apply median filter (window=3) for noise reduction
5. Interpolate missing values
6. Ensure non-negative values

---

## 🛠️ Dependencies

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scipy**: Scientific computing (signal processing, statistics)
- **scikit-learn**: Machine learning (scaling, PCA, clustering)
- **matplotlib**: Data visualization
- **joblib**: Model serialization

---

## ⚙️ Configuration

Edit these constants in `pluse.py`:

```python
DEFAULT_DATA_FOLDER = r"C:\Users\ndpac\OneDrive\Desktop\Pulse_CSV"
MODEL_DIR = r"models"
OUTPUT_DIR = r"outputs"
PLOTS_DIR = r"outputs\plots"
```

---

## 📝 Usage Examples

### Example 1: Full Pipeline Analysis
```python
from pluse import analyze_pulse_data

results, model_artifacts = analyze_pulse_data(
    folder_path=r"C:\Users\ndpac\OneDrive\Desktop\Pulse_CSV"
)

print(f"Patients: {len(results.features_df)}")
print(f"Features: {results.scaled_features.shape[1]}")
print(f"Silhouette Score: {results.metrics['silhouette']:.4f}")
```

### Example 2: Load Trained Model
```python
from pluse import ModelPersistence

model = ModelPersistence.load(model_dir="models/")
print(model["cluster_mapping"])
```

### Example 3: Predict New File
```python
from pluse import PulseAnalysisWorkflow

workflow = PulseAnalysisWorkflow()
result = workflow.predict_new_csv("new_patient.csv", model_dir="models/")
print(f"Predicted type: {result['pulse_type']}")
```

---

## 🐛 Troubleshooting

### No CSV files found
- Ensure CSV files are in `Pulse_CSV/` folder
- Check file extension is `.csv` (lowercase)

### Missing required columns
- Verify all CSVs have Time, S1, S2, S3 columns
- Column names are case-sensitive

### Model not loading
- Ensure `models/` directory exists
- Verify all three model files (.pkl) are present
- Check `model_summary.json` exists

### Insufficient data for PCA
- Ensure at least 10+ patients in dataset
- Check features don't have >30% missing values

---

## 📧 Notes

- The pipeline is optimized for pulse sensor data with ~100Hz sampling rate
- EDA and visualization outputs help validate feature quality
- Model is saved to `models/` for reproducible future predictions
- All timestamps should be numeric (seconds, milliseconds, or sample indices)

---

## 📄 License

This project is for research and educational purposes.

---

## 🎓 Ayurvedic Context

**Vata (Snake)**: Associated with movement and change; characterized by variable, light pulse
**Pitta (Crow/Frog)**: Associated with transformation and metabolism; characterized by strong, forceful pulse
**Kapha (Swan)**: Associated with stability and structure; characterized by slow, deep pulse

---

**Last Updated**: June 17, 2026
K-Means Clustering (K = 3)
        │
        ▼
Cluster Validation
        │
        ▼
Pattern Identification
        │
 ┌──────┼──────┐
 ▼      ▼      ▼
Snake  Frog   Swan
(Vata) (Pitta) (Kapha)
        │
        ▼
Save Model
        │
        ▼
Future CSV Prediction
```

---

# Folder Structure

```text
Pulse_Project/
│
├── pulse_analysis.py
├── pulse_model.pkl
├── model_summary.json
├── requirements.txt
├── README.md
│
└── Pulse_CSV/
    ├── patient_1.csv
    ├── patient_2.csv
    ├── patient_3.csv
    └── ...
```

---

# Input Data Format

Each CSV file must contain the following columns:

| Column | Description    |
| ------ | -------------- |
| Time   | Timestamp      |
| S1     | Pulse Sensor 1 |
| S2     | Pulse Sensor 2 |
| S3     | Pulse Sensor 3 |

Example:

```csv
Time,S1,S2,S3
1174247,871,1564,1013
1174270,854,1583,1065
1174293,861,1575,1048
```

---

# Features of the System

### Data Collection

* Reads all CSV files from a folder
* Combines patient data automatically
* Tracks patient identifiers

### Data Cleaning

* Removes invalid rows
* Removes duplicate records
* Handles missing values
* Removes noise
* Caps outliers

### Signal Processing

* Bandpass filtering
* Pulse peak detection
* RR interval calculation

### Feature Extraction

Extracts:

* Mean
* Standard Deviation
* Median
* Range
* Skewness
* Kurtosis
* Frequency Features
* Spectral Entropy
* Heart Rate Features
* HRV Features
* Pulse Shape Features

### Clustering

Uses:

* StandardScaler
* PCA
* K-Means (K = 3)

### Validation

Evaluates clusters using:

* Silhouette Score
* Davies-Bouldin Score
* Calinski-Harabasz Score

---

# Ayurvedic Pulse Mapping

## Snake (Vata)

Characteristics:

* Fast pulse
* Irregular pulse
* High variability
* High entropy

Symbol:

```text
🐍 Snake
```

---

## Crow/Frog (Pitta)

Characteristics:

* Strong pulse
* Forceful pulse
* Moderate variability
* Higher amplitude

Symbol:

```text
🐸 Frog
```

---

## Swan (Kapha)

Characteristics:

* Slow pulse
* Stable pulse
* Smooth waveform
* Low variability

Symbol:

```text
🦢 Swan
```

---

# Installation

## Step 1

Clone the repository

```bash
git clone <repository-url>
```

---

## Step 2

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Step 3

Update folder path

Inside the code:

```python
DEFAULT_DATA_FOLDER = r"C:\Users\ndpac\OneDrive\Desktop\Pulse_CSV"
```

---

# Run the Project

```bash
python pulse_analysis.py
```

---

# Output Files

After execution:

```text
pulse_model.pkl
model_summary.json
eda_summary.csv
eda_correlation_top20.png
```

---

# Example Output

```text
Patients processed: 50

Feature count: 72

Clusters: 3

Cluster 0:
Snake (Vata)

Cluster 1:
Crow/Frog (Pitta)

Cluster 2:
Swan (Kapha)

Silhouette Score: 0.81
```

---

# Future CSV Prediction

The trained model can classify new pulse recordings.

Workflow:

```text
New CSV File
      │
      ▼
Cleaning
      │
      ▼
Filtering
      │
      ▼
Feature Extraction
      │
      ▼
Scaling
      │
      ▼
PCA
      │
      ▼
K-Means Prediction
      │
      ▼
Pulse Type
```

Example:

```text
Patient: ABC

Predicted Pattern:

🐍 Snake (Vata)
```

---

# Technologies Used

* Python
* Pandas
* NumPy
* SciPy
* Scikit-Learn
* Matplotlib
* Joblib

---

# Future Improvements

* Real-time pulse monitoring
* Streamlit dashboard
* Deep Learning models
* Automatic report generation
* Multi-class supervised classification
* Clinical validation with experts

---

# Author

Nikita Pachkate

Pulse Pattern Clustering and Ayurvedic Pulse Analysis System

Built using Machine Learning, Signal Processing, and Data Analytics.
