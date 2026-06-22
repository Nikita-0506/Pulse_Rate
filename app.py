"""
Nadi Pulse Analysis - Streamlit UI
Ayurvedic Pulse Classification: Vata / Pitta / Kapha
"""

import io
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for Streamlit

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Make sure the project directory is on the path
sys.path.insert(0, os.path.dirname(__file__))

from pluse import (
    DEFAULT_DATA_FOLDER,
    MODEL_DIR,
    OUTPUT_DIR,
    PLOTS_DIR,
    ModelPersistence,
    PulseAnalysisWorkflow,
    PulseDataCleaner,
    PulseDataCollector,
    PulseFeatureExtractor,
    PulseSignalProcessor,
    SENSOR_COLUMNS,
    REQUIRED_COLUMNS,
    WorkflowResults,
)

# ------------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Pulse Analysis",
    page_icon="👩🏻‍💻",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# Custom CSS
# ------------------------------------------------------------------
st.markdown(
    """
    <style>
    .pulse-header {
        background: linear-gradient(90deg, #c0392b, #e74c3c);
        color: white;
        padding: 1.2rem 1.8rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .pulse-type-vata   { border-left: 5px solid #3498db; padding-left: 12px; }
    .pulse-type-pitta  { border-left: 5px solid #e67e22; padding-left: 12px; }
    .pulse-type-kapha  { border-left: 5px solid #27ae60; padding-left: 12px; }
    .stAlert { margin-top: 0.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# Sidebar navigation
# ------------------------------------------------------------------
st.sidebar.markdown("## 👩🏻‍💻 Pulse Analysis")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "🔬 Train & Analyze", "🔮 Predict New Patient", "📊 View Results"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Configuration")
data_folder = st.sidebar.text_input(
    "CSV Data Folder",
    value=DEFAULT_DATA_FOLDER,
    help="Path to the folder containing patient pulse CSV files",
)
model_dir = st.sidebar.text_input(
    "Model Directory",
    value=MODEL_DIR,
    help="Directory where trained models are saved/loaded",
)

st.sidebar.markdown("---")
st.sidebar.caption("Ayurvedic Pulse Classification\n🐍 Vata · 🐸 Pitta · 🦢 Kapha")


# ------------------------------------------------------------------
# Helper utilities
# ------------------------------------------------------------------

PULSE_COLORS = {
    "Vata":  "#3498db",
    "Pitta": "#e67e22",
    "Kapha": "#27ae60",
}

PULSE_ICONS = {
    "Vata":  "🐍",
    "Pitta": "🐸",
    "Kapha": "🦢",
}

PULSE_DESCRIPTIONS = {
    "Vata":  "Snake — Fast, feeble, irregular. High variability, low amplitude.",
    "Pitta": "Crow/Frog — Strong, high amplitude, forceful. Energetic pulse.",
    "Kapha": "Swan — Slow, deep, smooth. Low variability, stable rhythm.",
}

Y_AXIS_MIN = -700
Y_AXIS_MAX = 700
Y_AXIS_TICKS = list(range(Y_AXIS_MIN, Y_AXIS_MAX + 100, 100))
PLOT_DURATION_MS = 60000
SIGNAL_FIGSIZE_MAIN = (14, 10)
SIGNAL_FIGSIZE_PREVIEW = (14, 9)


def _model_exists(mdir: str) -> bool:
    return all(
        os.path.exists(os.path.join(mdir, f))
        for f in ["scaler.pkl", "pca.pkl", "kmeans.pkl", "model_summary.json"]
    )


def _capture_run(folder: str, save: bool = True, target_model_dir: str = MODEL_DIR):
    """Run the full workflow and capture logs in a string."""
    import io as _io
    from contextlib import redirect_stdout

    buf = _io.StringIO()
    with redirect_stdout(buf):
        workflow = PulseAnalysisWorkflow(folder_path=folder)
        results = workflow.run(save_model=False)
        artifacts = workflow.model_artifacts
        if save and artifacts is not None:
            ModelPersistence.save(artifacts, model_dir=target_model_dir)
    log = buf.getvalue()
    return results, artifacts, log


def _capture_run_uploaded_only(uploaded_file, save: bool, target_model_dir: str):
    """Run the full workflow only on the uploaded CSV file."""
    uploaded_name = Path(uploaded_file.name).name
    uploaded_patient_id = Path(uploaded_name).stem

    with tempfile.TemporaryDirectory() as tmp_dir:
        uploaded_file.seek(0)
        (Path(tmp_dir) / uploaded_name).write_bytes(uploaded_file.read())
        results, artifacts, log = _capture_run(tmp_dir, save=save, target_model_dir=target_model_dir)

    return results, artifacts, log, uploaded_patient_id


def _predict_single_patient(uploaded_file, model_dir: str):
    """Process the uploaded CSV with saved Scaler/PCA/KMeans and return rich prediction data."""
    import io as _io
    from contextlib import redirect_stdout

    patient_id = Path(uploaded_file.name).stem
    buf = _io.StringIO()

    with tempfile.TemporaryDirectory() as tmp_dir:
        uploaded_file.seek(0)
        tmp_path = Path(tmp_dir) / Path(uploaded_file.name).name
        tmp_path.write_bytes(uploaded_file.read())

        with redirect_stdout(buf):
            artifacts = ModelPersistence.load(model_dir)

            one = PulseAnalysisWorkflow._prepare_single_file(str(tmp_path))
            cleaned = PulseDataCleaner(one).clean()
            processor = PulseSignalProcessor(cleaned)
            filtered = processor.apply_filters(fs=100.0)
            peaks = processor.find_pulse_peaks(distance=30)
            rr = processor.calculate_rr_intervals(peaks)
            features = PulseFeatureExtractor(filtered, peaks, rr).extract_all_features()

    log = buf.getvalue()

    if features.empty:
        raise ValueError(
            "No features could be extracted from the uploaded file. "
            "Check that the file has enough valid rows after cleaning."
        )

    feat_indexed = features.set_index("patient_id")
    numeric = feat_indexed.select_dtypes(include=[np.number])
    feature_columns = artifacts["feature_columns"]
    for col in feature_columns:
        if col not in numeric.columns:
            numeric[col] = 0.0
    numeric = numeric[feature_columns].fillna(0.0)

    scaled = artifacts["scaler"].transform(numeric)
    pca_result = artifacts["pca"].transform(scaled)
    pred_cluster = int(artifacts["kmeans"].predict(pca_result)[0])
    mapping = artifacts["cluster_mapping"][pred_cluster]

    return {
        "patient_id": patient_id,
        "cluster": pred_cluster,
        "pulse_type": mapping["type"],
        "description": mapping["description"],
        "confidence_score": float(mapping.get("confidence_score", 0.0)),
        "filtered_data": filtered,
        "features_df": features,
        "peaks_df": peaks,
        "rr_df": rr,
        "cluster_mapping": artifacts["cluster_mapping"],
    }, log


def _save_uploaded_to_dataset(uploaded_file, dataset_folder: str) -> Path:
    """Save uploaded CSV to dataset folder, avoiding filename collisions."""
    os.makedirs(dataset_folder, exist_ok=True)

    source_name = Path(uploaded_file.name).name
    stem = Path(source_name).stem
    suffix = Path(source_name).suffix or ".csv"

    candidate = Path(dataset_folder) / f"{stem}{suffix}"
    index = 1
    while candidate.exists():
        candidate = Path(dataset_folder) / f"{stem}_{index}{suffix}"
        index += 1

    uploaded_file.seek(0)
    candidate.write_bytes(uploaded_file.read())
    uploaded_file.seek(0)
    return candidate


def _to_elapsed_ms(time_series: pd.Series) -> np.ndarray:
    """Convert time values to elapsed milliseconds from the first sample."""
    time_values = pd.to_numeric(time_series, errors="coerce")
    if time_values.isna().all():
        return np.arange(len(time_series), dtype=float)

    elapsed = (time_values - time_values.min()).to_numpy(dtype=float)
    if np.isnan(elapsed).any():
        return np.nan_to_num(elapsed, nan=0.0)
    return elapsed


def _plot_signals(filtered_data: pd.DataFrame, patient_id: str) -> plt.Figure:
    grp = filtered_data[filtered_data["patient_id"] == patient_id].sort_values("Time")
    elapsed_ms = _to_elapsed_ms(grp["Time"])

    fig, axes = plt.subplots(3, 1, figsize=SIGNAL_FIGSIZE_MAIN, sharex=True)
    for ax, col in zip(axes, SENSOR_COLUMNS):
        filt_col = f"{col}_filtered" if f"{col}_filtered" in grp.columns else col
        ax.plot(elapsed_ms, grp[filt_col].values, linewidth=0.8, color="#e74c3c")
        ax.set_ylabel(f"{col} Amplitude (a.u.)", fontsize=11, labelpad=12)
        ax.set_ylim(Y_AXIS_MIN, Y_AXIS_MAX)
        ax.set_yticks(Y_AXIS_TICKS)
        ax.tick_params(axis="y", labelsize=9, pad=5)
        ax.tick_params(axis="x", labelsize=9)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlim(0, PLOT_DURATION_MS)
    axes[-1].set_xlabel("Time (ms)", fontsize=11)
    fig.suptitle(f"Filtered Pulse Signal — {patient_id}", fontsize=16, y=0.98)
    fig.subplots_adjust(left=0.14, right=0.98, top=0.92, bottom=0.10, hspace=0.20)
    return fig


def _plot_pca_scatter(pca_features: np.ndarray, labels: np.ndarray, cluster_mapping: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 5))
    for cid in np.unique(labels):
        mask = labels == cid
        pulse_type = cluster_mapping[cid]["type"]
        color = PULSE_COLORS.get(pulse_type, "grey")
        icon = PULSE_ICONS.get(pulse_type, "●")
        ax.scatter(
            pca_features[mask, 0],
            pca_features[mask, 1] if pca_features.shape[1] > 1 else np.zeros(mask.sum()),
            c=color,
            label=f"Cluster {cid}: {icon} {pulse_type}",
            alpha=0.85,
            edgecolors="white",
            s=90,
        )
    ax.set_xlabel("PC1", fontsize=10)
    ax.set_ylabel("PC2", fontsize=10)
    ax.set_title("PCA — Patient Clusters", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def _plot_cluster_pie(cluster_mapping: dict) -> plt.Figure:
    labels_list, sizes, colors = [], [], []
    for cid, info in sorted(cluster_mapping.items()):
        pulse_type = info["type"]
        labels_list.append(f"{PULSE_ICONS.get(pulse_type, '')} {info['description']}\n(n={info['size']})")
        sizes.append(info["size"])
        colors.append(PULSE_COLORS.get(pulse_type, "grey"))

    fig, ax = plt.subplots(figsize=(6, 5))
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels_list,
        colors=colors,
        autopct="%1.0f%%",
        startangle=140,
        pctdistance=0.75,
    )
    for t in autotexts:
        t.set_fontsize(11)
        t.set_fontweight("bold")
        t.set_color("white")
    ax.set_title("Cluster Distribution", fontsize=12)
    plt.tight_layout()
    return fig


def _plot_feature_heatmap(scaled_features: pd.DataFrame) -> plt.Figure:
    numeric = scaled_features.select_dtypes(include=[np.number])
    top_cols = numeric.var().sort_values(ascending=False).head(20).index.tolist()
    if not top_cols:
        return plt.figure()
    corr = numeric[top_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(top_cols)))
    ax.set_xticklabels(top_cols, rotation=90, fontsize=7)
    ax.set_yticks(range(len(top_cols)))
    ax.set_yticklabels(top_cols, fontsize=7)
    ax.set_title("Feature Correlation (Top-20 by Variance)", fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.03)
    plt.tight_layout()
    return fig


# ------------------------------------------------------------------
# SESSION STATE keys
# ------------------------------------------------------------------
if "results" not in st.session_state:
    st.session_state["results"] = None
if "artifacts" not in st.session_state:
    st.session_state["artifacts"] = None


# ==================================================================
# PAGE: Home
# ==================================================================
if page == "🏠 Home":
    st.markdown(
        '<div class="pulse-header"><h2>👩🏻‍💻 Pulse Analysis System</h2>'
        '<p>Ayurvedic Pulse Classification using Machine Learning</p></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        This system analyzes multi-sensor pulse data and classifies each patient's pulse 
        pattern into one of three Ayurvedic types using unsupervised K-Means clustering.
        """
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            '<div class="pulse-type-vata"><h4>🐍 Vata — Snake</h4>'
            '<p style="color:#555">Fast, feeble, irregular pulse.<br>'
            'High variability · Low amplitude · Irregular rhythm.</p></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            '<div class="pulse-type-pitta"><h4>🐸 Pitta — Crow / Frog</h4>'
            '<p style="color:#555">Strong, high-amplitude, forceful pulse.<br>'
            'High amplitude · Energetic · Regular rhythm.</p></div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            '<div class="pulse-type-kapha"><h4>🦢 Kapha — Swan</h4>'
            '<p style="color:#555">Slow, deep, smooth pulse.<br>'
            'Low variability · Stable · Deep rhythm.</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    st.subheader("🔄 Analysis Pipeline")
    steps = [
        ("📁 Load CSVs", "Read all patient CSV files from the data folder."),
        ("🔍 Data Understanding", "Inspect structure and detect quality issues."),
        ("🧹 Data Cleaning", "Remove zeros, outliers; apply median filter."),
        ("📡 Signal Processing", "Bandpass filter (0.5 – 5 Hz) + peak detection."),
        ("📐 Feature Extraction", "Statistical · Frequency · Nadi variability · Pulse shape."),
        ("📊 EDA", "Correlation analysis and feature statistics."),
        ("⚖️ Scaling + PCA", "StandardScaler + dimensionality reduction."),
        ("🤖 K-Means (K=3)", "Cluster patients into 3 pulse groups."),
        ("🏷️ Pulse Mapping", "Assign Vata / Pitta / Kapha to each cluster."),
        ("💾 Save Model", "Persist scaler, PCA, and KMeans for future use."),
    ]
    for i, (title, desc) in enumerate(steps, 1):
        st.markdown(f"**{i}. {title}** — {desc}")

    st.markdown("---")
    st.subheader("📋 CSV Format Required")
    sample = pd.DataFrame({
        "Time": [0, 10, 20, 30],
        "S1":   [100, 102, 98, 101],
        "S2":   [95,  97,  93, 96],
        "S3":   [103, 105, 101, 104],
    })
    st.dataframe(sample, use_container_width=False)
    st.caption("Each CSV file represents one patient. Column names are case-sensitive.")


# ==================================================================
# PAGE: Train & Analyze
# ==================================================================
elif page == "🔬 Train & Analyze":
    st.markdown(
        '<div class="pulse-header"><h2>🔬 Train & Analyze</h2>'
        '<p>Run the full pipeline on your patient dataset</p></div>',
        unsafe_allow_html=True,
    )

    # ---- folder check ----
    folder_exists = os.path.isdir(data_folder)
    if not folder_exists:
        st.error(f"Folder not found: `{data_folder}`  \nUpdate the path in the sidebar.")
    else:
        csv_count = len(list(Path(data_folder).glob("*.csv")))
        st.info(f"📁 Found **{csv_count}** CSV file(s) in `{data_folder}`")

    col_btn, col_opt = st.columns([2, 1])
    with col_opt:
        save_model = st.checkbox("Save trained model", value=True)

    with col_btn:
        run_btn = st.button(
            "▶ Run Full Analysis",
            type="primary",
            disabled=not folder_exists,
            use_container_width=True,
        )

    if run_btn:
        with st.spinner("Running analysis pipeline … this may take a moment"):
            try:
                results, artifacts, log = _capture_run(data_folder, save=save_model, target_model_dir=model_dir)
                st.session_state["results"] = results
                st.session_state["artifacts"] = artifacts
                st.success("✅ Analysis complete!")
            except Exception as exc:
                st.error(f"❌ Pipeline failed: {exc}")
                st.stop()

        with st.expander("📋 Pipeline Log", expanded=False):
            st.code(log, language="text")

    # ---- Display results if available ----
    results: WorkflowResults = st.session_state.get("results")
    if results is None:
        st.info("Click **Run Full Analysis** above to start.")
        st.stop()

    # ---- Metrics ----
    st.markdown("---")
    st.subheader("📈 Clustering Metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Patients", results.features_df["patient_id"].nunique())
    m2.metric("Silhouette ↑", f"{results.metrics['silhouette']:.4f}")
    m3.metric("Davies-Bouldin ↓", f"{results.metrics['davies_bouldin']:.4f}")
    m4.metric("Calinski-Harabasz ↑", f"{results.metrics['calinski_harabasz']:.1f}")

    # ---- Cluster assignments table ----
    st.markdown("---")
    st.subheader("🏷️ Patient Cluster Assignments")
    cluster_rows = []
    for i, pid in enumerate(results.features_df["patient_id"].values):
        cid = int(results.labels[i])
        info = results.cluster_mapping[cid]
        pulse_type = info["type"]
        cluster_rows.append({
            "Patient": pid,
            "Cluster": cid,
            "Pulse Type": PULSE_ICONS.get(pulse_type, "") + " " + info["description"],
            "Confidence": f"{info['confidence_score']:.4f}",
        })

    df_clusters = pd.DataFrame(cluster_rows)
    st.dataframe(df_clusters, use_container_width=True, height=220)

    csv_bytes = df_clusters.to_csv(index=False).encode()
    st.download_button(
        "⬇ Download cluster_results.csv",
        data=csv_bytes,
        file_name="cluster_results.csv",
        mime="text/csv",
    )

    # ---- Visualizations ----
    st.markdown("---")
    st.subheader("📊 Visualizations")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🥧 Cluster Distribution", "🔵 PCA Scatter", "🌡️ Feature Correlation", "📡 Signal Waveform"]
    )

    with tab1:
        fig_pie = _plot_cluster_pie(results.cluster_mapping)
        st.pyplot(fig_pie, use_container_width=False)
        plt.close(fig_pie)

    with tab2:
        fig_pca = _plot_pca_scatter(results.pca_features, results.labels, results.cluster_mapping)
        st.pyplot(fig_pca, use_container_width=True)
        plt.close(fig_pca)

    with tab3:
        fig_heat = _plot_feature_heatmap(results.scaled_features)
        st.pyplot(fig_heat, use_container_width=True)
        plt.close(fig_heat)

    with tab4:
        patients = sorted(results.filtered_data["patient_id"].unique().tolist())
        sel_patient = st.selectbox("Select patient", patients)
        if sel_patient:
            fig_sig = _plot_signals(results.filtered_data, sel_patient)
            st.pyplot(fig_sig, use_container_width=True)
            plt.close(fig_sig)


# ==================================================================
# PAGE: Predict New Patient
# ==================================================================
elif page == "🔮 Predict New Patient":
    st.markdown(
        '<div class="pulse-header"><h2>🔮 Predict New Patient</h2>'
        '<p>Upload a single CSV file to classify its pulse type</p></div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload patient CSV file",
        type=["csv"],
        help="Supports flexible columns (for example: Time/Timestamp, Sensor1/2/3, S1/S2/S3)",
    )

    if uploaded_file is not None:
        try:
            preview_df = pd.read_csv(uploaded_file)
            uploaded_file.seek(0)  # reset for later use
        except Exception as e:
            st.error(f"Cannot read uploaded file: {e}")
            st.stop()

        st.subheader("📋 File Preview")
        st.dataframe(preview_df.head(10), use_container_width=True)

        normalized_preview = PulseDataCollector.normalize_input_schema(preview_df)
        missing_cols = [c for c in REQUIRED_COLUMNS if c not in normalized_preview.columns]
        if missing_cols:
            st.error(f"❌ File could not be mapped to required fields: `{missing_cols}`")
            st.stop()

        with st.expander("Mapped Columns", expanded=False):
            st.write("Detected schema mapped to canonical columns: Time, S1, S2, S3")
            st.dataframe(normalized_preview[REQUIRED_COLUMNS].head(10), use_container_width=True)

        # Option 1 requires a saved model
        model_exists = _model_exists(model_dir)

        col_btn1, col_btn2 = st.columns(2)

        run_uploaded_only_btn = col_btn1.button(
            "▶ Run Uploaded File Only",
            type="primary",
            use_container_width=True,
            disabled=not model_exists,
        )
        if not model_exists:
            col_btn1.caption(
                "⚠️ No saved model found. Train a model first in **Train & Analyze**."
            )

        save_after_run = col_btn2.checkbox(
            "Save model after full pipeline run",
            value=True,
            help="Applies to Option 2 only.",
        )
        run_all_btn = col_btn2.button(
            "▶ Add to Dataset & Run All Files",
            use_container_width=True,
        )

        # ------------------------------------------------------------------
        # Option 1: predict using saved model — no retraining
        # ------------------------------------------------------------------
        if run_uploaded_only_btn:
            with st.spinner("Running pipeline steps and predicting with saved models …"):
                try:
                    pred_result, log = _predict_single_patient(uploaded_file, model_dir)
                except Exception as exc:
                    st.error(f"❌ Prediction failed: {exc}")
                    st.stop()

            st.success("✅ Prediction complete using saved Scaler, PCA, and KMeans models.")
            with st.expander("📋 Pipeline Log", expanded=False):
                st.code(log, language="text")

            # --- Prediction Summary card ---
            st.markdown("---")
            st.subheader("🎯 Prediction Summary")
            p_type = pred_result["pulse_type"]
            p_icon = PULSE_ICONS.get(p_type, "●")
            p_color = PULSE_COLORS.get(p_type, "#555")
            p_desc = pred_result["description"]
            st.markdown(
                f"<div style='border-left:6px solid {p_color}; padding:1rem 1.5rem; "
                f"background:#f8f9fa; border-radius:8px;'>"
                f"<h3 style='color:{p_color}; margin:0'>{p_icon} {p_desc}</h3>"
                f"<p style='margin:0.4rem 0 0 0; color:#555;'>"
                f"Patient: <strong>{pred_result['patient_id']}</strong> &nbsp;|&nbsp; "
                f"Cluster: <strong>{pred_result['cluster']}</strong> &nbsp;|&nbsp; "
                f"Type: <strong>{p_type}</strong>"
                f"</p>"
                f"<p style='margin:0.3rem 0 0 0; color:#888; font-size:0.9rem'>"
                f"{PULSE_DESCRIPTIONS.get(p_type, '')}</p>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # --- Signal Waveform ---
            st.markdown("---")
            st.subheader("📡 Signal Waveform")
            fig_sig = _plot_signals(pred_result["filtered_data"], pred_result["patient_id"])
            st.pyplot(fig_sig, use_container_width=True)
            plt.close(fig_sig)

            # --- Signal Analysis Results ---
            st.markdown("---")
            st.subheader("📊 Signal Analysis Results")
            tab_peaks, tab_rr = st.tabs(["🔺 Detected Peaks", "⏱️ RR Intervals"])
            with tab_peaks:
                if pred_result["peaks_df"].empty:
                    st.info("No peaks detected for this patient.")
                else:
                    st.dataframe(pred_result["peaks_df"], use_container_width=True)
            with tab_rr:
                if pred_result["rr_df"].empty:
                    st.info("No RR intervals computed (not enough peaks).")
                else:
                    st.dataframe(pred_result["rr_df"], use_container_width=True)

        # ------------------------------------------------------------------
        # Option 2: save file to dataset, then run full clustering pipeline
        # ------------------------------------------------------------------
        elif run_all_btn:
            with st.spinner("Saving file and running full pipeline on all files …"):
                try:
                    if not os.path.isdir(data_folder):
                        os.makedirs(data_folder, exist_ok=True)

                    saved_csv = _save_uploaded_to_dataset(uploaded_file, data_folder)
                    uploaded_patient_id = saved_csv.stem
                    results, artifacts, log = _capture_run(
                        data_folder,
                        save=save_after_run,
                        target_model_dir=model_dir,
                    )
                except Exception as exc:
                    st.error(f"❌ Pipeline failed: {exc}")
                    st.stop()

            st.session_state["results"] = results
            st.session_state["artifacts"] = artifacts

            st.success(
                f"✅ Uploaded CSV saved as '{saved_csv.name}' and full pipeline completed on all files."
            )
            with st.expander("📋 Pipeline Log", expanded=False):
                st.code(log, language="text")

            st.markdown("---")
            st.subheader("📈 Full Pipeline Results")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Patients", results.features_df["patient_id"].nunique())
            m2.metric("Silhouette ↑", f"{results.metrics['silhouette']:.4f}")
            m3.metric("Davies-Bouldin ↓", f"{results.metrics['davies_bouldin']:.4f}")
            m4.metric("Calinski-Harabasz ↑", f"{results.metrics['calinski_harabasz']:.1f}")

            cluster_rows = []
            for i, pid in enumerate(results.features_df["patient_id"].values):
                cid = int(results.labels[i])
                info = results.cluster_mapping[cid]
                pulse_type = info["type"]
                cluster_rows.append({
                    "Patient": pid,
                    "Cluster": cid,
                    "Pulse Type": PULSE_ICONS.get(pulse_type, "") + " " + info["description"],
                    "Confidence": f"{info['confidence_score']:.4f}",
                })

            df_clusters = pd.DataFrame(cluster_rows)
            st.dataframe(df_clusters, use_container_width=True, height=260)

            uploaded_row = df_clusters[df_clusters["Patient"] == uploaded_patient_id]
            if not uploaded_row.empty:
                st.markdown("### 🎯 Newly Added Patient Result")
                row = uploaded_row.iloc[0]
                up_type = results.cluster_mapping[int(row["Cluster"])]["type"]
                up_icon = PULSE_ICONS.get(up_type, "●")
                up_color = PULSE_COLORS.get(up_type, "#555")
                st.markdown(
                    f"<div style='border-left:6px solid {up_color}; padding:0.8rem 1.2rem; "
                    f"background:#f8f9fa; border-radius:8px'>"
                    f"<strong style='color:{up_color}'>{up_icon} {row['Pulse Type']}</strong>"
                    f" &nbsp;|&nbsp; Patient: <strong>{row['Patient']}</strong>"
                    f" &nbsp;|&nbsp; Cluster: <strong>{row['Cluster']}</strong>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.warning(
                    "Uploaded patient is not present in final feature set. "
                    "This can happen if the file has too few valid rows after cleaning."
                )


# ==================================================================
# PAGE: View Results
# ==================================================================
elif page == "📊 View Results":
    st.markdown(
        '<div class="pulse-header"><h2>📊 View Results</h2>'
        '<p>Explore saved analysis outputs</p></div>',
        unsafe_allow_html=True,
    )

    # ---- Cluster results CSV ----
    cluster_csv_path = os.path.join(OUTPUT_DIR, "cluster_results.csv")
    eda_csv_path     = os.path.join(OUTPUT_DIR, "eda_summary.csv")
    corr_plot_path   = os.path.join(PLOTS_DIR,  "eda_correlation_top20.png")
    model_json_path  = os.path.join(model_dir,  "model_summary.json")

    tab_cr, tab_eda, tab_model = st.tabs(
        ["🏷️ Cluster Results", "📈 EDA Summary", "💾 Model Summary"]
    )

    with tab_cr:
        if os.path.exists(cluster_csv_path):
            df = pd.read_csv(cluster_csv_path)
            st.dataframe(df, use_container_width=True)

            st.download_button(
                "⬇ Download cluster_results.csv",
                data=df.to_csv(index=False).encode(),
                file_name="cluster_results.csv",
                mime="text/csv",
            )
        else:
            st.info(f"No file found at `{cluster_csv_path}`. Run the analysis first.")

    with tab_eda:
        if os.path.exists(eda_csv_path):
            eda_df = pd.read_csv(eda_csv_path, index_col=0)
            st.dataframe(eda_df, use_container_width=True)
        else:
            st.info(f"No file found at `{eda_csv_path}`.")

        if os.path.exists(corr_plot_path):
            st.image(corr_plot_path, caption="Feature Correlation Heatmap", use_container_width=True)
        else:
            st.info(f"No correlation plot found at `{corr_plot_path}`.")

    with tab_model:
        if os.path.exists(model_json_path):
            with open(model_json_path, encoding="utf-8") as f:
                summary = json.load(f)

            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Clusters", summary.get("n_clusters", "—"))
                st.metric("Features", summary.get("feature_count", "—"))
            with col_b:
                metrics = summary.get("metrics", {})
                if metrics:
                    st.metric("Silhouette ↑", f"{metrics.get('silhouette', 0):.4f}")
                    st.metric("Davies-Bouldin ↓", f"{metrics.get('davies_bouldin', 0):.4f}")

            st.subheader("Cluster Mapping")
            for cid, label in summary.get("cluster_mapping", {}).items():
                pulse_key = label.split("(")[-1].rstrip(")")
                icon = PULSE_ICONS.get(pulse_key, "●")
                color = PULSE_COLORS.get(pulse_key, "#555")
                st.markdown(
                    f"<span style='color:{color}; font-weight:bold'>"
                    f"{icon} Cluster {cid}: {label}</span>",
                    unsafe_allow_html=True,
                )

            st.subheader("Feature Columns")
            feat_cols = summary.get("feature_columns", [])
            st.write(f"Total: {len(feat_cols)} features")
            st.dataframe(pd.DataFrame({"feature": feat_cols}), use_container_width=True, height=200)

        else:
            st.info(f"No model summary found at `{model_json_path}`. Run analysis and save the model first.")
