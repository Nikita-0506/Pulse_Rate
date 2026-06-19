"""
Nadi Pulse Analysis - Streamlit UI
Ayurvedic Pulse Classification: Vata / Pitta / Kapha
"""

import io
import json
import os
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


def _model_exists(mdir: str) -> bool:
    return all(
        os.path.exists(os.path.join(mdir, f))
        for f in ["scaler.pkl", "pca.pkl", "kmeans.pkl", "model_summary.json"]
    )


def _capture_run(folder: str, save: bool = True):
    """Run the full workflow and capture logs in a string."""
    import io as _io
    from contextlib import redirect_stdout

    buf = _io.StringIO()
    with redirect_stdout(buf):
        workflow = PulseAnalysisWorkflow(folder_path=folder)
        results = workflow.run(save_model=save)
        artifacts = workflow.model_artifacts
    log = buf.getvalue()
    return results, artifacts, log


def _plot_signals(filtered_data: pd.DataFrame, patient_id: str) -> plt.Figure:
    grp = filtered_data[filtered_data["patient_id"] == patient_id].sort_values("Time")
    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    for ax, col in zip(axes, SENSOR_COLUMNS):
        filt_col = f"{col}_filtered" if f"{col}_filtered" in grp.columns else col
        ax.plot(grp["Time"].values, grp[filt_col].values, linewidth=0.8, color="#e74c3c")
        ax.set_ylabel(col, fontsize=9)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time")
    fig.suptitle(f"Filtered Pulse Signal — {patient_id}", fontsize=11)
    plt.tight_layout()
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
                results, artifacts, log = _capture_run(data_folder, save=save_model)
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

    if not _model_exists(model_dir):
        st.warning(
            f"⚠️ No trained model found in `{model_dir}`.  \n"
            "Go to **Train & Analyze** first to train and save a model."
        )
        st.stop()

    st.success(f"✅ Trained model loaded from `{model_dir}`")

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

        predict_btn = st.button("▶ Predict Pulse Type", type="primary")

        if predict_btn:
            with st.spinner("Processing …"):
                try:
                    # Save uploaded file to a temp file for the pipeline
                    with tempfile.NamedTemporaryFile(
                        suffix=".csv", delete=False, mode="wb"
                    ) as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name

                    workflow = PulseAnalysisWorkflow()
                    result = workflow.predict_new_csv(tmp_path, model_dir=model_dir)
                    os.unlink(tmp_path)

                except Exception as exc:
                    st.error(f"❌ Prediction failed: {exc}")
                    st.stop()

            pulse_type_key = result["pulse_type"].split("(")[-1].rstrip(")")  # e.g. "Vata"
            color = PULSE_COLORS.get(pulse_type_key, "#555")
            icon = PULSE_ICONS.get(pulse_type_key, "●")
            desc = PULSE_DESCRIPTIONS.get(pulse_type_key, "")

            st.markdown("---")
            st.subheader("🎯 Prediction Result")

            col_res, col_info = st.columns([1, 2])
            with col_res:
                st.markdown(
                    f"""
                    <div style="background:{color}22; border:2px solid {color};
                         border-radius:12px; padding:1.5rem; text-align:center;">
                        <div style="font-size:3rem">{icon}</div>
                        <div style="font-size:1.4rem; font-weight:bold; color:{color}">
                            {result['pulse_type']}
                        </div>
                        <div style="color:#777; margin-top:0.4rem">
                            Cluster {result['cluster']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col_info:
                st.markdown(f"**Patient:** `{result['patient_id']}`")
                st.markdown(f"**Pulse Type:** {icon} {result['pulse_type']}")
                st.markdown(f"**Description:** {desc}")
                st.markdown(f"**Assigned Cluster:** {result['cluster']}")

            # Signal waveform preview
            st.markdown("---")
            st.subheader("📡 Signal Waveform")
            try:
                uploaded_file.seek(0)
                preview_raw = pd.read_csv(uploaded_file)
                fig, axes = plt.subplots(3, 1, figsize=(10, 5), sharex=True)
                for ax, col in zip(axes, SENSOR_COLUMNS):
                    if col in preview_raw.columns:
                        ax.plot(preview_raw["Time"].values, preview_raw[col].values,
                                linewidth=0.8, color=color)
                        ax.set_ylabel(col, fontsize=9)
                        ax.grid(True, alpha=0.3)
                axes[-1].set_xlabel("Time")
                fig.suptitle(f"Raw Pulse Signal — {result['patient_id']}", fontsize=11)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            except Exception:
                pass


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
