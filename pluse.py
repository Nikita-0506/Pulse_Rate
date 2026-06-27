import glob
import io
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler


# =================================================
# Configuration
# =================================================

DEFAULT_DATA_FOLDER = r"C:\Users\ndpac\OneDrive\Desktop\Pulse_CSV"
MODEL_DIR = r"models"
OUTPUT_DIR = r"outputs"
PLOTS_DIR = r"outputs\plots"
REQUIRED_COLUMNS = ["Time", "S1", "S2", "S3"]
SENSOR_COLUMNS = ["S1", "S2", "S3"]
INPUT_COLUMN_ALIASES = {
    "Time": ["time", "timestamp", "sample", "sampleindex", "index", "t"],
    "S1": ["s1", "sensor1", "channel1", "ch1", "pulse1", "nadi1"],
    "S2": ["s2", "sensor2", "channel2", "ch2", "pulse2", "nadi2"],
    "S3": ["s3", "sensor3", "channel3", "ch3", "pulse3", "nadi3"],
}

# Create directories if they don't exist
import sys
for d in [MODEL_DIR, OUTPUT_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)


# =================================================
# CSV Loading
# =================================================


def read_pulse_csv(source, **kwargs) -> pd.DataFrame:
    """Read pulse CSVs even when metadata lines appear before the real header row."""
    if hasattr(source, "seek"):
        source.seek(0)

    if hasattr(source, "read"):
        raw = source.read()
        if hasattr(source, "seek"):
            source.seek(0)
        if isinstance(raw, bytes):
            text = raw.decode("utf-8-sig", errors="replace")
        else:
            text = str(raw)
    else:
        path = Path(source)
        text = path.read_text(encoding="utf-8-sig", errors="replace")

    lines = [line.rstrip("\r\n") for line in text.splitlines() if line.strip()]
    if not lines:
        return pd.DataFrame()

    candidate_separators = [",", ";", "\t", "|"]
    header_aliases = {
        "time",
        "s1",
        "s2",
        "s3",
        "sensor1",
        "sensor2",
        "sensor3",
        "channel1",
        "channel2",
        "channel3",
        "pulse1",
        "pulse2",
        "pulse3",
        "nadi1",
        "nadi2",
        "nadi3",
        "sample",
        "sampleindex",
        "index",
        "t",
    }

    for start_idx in range(len(lines)):
        for sep in candidate_separators:
            try:
                df = pd.read_csv(
                    io.StringIO("\n".join(lines[start_idx:])),
                    sep=sep,
                    engine="python",
                    skip_blank_lines=True,
                    **kwargs,
                )
            except Exception:
                continue

            if df.empty or df.shape[1] < 2:
                continue

            normalized_columns = [str(col).strip().lower() for col in df.columns]
            if any(alias in normalized_columns for alias in header_aliases):
                return df

    try:
        if hasattr(source, "read"):
            source.seek(0)
            return pd.read_csv(source, **kwargs)
        return pd.read_csv(source, **kwargs)
    except Exception:
        return pd.read_csv(io.StringIO(text), sep=None, engine="python", **kwargs)


# =================================================
# Data Collection
# =================================================


class PulseDataCollector:
    """Collect and aggregate pulse data from multiple CSV files."""

    def __init__(self, folder_path: str):
        self.folder_path = folder_path
        self.raw_data: Optional[pd.DataFrame] = None
        self.file_metadata: Dict[str, Dict] = {}

    def _validate_columns(self, df: pd.DataFrame, file_path: str) -> None:
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns {missing} in file: {file_path}")

    @staticmethod
    def _sanitize_name(name: str) -> str:
        return re.sub(r"[^a-z0-9]", "", str(name).lower())

    @classmethod
    def normalize_input_schema(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize variable patient CSV schemas to canonical columns: Time, S1, S2, S3."""
        out = df.copy()
        col_map = {c: cls._sanitize_name(c) for c in out.columns}
        reverse_map = {v: k for k, v in col_map.items()}

        picked: Dict[str, str] = {}
        for target, aliases in INPUT_COLUMN_ALIASES.items():
            for alias in aliases:
                if alias in reverse_map and reverse_map[alias] not in picked.values():
                    picked[target] = reverse_map[alias]
                    break

        numeric_cols = out.select_dtypes(include=[np.number]).columns.tolist()

        # If Time is missing, use a numeric index-like column when available, else create sequence.
        if "Time" not in picked:
            for c in numeric_cols:
                if c not in picked.values():
                    picked["Time"] = c
                    break
            if "Time" not in picked:
                out["Time"] = np.arange(len(out), dtype=float)
                picked["Time"] = "Time"

        # Fill missing sensor channels using remaining numeric columns.
        for target in ["S1", "S2", "S3"]:
            if target in picked:
                continue
            for c in numeric_cols:
                if c != picked["Time"] and c not in picked.values():
                    picked[target] = c
                    break

        rename_map = {source: target for target, source in picked.items() if source in out.columns}
        out = out.rename(columns=rename_map)

        # Ensure all required columns exist after mapping.
        for col in REQUIRED_COLUMNS:
            if col not in out.columns:
                out[col] = np.nan

        return out

    def load_all_files(self) -> pd.DataFrame:
        csv_files = sorted(
            str(p) for p in Path(self.folder_path).iterdir() if p.is_file() and p.suffix.lower() == ".csv"
        )
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in folder: {self.folder_path}")

        print(f"Found {len(csv_files)} CSV files")
        all_frames: List[pd.DataFrame] = []

        for file_index, file_path in enumerate(csv_files):
            patient_id = Path(file_path).stem

            try:
                df = read_pulse_csv(file_path)
            except Exception as e:
                print(f"  [SKIP] Could not read {Path(file_path).name}: {e}")
                continue

            if df.empty:
                print(f"  [SKIP] No data rows in: {Path(file_path).name}")
                continue

            original_columns = df.columns.tolist()
            df = self.normalize_input_schema(df)
            self._validate_columns(df, file_path)

            df = df.copy()
            for col in REQUIRED_COLUMNS:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df["patient_id"] = patient_id
            df["file_index"] = file_index

            self.file_metadata[patient_id] = {
                "file_path": file_path,
                "rows": int(len(df)),
                "original_columns": original_columns,
                "columns": df.columns.tolist(),
                "time_min": float(df["Time"].min()) if not df["Time"].isna().all() else np.nan,
                "time_max": float(df["Time"].max()) if not df["Time"].isna().all() else np.nan,
            }

            all_frames.append(df)

        if not all_frames:
            raise ValueError("No valid CSV files could be loaded. Check that files are non-empty and have required columns.")

        self.raw_data = pd.concat(all_frames, ignore_index=True)
        print(
            f"Combined dataset created: {len(self.raw_data)} rows, "
            f"{self.raw_data['patient_id'].nunique()} patients"
        )
        return self.raw_data

    def get_summary(self) -> pd.DataFrame:
        if not self.file_metadata:
            return pd.DataFrame()
        return pd.DataFrame(self.file_metadata).T.sort_index()


# =================================================
# Data Understanding
# =================================================


class PulseDataUnderstanding:
    """Inspect structure and quality of pulse data."""

    def __init__(self, data: pd.DataFrame):
        self.data = data

    def inspect_structure(self) -> None:
        print("\nDATA UNDERSTANDING")
        print("=" * 60)
        print(f"Rows: {len(self.data):,}")
        print(f"Patients: {self.data['patient_id'].nunique()}")
        print(f"Columns: {self.data.columns.tolist()}")

        print("\nSensor statistics")
        for col in SENSOR_COLUMNS:
            series = self.data[col]
            print(
                f"{col}: mean={series.mean():.3f}, std={series.std():.3f}, "
                f"min={series.min():.3f}, max={series.max():.3f}, "
                f"zero_pct={(series == 0).mean() * 100:.2f}%"
            )

    def detect_quality_issues(self) -> Dict[str, Dict]:
        issues: Dict[str, Dict] = {}

        zero_mask = (self.data["S1"] == 0) & (self.data["S2"] == 0) & (self.data["S3"] == 0)
        issues["zero_records"] = {
            "count": int(zero_mask.sum()),
            "percentage": float(zero_mask.mean() * 100),
        }

        for col in SENSOR_COLUMNS:
            neg_count = int((self.data[col] < 0).sum())
            if neg_count > 0:
                issues[f"negative_{col}"] = {"count": neg_count}

        duplicates = self.data.duplicated(subset=["patient_id", "Time"]).sum()
        if duplicates > 0:
            issues["duplicate_patient_time"] = {"count": int(duplicates)}

        print("\nQuality issues")
        if not issues:
            print("No issues detected")
        else:
            for k, v in issues.items():
                print(f"- {k}: {v}")

        return issues


# =================================================
# Data Cleaning
# =================================================


class PulseDataCleaner:
    """Clean pulse signals before signal processing."""

    def __init__(self, data: pd.DataFrame):
        self.raw_data = data
        self.cleaned_data: Optional[pd.DataFrame] = None

    def clean(self) -> pd.DataFrame:
        print("\nDATA CLEANING")
        print("=" * 60)

        df = self.raw_data.copy()

        # Remove invalid rows in required columns.
        df = df.dropna(subset=REQUIRED_COLUMNS + ["patient_id"]).copy()

        zero_mask = (df["S1"] == 0) & (df["S2"] == 0) & (df["S3"] == 0)
        removed_zero = int(zero_mask.sum())
        df = df.loc[~zero_mask].copy()
        print(f"Removed zero-only rows: {removed_zero}")

        before = len(df)
        df = df.drop_duplicates(subset=["patient_id", "Time"]).copy()
        print(f"Removed duplicates: {before - len(df)}")

        for col in SENSOR_COLUMNS:
            q1, q3 = df[col].quantile(0.01), df[col].quantile(0.99)
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            capped_count = int(((df[col] < lower) | (df[col] > upper)).sum())
            df[col] = df[col].clip(lower=lower, upper=upper)
            print(f"Capped outliers in {col}: {capped_count}")

        df = df.sort_values(["patient_id", "Time"]).reset_index(drop=True)

        for col in SENSOR_COLUMNS:
            df[col] = df.groupby("patient_id")[col].transform(
                lambda x: x.rolling(window=3, min_periods=1, center=True).median()
            )

        for col in SENSOR_COLUMNS:
            df[col] = df.groupby("patient_id")[col].transform(
                lambda x: x.interpolate(method="linear", limit_direction="both")
            )
            df[col] = df[col].clip(lower=0)

        self.cleaned_data = df
        print(f"Cleaning complete: {len(df)} rows")
        return df


# =================================================
# Signal Processing
# =================================================


class PulseSignalProcessor:
    """Apply signal filters and detect peaks for pulse analysis."""

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.processed_data: Optional[pd.DataFrame] = None

    @staticmethod
    def butter_bandpass(lowcut: float, highcut: float, fs: float, order: int = 4):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        return butter(order, [low, high], btype="band")

    def apply_filters(self, fs: float = 100.0) -> pd.DataFrame:
        print("\nSIGNAL PROCESSING")
        print("=" * 60)

        df = self.data.copy()
        b, a = self.butter_bandpass(0.5, 5.0, fs)

        for col in SENSOR_COLUMNS:
            filtered_col = f"{col}_filtered"

            def _filt(x: pd.Series):
                arr = x.values.astype(float)
                if len(arr) < 12:
                    return arr
                return filtfilt(b, a, arr)

            df[filtered_col] = df.groupby("patient_id")[col].transform(_filt)

        self.processed_data = df
        print("Bandpass filter applied: 0.5-5.0 Hz")
        return df

    def find_pulse_peaks(self, signal_col: str = "S1_filtered", distance: int = 30) -> pd.DataFrame:
        if self.processed_data is None:
            raise ValueError("Run apply_filters before find_pulse_peaks")

        peaks_rows: List[Dict] = []

        for patient_id, group in self.processed_data.groupby("patient_id"):
            group = group.sort_values("Time")
            signal_data = group[signal_col].values
            std = float(np.std(signal_data))
            if len(signal_data) < 5 or std == 0:
                continue

            peaks, _ = find_peaks(
                signal_data,
                height=float(np.mean(signal_data) + 0.5 * std),
                prominence=float(0.3 * std),
                distance=distance,
                width=1,
            )

            for p in peaks:
                peaks_rows.append(
                    {
                        "patient_id": patient_id,
                        "peak_time": float(group.iloc[p]["Time"]),
                        "peak_height": float(signal_data[p]),
                        "peak_position": int(p),
                    }
                )

        peaks_df = pd.DataFrame(peaks_rows)
        print(f"Detected pulse peaks: {len(peaks_df)}")
        return peaks_df

    @staticmethod
    def calculate_rr_intervals(peaks_df: pd.DataFrame) -> pd.DataFrame:
        if peaks_df.empty:
            return pd.DataFrame(columns=["patient_id", "rr_interval_ms", "nadi_rate_bpm", "heart_rate_bpm"])

        rr_rows: List[Dict] = []
        for patient_id, grp in peaks_df.groupby("patient_id"):
            grp = grp.sort_values("peak_time")
            times = grp["peak_time"].values
            rr = np.diff(times)
            valid = rr[(rr > 300) & (rr < 2000)]
            for x in valid:
                rr_rows.append(
                    {
                        "patient_id": patient_id,
                        "rr_interval_ms": float(x),
                        "nadi_rate_bpm": float(60000.0 / x),
                        "heart_rate_bpm": float(60000.0 / x),
                    }
                )

        return pd.DataFrame(rr_rows)


# =================================================
# Feature Extraction
# =================================================


class PulseFeatureExtractor:
    """Extract statistical, frequency, Nadi variability, and pulse-shape features."""

    def __init__(self, data: pd.DataFrame, peaks_df: pd.DataFrame, rr_df: pd.DataFrame):
        self.data = data
        self.peaks = peaks_df
        self.rr = rr_df

    def extract_statistical_features(self) -> pd.DataFrame:
        rows: List[Dict] = []
        for patient_id, grp in self.data.groupby("patient_id"):
            row: Dict = {"patient_id": patient_id}
            for col in SENSOR_COLUMNS:
                source_col = f"{col}_filtered" if f"{col}_filtered" in grp.columns else col
                s = grp[source_col].values
                mean = float(np.mean(s))
                std = float(np.std(s))
                row[f"{col}_mean"] = mean
                row[f"{col}_std"] = std
                row[f"{col}_median"] = float(np.median(s))
                row[f"{col}_min"] = float(np.min(s))
                row[f"{col}_max"] = float(np.max(s))
                row[f"{col}_range"] = float(np.max(s) - np.min(s))
                row[f"{col}_cv"] = float(std / mean) if mean > 0 else 0.0
                row[f"{col}_skew"] = float(stats.skew(s)) if len(s) > 2 else 0.0
                row[f"{col}_kurtosis"] = float(stats.kurtosis(s)) if len(s) > 3 else 0.0
                row[f"{col}_p25"] = float(np.percentile(s, 25))
                row[f"{col}_p75"] = float(np.percentile(s, 75))
                row[f"{col}_p90"] = float(np.percentile(s, 90))
            rows.append(row)
        return pd.DataFrame(rows)

    def extract_frequency_features(self) -> pd.DataFrame:
        rows: List[Dict] = []
        for patient_id, grp in self.data.groupby("patient_id"):
            row: Dict = {"patient_id": patient_id}
            for col in SENSOR_COLUMNS:
                source_col = f"{col}_filtered" if f"{col}_filtered" in grp.columns else col
                s = grp[source_col].values
                if len(s) < 12:
                    continue

                fft_vals = np.abs(fft(s))
                freqs = fftfreq(len(s), d=1.0)
                mask = freqs > 0
                fvals = fft_vals[mask]
                f = freqs[mask]
                if len(fvals) == 0:
                    continue

                row[f"{col}_fft_mean"] = float(np.mean(fvals))
                row[f"{col}_fft_std"] = float(np.std(fvals))
                row[f"{col}_fft_max"] = float(np.max(fvals))
                row[f"{col}_dominant_freq"] = float(f[np.argmax(fvals)])

                power = fvals ** 2
                p = power / (np.sum(power) + 1e-12)
                row[f"{col}_spectral_entropy"] = float(-np.sum(p * np.log2(p + 1e-12)))
            rows.append(row)
        return pd.DataFrame(rows)

    def extract_nadi_variability_features(self) -> pd.DataFrame:
        if self.rr.empty:
            return pd.DataFrame(columns=["patient_id"])

        rows: List[Dict] = []
        for patient_id, grp in self.rr.groupby("patient_id"):
            rr = grp["rr_interval_ms"].values
            nadi_rate = grp["nadi_rate_bpm"].values if "nadi_rate_bpm" in grp.columns else grp["heart_rate_bpm"].values
            if len(rr) < 5:
                continue

            diff_rr = np.diff(rr)
            row = {
                "patient_id": patient_id,
                "nadi_rate_mean": float(np.mean(nadi_rate)),
                "nadi_rate_std": float(np.std(nadi_rate)),
                "nadi_rate_min": float(np.min(nadi_rate)),
                "nadi_rate_max": float(np.max(nadi_rate)),
                "heart_rate_mean": float(np.mean(nadi_rate)),
                "heart_rate_std": float(np.std(nadi_rate)),
                "heart_rate_min": float(np.min(nadi_rate)),
                "heart_rate_max": float(np.max(nadi_rate)),
                "rr_mean": float(np.mean(rr)),
                "rr_std": float(np.std(rr)),
                "rr_cv": float(np.std(rr) / np.mean(rr)) if np.mean(rr) > 0 else 0.0,
                "rmssd": float(np.sqrt(np.mean(diff_rr ** 2))) if len(diff_rr) else 0.0,
                "pnn50": float(np.mean(np.abs(diff_rr) > 50) * 100) if len(diff_rr) else 0.0,
                "stress_index": float(np.std(rr) / (np.mean(rr) * np.median(rr) + 1e-12)),
                "total_power": float(np.sum(rr ** 2)),
            }
            rows.append(row)

        return pd.DataFrame(rows)

    def extract_hrv_features(self) -> pd.DataFrame:
        """Backward-compatible alias for legacy HRV naming."""
        return self.extract_nadi_variability_features()

    def extract_pulse_shape_features(self) -> pd.DataFrame:
        if self.peaks.empty:
            return pd.DataFrame(columns=["patient_id"])

        rows: List[Dict] = []
        for patient_id, grp in self.peaks.groupby("patient_id"):
            grp = grp.sort_values("peak_time")
            if len(grp) < 3:
                continue

            intervals = np.diff(grp["peak_time"].values)
            valid = intervals[(intervals > 200) & (intervals < 1500)]
            h = grp["peak_height"].values
            mean_h = float(np.mean(h))

            row = {
                "patient_id": patient_id,
                "avg_peak_height": mean_h,
                "peak_height_cv": float(np.std(h) / mean_h) if mean_h > 0 else 0.0,
                "num_peaks": int(len(grp)),
                "avg_peak_interval": float(np.mean(valid)) if len(valid) else 0.0,
                "peak_interval_cv": float(np.std(valid) / np.mean(valid)) if len(valid) and np.mean(valid) > 0 else 0.0,
                "pulse_duration_ms": float(grp["peak_time"].max() - grp["peak_time"].min()),
            }
            rows.append(row)

        return pd.DataFrame(rows)

    def extract_all_features(self) -> pd.DataFrame:
        print("\nFEATURE EXTRACTION")
        print("=" * 60)

        dfs = [
            self.extract_statistical_features(),
            self.extract_frequency_features(),
            self.extract_nadi_variability_features(),
            self.extract_pulse_shape_features(),
        ]

        merged = dfs[0]
        for df in dfs[1:]:
            if not df.empty:
                merged = merged.merge(df, on="patient_id", how="left")

        print(f"Feature dataset shape: {merged.shape}")
        return merged


# =================================================
# EDA
# =================================================


class PulseEDA:
    """Minimal but useful EDA outputs for feature dataset."""

    @staticmethod
    def run(features_df: pd.DataFrame, output_dir: str = OUTPUT_DIR, plots_dir: str = PLOTS_DIR) -> None:
        print("\nEDA")
        print("=" * 60)

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)

        numeric = features_df.drop(columns=["patient_id"], errors="ignore").select_dtypes(include=[np.number])
        if numeric.empty:
            print("No numeric features available for EDA")
            return

        desc_path = os.path.join(output_dir, "eda_summary.csv")
        numeric.describe().T.to_csv(desc_path, index=True)
        print(f"Saved EDA summary: {desc_path}")

        top_cols = numeric.var().sort_values(ascending=False).head(20).index.tolist()
        if len(top_cols) > 0:
            # Keep fewer columns for readability and a clean beginner experience.
            display_cols = top_cols[:8]
            corr_display = numeric[display_cols].corr()
            variance_display = numeric[top_cols[:10]].var().sort_values(ascending=True)

            # Find strongest feature-pair relationships (excluding diagonal).
            pairs = []
            for i in range(len(display_cols)):
                for j in range(i + 1, len(display_cols)):
                    val = float(corr_display.iloc[i, j])
                    pairs.append((display_cols[i], display_cols[j], val, abs(val)))
            pairs = sorted(pairs, key=lambda x: x[3], reverse=True)[:8]

            fig = plt.figure(figsize=(16, 9))
            gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1.3], height_ratios=[1.0, 0.7])
            ax_bar = fig.add_subplot(gs[:, 0])
            ax_heat = fig.add_subplot(gs[0, 1])
            ax_text = fig.add_subplot(gs[1, 1])

            # Panel 1: Top variability features.
            ax_bar.barh(
                variance_display.index,
                variance_display.values,
                color="#1F77B4",
                alpha=0.92,
            )
            ax_bar.set_title("Most Informative Features (by Variability)", fontsize=12, fontweight="bold")
            ax_bar.set_xlabel("Variance (higher = more variation across patients)")
            ax_bar.grid(axis="x", linestyle="--", alpha=0.3)

            # Panel 2: Compact annotated correlation map.
            short_labels = [c.replace("_", " ") for c in corr_display.columns]
            heat = ax_heat.imshow(corr_display.values, cmap="RdBu_r", vmin=-1, vmax=1)
            ax_heat.set_title("Correlation Map (Top-8 Features)", fontsize=12, fontweight="bold")
            ax_heat.set_xticks(range(len(short_labels)))
            ax_heat.set_yticks(range(len(short_labels)))
            ax_heat.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)
            ax_heat.set_yticklabels(short_labels, fontsize=8)

            for i in range(len(short_labels)):
                for j in range(len(short_labels)):
                    value = corr_display.values[i, j]
                    text_color = "white" if abs(value) >= 0.6 else "black"
                    ax_heat.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7, color=text_color)

            cbar = fig.colorbar(heat, ax=ax_heat, fraction=0.046, pad=0.04)
            cbar.set_label("Correlation Coefficient", rotation=90)

            # Panel 3: Human-readable summary for beginners.
            ax_text.axis("off")
            ax_text.set_title("Strongest Feature Relationships", fontsize=12, fontweight="bold", loc="left")
            if pairs:
                lines = []
                for idx, (a, b, val, _) in enumerate(pairs, start=1):
                    direction = "move together" if val >= 0 else "move opposite"
                    lines.append(
                        f"{idx}. {a.replace('_', ' ')} ↔ {b.replace('_', ' ')}: {val:+.2f} ({direction})"
                    )
                ax_text.text(0.0, 0.95, "\n".join(lines), va="top", fontsize=10)
            else:
                ax_text.text(0.0, 0.95, "Not enough features for pairwise correlation summary.", va="top", fontsize=10)

            fig.suptitle(
                "Nadi Pulse Feature Dashboard (Beginner-Friendly)",
                fontsize=14,
                fontweight="bold",
            )
            fig.text(
                0.5,
                0.01,
                "Left: features that vary most across patients | Top-right: correlation map | Bottom-right: plain-language relationship summary",
                ha="center",
                fontsize=10,
                color="#444444",
            )
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])

            corr_path = os.path.join(plots_dir, "eda_correlation_top20.png")
            fig.savefig(corr_path, dpi=180)
            plt.close(fig)
            print(f"Saved EDA plot: {corr_path}")


# =================================================
# Feature Scaling + PCA
# =================================================


class FeaturePreprocessor:
    """Clean feature matrix, scale data, and apply PCA."""

    def __init__(self, features_df: pd.DataFrame):
        self.features_df = features_df
        self.cleaned_features: Optional[pd.DataFrame] = None
        self.scaled_features: Optional[pd.DataFrame] = None
        self.pca_features: Optional[np.ndarray] = None
        self.scaler: Optional[StandardScaler] = None
        self.pca: Optional[PCA] = None

    def clean_features(self, na_threshold: float = 0.3) -> pd.DataFrame:
        df = self.features_df.copy()
        if "patient_id" not in df.columns:
            raise ValueError("Feature dataframe must contain patient_id")

        df = df.set_index("patient_id")
        df = df.select_dtypes(include=[np.number])
        df = df.loc[:, df.isna().mean() < na_threshold]

        for col in df.columns:
            df[col] = df[col].fillna(df[col].median())

        df = df.loc[:, df.std(ddof=0) > 0]
        if df.empty:
            raise ValueError("No usable features remain after cleaning")

        self.cleaned_features = df
        print(f"Cleaned feature matrix shape: {df.shape}")
        return df

    def scale(self) -> pd.DataFrame:
        print("\nFEATURE SCALING")
        print("=" * 60)

        if self.cleaned_features is None:
            self.clean_features()

        self.scaler = StandardScaler()
        scaled = self.scaler.fit_transform(self.cleaned_features)
        self.scaled_features = pd.DataFrame(
            scaled,
            index=self.cleaned_features.index,
            columns=self.cleaned_features.columns,
        )
        print(f"Scaled feature matrix shape: {self.scaled_features.shape}")
        return self.scaled_features

    def apply_pca(self, variance_threshold: float = 0.90) -> np.ndarray:
        print("\nPCA")
        print("=" * 60)

        if self.scaled_features is None:
            self.scale()

        pca_probe = PCA()
        pca_probe.fit(self.scaled_features)
        cumulative = np.cumsum(pca_probe.explained_variance_ratio_)
        n_components = int(np.argmax(cumulative >= variance_threshold) + 1)

        self.pca = PCA(n_components=n_components, random_state=42)
        self.pca_features = self.pca.fit_transform(self.scaled_features)

        print(
            f"PCA components: {n_components}, "
            f"explained variance: {np.sum(self.pca.explained_variance_ratio_) * 100:.2f}%"
        )
        return self.pca_features


# =================================================
# K-Means Clustering (K=3)
# =================================================


class PulseClusterer:
    """Cluster patient features and map clusters to pulse patterns."""

    def __init__(self, scaled_features: pd.DataFrame, pca_features: np.ndarray):
        self.scaled_features = scaled_features
        self.pca_features = pca_features
        self.labels: Optional[np.ndarray] = None
        self.model: Optional[KMeans] = None

    def perform_kmeans(self, n_clusters: int = 3) -> Tuple[np.ndarray, KMeans, Dict[str, float]]:
        print("\nK-MEANS (K=3)")
        print("=" * 60)

        x = self.pca_features if self.pca_features is not None else self.scaled_features.values
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
        labels = model.fit_predict(x)

        metrics = {
            "silhouette": float(silhouette_score(x, labels)),
            "davies_bouldin": float(davies_bouldin_score(x, labels)),
            "calinski_harabasz": float(calinski_harabasz_score(x, labels)),
        }

        print(f"Silhouette score: {metrics['silhouette']:.4f}")
        print(f"Davies-Bouldin score: {metrics['davies_bouldin']:.4f}")
        print(f"Calinski-Harabasz score: {metrics['calinski_harabasz']:.4f}")
        print(f"Cluster sizes: {np.bincount(labels)}")

        self.labels = labels
        self.model = model
        return labels, model, metrics

    def cluster_profiles(self) -> Dict[int, Dict]:
        if self.labels is None:
            raise ValueError("Run perform_kmeans before cluster_profiles")

        profiles: Dict[int, Dict] = {}
        overall_mean = self.scaled_features.mean()

        for c in sorted(np.unique(self.labels)):
            mask = self.labels == c
            frame = self.scaled_features.iloc[mask]
            mean = frame.mean()
            std = frame.std(ddof=0)
            importance = ((mean - overall_mean).abs() / (self.scaled_features.std(ddof=0) + 1e-12)).sort_values(ascending=False)

            profiles[int(c)] = {
                "size": int(mask.sum()),
                "mean": mean,
                "std": std,
                "top_features": importance.head(5).to_dict(),
            }

        return profiles

    def map_to_pulse_types(self, profiles: Dict[int, Dict]) -> Dict[int, Dict]:
        mapping_rules = {
            "Vata": "Snake (Vata)",
            "Pitta": "Crow/Frog (Pitta)",
            "Kapha": "Swan (Kapha)",
        }

        scores: Dict[int, Dict[str, float]] = {}
        for c, p in profiles.items():
            mean = p["mean"]

            variability = float(np.mean([mean.get("S1_std", 0), mean.get("S2_std", 0), mean.get("S3_std", 0)]))
            amplitude = float(np.mean([mean.get("S1_mean", 0), mean.get("S2_mean", 0), mean.get("S3_mean", 0)]))
            stability = -variability

            scores[c] = {
                "Vata": variability,
                "Pitta": amplitude,
                "Kapha": stability,
            }

        remaining_clusters = set(scores.keys())
        final: Dict[int, Dict] = {}

        for pulse in ["Pitta", "Vata", "Kapha"]:
            best_cluster = max(remaining_clusters, key=lambda c: scores[c][pulse])
            remaining_clusters.remove(best_cluster)
            confidence_raw = scores[best_cluster][pulse]

            final[best_cluster] = {
                "type": pulse,
                "description": mapping_rules[pulse],
                "confidence_score": float(confidence_raw),
                "size": profiles[best_cluster]["size"],
            }

        return final


# =================================================
# Model Persistence
# =================================================


class ModelPersistence:
    """Save and load trained pipeline artifacts."""

    @staticmethod
    def _extract_pulse_type(description: str) -> str:
        match = re.search(r"\(([^)]+)\)", str(description))
        if match:
            return match.group(1).strip()
        return str(description).strip()

    @classmethod
    def _normalize_cluster_mapping(cls, raw_mapping: Dict) -> Dict[int, Dict]:
        normalized: Dict[int, Dict] = {}

        for key, value in raw_mapping.items():
            cluster_id = int(key)

            if isinstance(value, dict):
                description = str(value.get("description") or value.get("label") or "")
                pulse_type = str(value.get("type") or cls._extract_pulse_type(description))
                confidence_score = float(value.get("confidence_score", 0.0))
                size = int(value.get("size", 0))
            else:
                description = str(value)
                pulse_type = cls._extract_pulse_type(description)
                confidence_score = 0.0
                size = 0

            normalized[cluster_id] = {
                "type": pulse_type,
                "description": description,
                "confidence_score": confidence_score,
                "size": size,
            }

        return normalized

    @staticmethod
    def save(artifacts: Dict, model_dir: str = MODEL_DIR) -> Dict[str, str]:
        os.makedirs(model_dir, exist_ok=True)
        
        scaler_path = os.path.join(model_dir, "scaler.pkl")
        pca_path = os.path.join(model_dir, "pca.pkl")
        kmeans_path = os.path.join(model_dir, "kmeans.pkl")
        summary_path = os.path.join(model_dir, "model_summary.json")
        
        joblib.dump(artifacts["scaler"], scaler_path)
        joblib.dump(artifacts["pca"], pca_path)
        joblib.dump(artifacts["kmeans"], kmeans_path)

        summary = {
            "n_clusters": 3,
            "feature_count": len(artifacts["feature_columns"]),
            "cluster_mapping": {
                str(k): v["description"] for k, v in artifacts["cluster_mapping"].items()
            },
            "feature_columns": artifacts["feature_columns"],
            "metrics": artifacts["metrics"],
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"Saved scaler: {scaler_path}")
        print(f"Saved PCA: {pca_path}")
        print(f"Saved KMeans: {kmeans_path}")
        print(f"Saved summary: {summary_path}")
        
        return {
            "scaler_path": scaler_path,
            "pca_path": pca_path,
            "kmeans_path": kmeans_path,
            "summary_path": summary_path,
        }

    @staticmethod
    def load(model_dir: str = MODEL_DIR) -> Dict:
        scaler_path = os.path.join(model_dir, "scaler.pkl")
        pca_path = os.path.join(model_dir, "pca.pkl")
        kmeans_path = os.path.join(model_dir, "kmeans.pkl")
        summary_path = os.path.join(model_dir, "model_summary.json")
        
        scaler = joblib.load(scaler_path)
        pca = joblib.load(pca_path)
        kmeans = joblib.load(kmeans_path)
        
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        raw_mapping = summary.get("cluster_mapping", {})
        cluster_mapping = ModelPersistence._normalize_cluster_mapping(raw_mapping)
        
        artifacts = {
            "scaler": scaler,
            "pca": pca,
            "kmeans": kmeans,
            "feature_columns": summary["feature_columns"],
            "cluster_mapping": cluster_mapping,
            "metrics": summary["metrics"],
        }
        
        print(f"Loaded model from {model_dir}")
        return artifacts


# =================================================
# End-to-End Workflow
# =================================================


@dataclass
class WorkflowResults:
    summary: pd.DataFrame
    raw_data: pd.DataFrame
    cleaned_data: pd.DataFrame
    filtered_data: pd.DataFrame
    peaks_df: pd.DataFrame
    rr_df: pd.DataFrame
    features_df: pd.DataFrame
    scaled_features: pd.DataFrame
    pca_features: np.ndarray
    labels: np.ndarray
    cluster_profiles: Dict[int, Dict]
    cluster_mapping: Dict[int, Dict]
    metrics: Dict[str, float]


class PulseAnalysisWorkflow:
    """Run complete architecture: load -> understand -> clean -> process -> features -> EDA -> scale -> PCA -> KMeans -> map -> validate -> save."""

    def __init__(self, folder_path: str = DEFAULT_DATA_FOLDER):
        self.folder_path = folder_path
        self.results: Optional[WorkflowResults] = None
        self.model_artifacts: Optional[Dict] = None

    def run(self, save_model: bool = True) -> WorkflowResults:
        print("\n" + "=" * 60)
        print("PULSE ANALYSIS WORKFLOW")
        print("=" * 60)

        collector = PulseDataCollector(self.folder_path)
        raw = collector.load_all_files()
        summary = collector.get_summary()

        understanding = PulseDataUnderstanding(raw)
        understanding.inspect_structure()
        understanding.detect_quality_issues()

        cleaner = PulseDataCleaner(raw)
        cleaned = cleaner.clean()

        processor = PulseSignalProcessor(cleaned)
        filtered = processor.apply_filters(fs=100.0)
        peaks = processor.find_pulse_peaks(distance=30)
        rr = processor.calculate_rr_intervals(peaks)

        extractor = PulseFeatureExtractor(filtered, peaks, rr)
        features = extractor.extract_all_features()

        PulseEDA.run(features, output_dir=OUTPUT_DIR, plots_dir=PLOTS_DIR)

        pre = FeaturePreprocessor(features)
        scaled = pre.scale()
        pca = pre.apply_pca(variance_threshold=0.90)

        clusterer = PulseClusterer(scaled, pca)
        labels, model, metrics = clusterer.perform_kmeans(n_clusters=3)
        profiles = clusterer.cluster_profiles()
        mapping = clusterer.map_to_pulse_types(profiles)

        self.results = WorkflowResults(
            summary=summary,
            raw_data=raw,
            cleaned_data=cleaned,
            filtered_data=filtered,
            peaks_df=peaks,
            rr_df=rr,
            features_df=features,
            scaled_features=scaled,
            pca_features=pca,
            labels=labels,
            cluster_profiles=profiles,
            cluster_mapping=mapping,
            metrics=metrics,
        )

        self.model_artifacts = {
            "scaler": pre.scaler,
            "pca": pre.pca,
            "kmeans": model,
            "feature_columns": scaled.columns.tolist(),
            "cluster_mapping": mapping,
            "metrics": metrics,
        }

        if save_model:
            ModelPersistence.save(self.model_artifacts, model_dir=MODEL_DIR)

        self._export_cluster_results()
        self._print_final_summary()
        return self.results

    def _export_cluster_results(self) -> None:
        if self.results is None:
            return

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        patient_clusters = pd.DataFrame({
            "patient_id": self.results.features_df["patient_id"].values,
            "cluster_id": self.results.labels,
        })
        
        cluster_info = []
        for cluster_id, label in zip(self.results.labels, self.results.features_df["patient_id"]):
            mapping = self.results.cluster_mapping[cluster_id]
            cluster_info.append({
                "patient_id": label,
                "cluster_id": cluster_id,
                "pulse_type": mapping["type"],
                "description": mapping["description"],
                "confidence_score": float(mapping["confidence_score"]),
                "cluster_size": mapping["size"],
            })
        
        results_df = pd.DataFrame(cluster_info)
        results_path = os.path.join(OUTPUT_DIR, "cluster_results.csv")
        results_df.to_csv(results_path, index=False)
        print(f"\nSaved cluster results: {results_path}")

    def _print_final_summary(self) -> None:
        if self.results is None:
            return

        print("\nVALIDATION + SUMMARY")
        print("=" * 60)
        print(f"Patients processed: {self.results.features_df['patient_id'].nunique()}")
        print(f"Feature count: {self.results.scaled_features.shape[1]}")
        print("Clusters: 3")
        print(f"Silhouette: {self.results.metrics['silhouette']:.4f}")
        print(f"Davies-Bouldin: {self.results.metrics['davies_bouldin']:.4f}")
        print(f"Calinski-Harabasz: {self.results.metrics['calinski_harabasz']:.4f}")

        for cid, info in sorted(self.results.cluster_mapping.items()):
            print(
                f"Cluster {cid}: {info['description']}, size={info['size']}, "
                f"score={info['confidence_score']:.4f}"
            )

    @staticmethod
    def _prepare_single_file(file_path: str) -> pd.DataFrame:
        df = read_pulse_csv(file_path)
        df = PulseDataCollector.normalize_input_schema(df)
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Prediction file missing columns after normalization {missing}: {file_path}")

        df = df.copy()
        for col in REQUIRED_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["patient_id"] = Path(file_path).stem
        df["file_index"] = 0
        return df

    def predict_new_csv(self, file_path: str, model_dir: str = MODEL_DIR) -> Dict:
        artifacts = self.model_artifacts or ModelPersistence.load(model_dir)

        one = self._prepare_single_file(file_path)
        cleaned = PulseDataCleaner(one).clean()
        processor = PulseSignalProcessor(cleaned)
        filtered = processor.apply_filters(fs=100.0)
        peaks = processor.find_pulse_peaks(distance=30)
        rr = processor.calculate_rr_intervals(peaks)

        features = PulseFeatureExtractor(filtered, peaks, rr).extract_all_features()
        if features.empty:
            raise ValueError("No features extracted for new CSV")

        features = features.set_index("patient_id")
        numeric = features.select_dtypes(include=[np.number])

        feature_columns = artifacts["feature_columns"]
        for col in feature_columns:
            if col not in numeric.columns:
                numeric[col] = 0.0
        numeric = numeric[feature_columns].fillna(0.0)

        scaled = artifacts["scaler"].transform(numeric)
        pca = artifacts["pca"].transform(scaled)
        pred = int(artifacts["kmeans"].predict(pca)[0])

        mapping = artifacts["cluster_mapping"][pred]
        result = {
            "patient_id": features.index[0],
            "cluster": pred,
            "pulse_type": mapping["description"],
        }

        print("\nFUTURE CSV PREDICTION")
        print("=" * 60)
        print(f"Patient: {result['patient_id']}")
        print(f"Predicted cluster: {result['cluster']}")
        print(f"Predicted type: {result['pulse_type']}")

        return result


# =================================================
# Main
# =================================================


def analyze_pulse_data(folder_path: str = DEFAULT_DATA_FOLDER) -> Tuple[WorkflowResults, Dict]:
    workflow = PulseAnalysisWorkflow(folder_path=folder_path)
    results = workflow.run(save_model=True)
    return results, workflow.model_artifacts


if __name__ == "__main__":
    results, model_artifacts = analyze_pulse_data(DEFAULT_DATA_FOLDER)

    print("\n" + "=" * 60)
    print("WORKFLOW COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"Data folder: {DEFAULT_DATA_FOLDER}")
    print(f"Models saved to: {MODEL_DIR}/")
    print(f"Results saved to: {OUTPUT_DIR}/")
    print(f"Plots saved to: {PLOTS_DIR}/")
