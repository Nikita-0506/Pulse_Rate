#!/usr/bin/env python
"""
QUICK START GUIDE - Nadi Pulse Analysis System
================================================

This script demonstrates how to use the pulse analysis pipeline.
"""

# ============================================
# OPTION 1: Run Full Analysis Pipeline
# ============================================

def run_full_analysis():
    """Execute complete analysis on all patient data."""
    from pluse import analyze_pulse_data
    
    print("Starting full analysis pipeline...")
    results, model_artifacts = analyze_pulse_data()
    
    print("\n✅ Analysis complete!")
    print(f"   - Patients processed: {len(results.features_df)}")
    print(f"   - Features extracted: {results.scaled_features.shape[1]}")
    print(f"   - Clusters identified: 3")
    print(f"   - Silhouette score: {results.metrics['silhouette']:.4f}")
    
    return results, model_artifacts


# ============================================
# OPTION 2: Load Pre-trained Model
# ============================================

def load_trained_model():
    """Load a previously trained model from disk."""
    from pluse import ModelPersistence
    
    print("Loading trained model...")
    model = ModelPersistence.load(model_dir="models/")
    
    print("\n✅ Model loaded successfully!")
    print(f"   - Features: {len(model['feature_columns'])}")
    print(f"   - Clusters: 3")
    print(f"   - Mapping: {model['cluster_mapping']}")
    
    return model


# ============================================
# OPTION 3: Predict New Patient
# ============================================

def predict_new_patient(patient_csv_path: str):
    """Predict pulse type for a new patient."""
    from pluse import PulseAnalysisWorkflow
    
    print(f"Predicting pulse type for: {patient_csv_path}")
    
    workflow = PulseAnalysisWorkflow()
    result = workflow.predict_new_csv(
        file_path=patient_csv_path,
        model_dir="models/"
    )
    
    print("\n✅ Prediction complete!")
    print(f"   - Patient: {result['patient_id']}")
    print(f"   - Predicted cluster: {result['cluster']}")
    print(f"   - Pulse type: {result['pulse_type']}")
    
    return result


# ============================================
# OPTION 4: Extract Features Only
# ============================================

def extract_features_only():
    """Extract features without clustering."""
    from pluse import (
        PulseDataCollector, 
        PulseDataCleaner,
        PulseSignalProcessor,
        PulseFeatureExtractor
    )
    
    print("Extracting features from patient data...")
    
    # Load and preprocess
    collector = PulseDataCollector(r"C:\Users\ndpac\OneDrive\Desktop\Pulse_CSV")
    raw_data = collector.load_all_files()
    
    cleaner = PulseDataCleaner(raw_data)
    cleaned = cleaner.clean()
    
    # Process signals
    processor = PulseSignalProcessor(cleaned)
    filtered = processor.apply_filters()
    peaks = processor.find_pulse_peaks()
    rr = processor.calculate_rr_intervals(peaks)
    
    # Extract features
    extractor = PulseFeatureExtractor(filtered, peaks, rr)
    features = extractor.extract_all_features()
    
    print(f"\n✅ Features extracted!")
    print(f"   - Shape: {features.shape}")
    print(f"   - Columns: {features.columns.tolist()[:5]}... (showing first 5)")
    
    return features


# ============================================
# MAIN ENTRY POINT
# ============================================

if __name__ == "__main__":
    import sys
    
    print("\n" + "="*60)
    print("NADI PULSE ANALYSIS - QUICK START")
    print("="*60)
    
    print("\nAvailable options:")
    print("  1. Run full analysis pipeline")
    print("  2. Load pre-trained model")
    print("  3. Predict new patient (requires model)")
    print("  4. Extract features only")
    
    # Example: Run full analysis
    print("\n[Running: Full Analysis Pipeline]")
    print("-" * 60)
    
    try:
        results, artifacts = run_full_analysis()
    except Exception as e:
        print(f"❌ Error: {e}")
        print("   Make sure:")
        print("   - Patient CSV files are in Pulse_CSV/ folder")
        print("   - Each CSV has columns: Time, S1, S2, S3")
