import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler, StandardScaler

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ml.data_loader import load_raw_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("preprocess")

TARGET_COLS = ["Cath", "LAD", "LCX", "RCA"]

# Feature metadata catalog: clinical group, unit, reference range, display label
FEATURE_METADATA: dict[str, dict[str, Any]] = {
    # 1. Demographics
    "Age": {"group": "demographics", "unit": "years", "normal_range": None, "display_label": "Age", "type": "continuous"},
    "Weight": {"group": "demographics", "unit": "kg", "normal_range": None, "display_label": "Weight", "type": "continuous"},
    "Length": {"group": "demographics", "unit": "cm", "normal_range": None, "display_label": "Height (Length)", "type": "continuous"},
    "Sex": {"group": "demographics", "unit": None, "normal_range": None, "display_label": "Sex (Male / Female)", "type": "binary"},
    "BMI": {"group": "demographics", "unit": "kg/m²", "normal_range": "18.5 - 24.9", "display_label": "Body Mass Index (BMI)", "type": "continuous"},

    # 2. History
    "DM": {"group": "history", "unit": None, "normal_range": "0 (Absent)", "display_label": "Diabetes Mellitus", "type": "binary"},
    "HTN": {"group": "history", "unit": None, "normal_range": "0 (Absent)", "display_label": "Hypertension", "type": "binary"},
    "Current Smoker": {"group": "history", "unit": None, "normal_range": "0 (Absent)", "display_label": "Current Smoker", "type": "binary"},
    "EX-Smoker": {"group": "history", "unit": None, "normal_range": "0 (Absent)", "display_label": "Ex-Smoker", "type": "binary"},
    "FH": {"group": "history", "unit": None, "normal_range": "0 (Absent)", "display_label": "Family History of CAD", "type": "binary"},
    "Obesity": {"group": "history", "unit": None, "normal_range": "N (BMI < 30)", "display_label": "Obesity", "type": "binary"},
    "CRF": {"group": "history", "unit": None, "normal_range": "N (Absent)", "display_label": "Chronic Renal Failure", "type": "binary"},
    "CVA": {"group": "history", "unit": None, "normal_range": "N (Absent)", "display_label": "Cerebrovascular Accident", "type": "binary"},
    "Airway disease": {"group": "history", "unit": None, "normal_range": "N (Absent)", "display_label": "Airway Disease (COPD/Asthma)", "type": "binary"},
    "Thyroid Disease": {"group": "history", "unit": None, "normal_range": "N (Absent)", "display_label": "Thyroid Disease", "type": "binary"},
    "CHF": {"group": "history", "unit": None, "normal_range": "N (Absent)", "display_label": "Congestive Heart Failure", "type": "binary"},
    "DLP": {"group": "history", "unit": None, "normal_range": "N (Absent)", "display_label": "Dyslipidemia", "type": "binary"},
    "Typical Chest Pain": {"group": "history", "unit": None, "normal_range": "0 (Absent)", "display_label": "Typical Anginal Chest Pain", "type": "binary"},
    "Dyspnea": {"group": "history", "unit": None, "normal_range": "N (Absent)", "display_label": "Dyspnea", "type": "binary"},
    "Function Class": {"group": "history", "unit": "NYHA Class", "normal_range": "0 (None)", "display_label": "NYHA Functional Class (0-3)", "type": "ordinal"},
    "Atypical": {"group": "history", "unit": None, "normal_range": "N (Absent)", "display_label": "Atypical Chest Pain", "type": "binary"},
    "Nonanginal": {"group": "history", "unit": None, "normal_range": "N (Absent)", "display_label": "Non-Anginal Chest Pain", "type": "binary"},
    "Exertional CP": {"group": "history", "unit": None, "normal_range": "N (Absent)", "display_label": "Exertional Chest Pain (Constant=N)", "type": "constant"},
    "LowTH Ang": {"group": "history", "unit": None, "normal_range": "N (Absent)", "display_label": "Low-Threshold Angina", "type": "binary"},

    # 3. Exam
    "BP": {"group": "exam", "unit": "mmHg", "normal_range": "90 - 120", "display_label": "Systolic Blood Pressure", "type": "continuous"},
    "PR": {"group": "exam", "unit": "bpm", "normal_range": "60 - 100", "display_label": "Pulse Rate", "type": "continuous"},
    "Edema": {"group": "exam", "unit": None, "normal_range": "0 (Absent)", "display_label": "Peripheral Edema", "type": "binary"},
    "Weak Peripheral Pulse": {"group": "exam", "unit": None, "normal_range": "N (Absent)", "display_label": "Weak Peripheral Pulse", "type": "binary"},
    "Lung rales": {"group": "exam", "unit": None, "normal_range": "N (Absent)", "display_label": "Lung Rales / Crackles", "type": "binary"},
    "Systolic Murmur": {"group": "exam", "unit": None, "normal_range": "N (Absent)", "display_label": "Systolic Murmur", "type": "binary"},
    "Diastolic Murmur": {"group": "exam", "unit": None, "normal_range": "N (Absent)", "display_label": "Diastolic Murmur", "type": "binary"},

    # 4. ECG
    "Q Wave": {"group": "ECG", "unit": None, "normal_range": "0 (Absent)", "display_label": "Pathological Q Wave", "type": "binary"},
    "St Elevation": {"group": "ECG", "unit": None, "normal_range": "0 (Absent)", "display_label": "ST Elevation", "type": "binary"},
    "St Depression": {"group": "ECG", "unit": None, "normal_range": "0 (Absent)", "display_label": "ST Depression", "type": "binary"},
    "Tinversion": {"group": "ECG", "unit": None, "normal_range": "0 (Absent)", "display_label": "T-Wave Inversion", "type": "binary"},
    "LVH": {"group": "ECG", "unit": None, "normal_range": "N (Absent)", "display_label": "Left Ventricular Hypertrophy (ECG)", "type": "binary"},
    "Poor R Progression": {"group": "ECG", "unit": None, "normal_range": "N (Absent)", "display_label": "Poor R-Wave Progression", "type": "binary"},
    "BBB": {"group": "ECG", "unit": None, "normal_range": "N (Normal)", "display_label": "Bundle Branch Block (N, LBBB, RBBB)", "type": "nominal"},

    # 5. Lab
    "FBS": {"group": "lab", "unit": "mg/dL", "normal_range": "70 - 99", "display_label": "Fasting Blood Sugar", "type": "continuous"},
    "CR": {"group": "lab", "unit": "mg/dL", "normal_range": "0.6 - 1.2", "display_label": "Serum Creatinine", "type": "continuous"},
    "TG": {"group": "lab", "unit": "mg/dL", "normal_range": "< 150", "display_label": "Triglycerides", "type": "continuous"},
    "LDL": {"group": "lab", "unit": "mg/dL", "normal_range": "< 100", "display_label": "LDL Cholesterol", "type": "continuous"},
    "HDL": {"group": "lab", "unit": "mg/dL", "normal_range": "> 40 (M) / > 50 (F)", "display_label": "HDL Cholesterol", "type": "continuous"},
    "BUN": {"group": "lab", "unit": "mg/dL", "normal_range": "7 - 20", "display_label": "Blood Urea Nitrogen", "type": "continuous"},
    "ESR": {"group": "lab", "unit": "mm/hr", "normal_range": "0 - 20", "display_label": "Erythrocyte Sedimentation Rate", "type": "continuous"},
    "HB": {"group": "lab", "unit": "g/dL", "normal_range": "12.0 - 17.5", "display_label": "Hemoglobin", "type": "continuous"},
    "K": {"group": "lab", "unit": "mEq/L", "normal_range": "3.5 - 5.0", "display_label": "Potassium (K+)", "type": "continuous"},
    "Na": {"group": "lab", "unit": "mEq/L", "normal_range": "135 - 145", "display_label": "Sodium (Na+)", "type": "continuous"},
    "WBC": {"group": "lab", "unit": "cells/µL", "normal_range": "4000 - 11000", "display_label": "White Blood Cell Count", "type": "continuous"},
    "Lymph": {"group": "lab", "unit": "%", "normal_range": "20 - 40", "display_label": "Lymphocyte Percentage", "type": "continuous"},
    "Neut": {"group": "lab", "unit": "%", "normal_range": "40 - 70", "display_label": "Neutrophil Percentage", "type": "continuous"},
    "PLT": {"group": "lab", "unit": "x10³/µL", "normal_range": "150 - 450", "display_label": "Platelet Count", "type": "continuous"},

    # 6. Echo
    "EF-TTE": {"group": "echo", "unit": "%", "normal_range": "50 - 70", "display_label": "Ejection Fraction (TTE)", "type": "continuous"},
    "Region RWMA": {"group": "echo", "unit": "segments", "normal_range": "0 (None)", "display_label": "Regional Wall Motion Abnormality Count", "type": "discrete"},
    "VHD": {"group": "echo", "unit": "grade", "normal_range": "N (None)", "display_label": "Valvular Heart Disease Severity", "type": "ordinal"},
}

VHD_ORDINAL_MAP = {
    "N": 0,
    "mild": 1,
    "Moderate": 2,
    "Severe": 3,
}


def encode_dataset(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Encode raw dataset into clean numerical predictor matrix X and target matrix Y.

    Leakage prevention: Strictly excludes LAD, LCX, RCA, and Cath from X.
    Scalers are NOT fitted here, strictly adhering to CV-only scaling rule.
    """
    df = df_raw.copy()

    # Targets
    y_dict = {
        "Cath": (df["Cath"] == "CAD").astype(int),
        "LAD": (df["LAD"] == "Stenotic").astype(int),
        "LCX": (df["LCX"] == "Stenotic").astype(int),
        "RCA": (df["RCA"] == "Stenotic").astype(int),
    }
    y_df = pd.DataFrame(y_dict)

    x_dict: dict[str, Any] = {}

    # Continuous & discrete numeric features
    numeric_features = [
        "Age", "Weight", "Length", "BMI", "BP", "PR", "FBS", "CR", "TG",
        "LDL", "HDL", "BUN", "ESR", "HB", "K", "Na", "WBC", "Lymph",
        "Neut", "PLT", "EF-TTE", "Region RWMA"
    ]
    for feat in numeric_features:
        x_dict[feat] = pd.to_numeric(df[feat], errors="raise").astype(float)

    # Sex (normalize 'Fmale' / 'Female' -> 0, 'Male' -> 1)
    x_dict["Sex"] = df["Sex"].apply(lambda v: 1 if str(v).strip().lower() == "male" else 0).astype(int)

    # Binary Y/N features
    yn_features = [
        "Obesity", "CRF", "CVA", "Airway disease", "Thyroid Disease",
        "CHF", "DLP", "Weak Peripheral Pulse", "Lung rales",
        "Systolic Murmur", "Diastolic Murmur", "Dyspnea", "Atypical",
        "Nonanginal", "Exertional CP", "LowTH Ang", "LVH", "Poor R Progression"
    ]
    for feat in yn_features:
        x_dict[feat] = df[feat].apply(lambda v: 1 if str(v).strip().upper() == "Y" else 0).astype(int)

    # Binary 0/1 features
    binary_01_features = [
        "DM", "HTN", "Current Smoker", "EX-Smoker", "FH", "Edema",
        "Typical Chest Pain", "Q Wave", "St Elevation", "St Depression", "Tinversion"
    ]
    for feat in binary_01_features:
        x_dict[feat] = pd.to_numeric(df[feat], errors="raise").astype(int)

    # Ordinal features
    x_dict["Function Class"] = pd.to_numeric(df["Function Class"], errors="raise").astype(int)
    x_dict["VHD"] = df["VHD"].map(VHD_ORDINAL_MAP).fillna(0).astype(int)

    # Nominal feature BBB: create binary indicators for LBBB and RBBB (N is reference baseline 0)
    x_dict["BBB_LBBB"] = (df["BBB"] == "LBBB").astype(int)
    x_dict["BBB_RBBB"] = (df["BBB"] == "RBBB").astype(int)

    x_df = pd.DataFrame(x_dict)
    return x_df, y_df


class CVFoldScaler(BaseEstimator, TransformerMixin):
    """Scaler transformer designed to be fitted strictly within CV training folds.

    Never to be fitted on the entire dataset.
    """

    def __init__(self, scaler_type: str = "standard", columns: list[str] | None = None) -> None:
        self.scaler_type = scaler_type
        self.columns = columns
        if scaler_type == "robust":
            self.scaler = RobustScaler()
        elif scaler_type == "standard":
            self.scaler = StandardScaler()
        else:
            raise ValueError(f"Unsupported scaler_type '{scaler_type}'. Choose 'standard' or 'robust'.")
        self.fitted_columns_: list[str] = []

    def fit(self, X: pd.DataFrame, y: Any = None) -> "CVFoldScaler":
        cols_to_scale = self.columns or [
            c for c in X.columns
            if FEATURE_METADATA.get(c, {}).get("type") in ("continuous", "discrete")
        ]
        self.fitted_columns_ = [c for c in cols_to_scale if c in X.columns]
        if self.fitted_columns_:
            self.scaler.fit(X[self.fitted_columns_].values)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        if self.fitted_columns_:
            scaled_values = self.scaler.transform(X[self.fitted_columns_].values)
            X_out[self.fitted_columns_] = scaled_values
        return X_out


def build_feature_schema(x_df: pd.DataFrame) -> dict[str, Any]:
    """Build and validate feature schema with clinical metadata for serve time and modeling."""
    encoded_features = list(x_df.columns)
    continuous_cols = [
        col for col in encoded_features
        if FEATURE_METADATA.get(col, {}).get("type") in ("continuous", "discrete")
    ]
    binary_cols = [
        col for col in encoded_features
        if FEATURE_METADATA.get(col, {}).get("type") == "binary"
        or col.startswith("BBB_")
    ]
    ordinal_cols = [
        col for col in encoded_features
        if FEATURE_METADATA.get(col, {}).get("type") == "ordinal"
    ]
    constant_cols = [col for col in encoded_features if x_df[col].nunique() <= 1]

    schema = {
        "total_encoded_features": len(encoded_features),
        "target_names": TARGET_COLS,
        "feature_names": encoded_features,
        "column_types": {
            "continuous_or_discrete": continuous_cols,
            "binary": binary_cols,
            "ordinal": ordinal_cols,
            "constant": constant_cols,
        },
        "constant_features": constant_cols,
        "feature_catalog": {},
    }

    for col in encoded_features:
        if col in FEATURE_METADATA:
            schema["feature_catalog"][col] = {
                "name": col,
                **FEATURE_METADATA[col],
                "min": float(x_df[col].min()),
                "max": float(x_df[col].max()),
                "mean": round(float(x_df[col].mean()), 4),
                "std": round(float(x_df[col].std()), 4),
            }
        elif col.startswith("BBB_"):
            variant = col.replace("BBB_", "")
            schema["feature_catalog"][col] = {
                "name": col,
                "group": "ECG",
                "unit": None,
                "normal_range": "0 (Absent)",
                "display_label": f"Bundle Branch Block: {variant}",
                "type": "binary",
                "min": float(x_df[col].min()),
                "max": float(x_df[col].max()),
                "mean": round(float(x_df[col].mean()), 4),
                "std": round(float(x_df[col].std()), 4),
            }

    return schema


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess Z-Alizadeh Sani extension dataset.")
    parser.add_argument("--data-path", type=str, default=None, help="Path to raw dataset.")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Output directory for processed data.")
    parser.add_argument("--schema-path", type=str, default="models/feature_schema.json", help="Path to output schema JSON.")
    args = parser.parse_args()

    df_raw = load_raw_dataset(args.data_path)
    logger.info("Loaded raw dataset (%d rows, %d cols)", len(df_raw), len(df_raw.columns))

    x_df, y_df = encode_dataset(df_raw)
    logger.info("Encoded predictor matrix X: %s; Target matrix Y: %s", x_df.shape, y_df.shape)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save unscaled matrices (scaling is strictly performed during CV folds)
    x_path = out_dir / "features_unscaled.csv"
    y_path = out_dir / "targets.csv"
    x_df.to_csv(x_path, index=False)
    y_df.to_csv(y_path, index=False)
    logger.info("Saved unscaled features to %s and targets to %s", x_path, y_path)

    schema = build_feature_schema(x_df)
    schema_path = Path(args.schema_path)
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    logger.info("Saved feature schema to %s", schema_path)


if __name__ == "__main__":
    main()
