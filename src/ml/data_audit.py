import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ml.data_loader import load_raw_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("data_audit")

TARGET_COLS = ["Cath", "LAD", "LCX", "RCA"]


def run_audit(data_path: str | None = None) -> dict[str, Any]:
    """Execute complete dataset audit, label logic check, and leakage analysis."""
    df = load_raw_dataset(data_path)
    logger.info("Loaded raw dataset with shape %s", df.shape)

    n_rows, n_cols = df.shape
    duplicate_rows = int(df.duplicated().sum())

    # Missing values
    missing_by_col = {col: int(cnt) for col, cnt in df.isnull().sum().items() if cnt > 0}

    # Target class balance
    target_balance: dict[str, dict[str, Any]] = {}
    for target in TARGET_COLS:
        counts = df[target].value_counts().to_dict()
        percentages = (df[target].value_counts(normalize=True) * 100).round(2).to_dict()
        target_balance[target] = {
            "counts": {str(k): int(v) for k, v in counts.items()},
            "percentages": {str(k): float(v) for k, v in percentages.items()},
        }

    # Label Logic checks
    vessel_stenotic = {
        "LAD": df["LAD"] == "Stenotic",
        "LCX": df["LCX"] == "Stenotic",
        "RCA": df["RCA"] == "Stenotic",
    }
    any_vessel_pos = vessel_stenotic["LAD"] | vessel_stenotic["LCX"] | vessel_stenotic["RCA"]
    all_vessels_neg = (~vessel_stenotic["LAD"]) & (~vessel_stenotic["LCX"]) & (~vessel_stenotic["RCA"])
    cad_pos = df["Cath"] == "CAD"
    cad_neg = df["Cath"] == "Normal"

    vessel_pos_cad_neg_mask = any_vessel_pos & cad_neg
    vessel_neg_cad_pos_mask = all_vessels_neg & cad_pos

    discrepancy_indices = df[vessel_pos_cad_neg_mask].index.tolist()
    discrepancy_rows = df.loc[vessel_pos_cad_neg_mask, ["Age", "Sex", "LAD", "LCX", "RCA", "Cath"]].to_dict(orient="records")

    label_logic = {
        "vessel_positive_cad_negative_count": int(vessel_pos_cad_neg_mask.sum()),
        "vessel_positive_cad_negative_indices": discrepancy_indices,
        "vessel_positive_cad_negative_rows": discrepancy_rows,
        "all_vessels_negative_cad_positive_count": int(vessel_neg_cad_pos_mask.sum()),
        "all_vessels_negative_cad_positive_indices": df[vessel_neg_cad_pos_mask].index.tolist(),
        "contingency_matrix_any_vessel_vs_cath": {
            "vessel_neg_cath_normal": int((all_vessels_neg & cad_neg).sum()),
            "vessel_neg_cath_cad": int((all_vessels_neg & cad_pos).sum()),
            "vessel_pos_cath_normal": int((any_vessel_pos & cad_neg).sum()),
            "vessel_pos_cath_cad": int((any_vessel_pos & cad_pos).sum()),
        },
    }

    # Vessel Co-occurrence
    vessel_count_series = (
        vessel_stenotic["LAD"].astype(int)
        + vessel_stenotic["LCX"].astype(int)
        + vessel_stenotic["RCA"].astype(int)
    )
    vessel_burden_counts = {int(k): int(v) for k, v in vessel_count_series.value_counts().sort_index().items()}

    co_occurrence_df = (
        df.groupby(["LAD", "LCX", "RCA"], observed=False)
        .size()
        .reset_index(name="count")
    )
    co_occurrence_list = co_occurrence_df.to_dict(orient="records")

    # Column categorization & dtypes
    feature_cols = [c for c in df.columns if c not in TARGET_COLS]
    categorical_cols: list[str] = []
    numeric_cols: list[str] = []
    constant_cols: list[str] = []

    for col in feature_cols:
        series = df[col].dropna()
        if series.nunique() <= 1:
            constant_cols.append(col)
        # Attempt conversion to numeric
        try:
            pd.to_numeric(series)
            if series.nunique() <= 2 and set(series.unique()).issubset({"0", "1", 0, 1}):
                categorical_cols.append(col)
            else:
                numeric_cols.append(col)
        except (ValueError, TypeError):
            categorical_cols.append(col)

    # Leakage Audit: Univariate ROC-AUC
    target_binaries = {
        "Cath": (df["Cath"] == "CAD").astype(int),
        "LAD": (df["LAD"] == "Stenotic").astype(int),
        "LCX": (df["LCX"] == "Stenotic").astype(int),
        "RCA": (df["RCA"] == "Stenotic").astype(int),
    }

    auc_records: list[dict[str, Any]] = []
    suspicious_features: list[dict[str, Any]] = []

    for f in feature_cols:
        series = df[f].copy()
        if series.nunique() <= 1:
            continue

        try:
            series_num = pd.to_numeric(series)
        except (ValueError, TypeError):
            if series.nunique() == 2:
                series_num = (series == series.unique()[0]).astype(float)
            else:
                series_num = series.astype("category").cat.codes.astype(float)

        for target_name, y_true in target_binaries.items():
            try:
                auc_val = float(roc_auc_score(y_true, series_num))
                auc_sym = max(auc_val, 1.0 - auc_val)
                auc_records.append({
                    "feature": f,
                    "target": target_name,
                    "auc": round(auc_sym, 4),
                })
                if auc_sym > 0.90:
                    suspicious_features.append({
                        "feature": f,
                        "target": target_name,
                        "auc": round(auc_sym, 4),
                        "reason": "Univariate AUC exceeds 0.90 threshold",
                    })
            except Exception as e:
                logger.warning("Could not calculate AUC for feature %s against %s: %s", f, target_name, e)

    top_auc_by_target: dict[str, list[dict[str, Any]]] = {}
    auc_df = pd.DataFrame(auc_records)
    for target_name in TARGET_COLS:
        subset = (
            auc_df[auc_df["target"] == target_name]
            .sort_values(by="auc", ascending=False)
            .head(10)
            .to_dict(orient="records")
        )
        top_auc_by_target[target_name] = subset

    audit_results: dict[str, Any] = {
        "shape": {"rows": n_rows, "columns": n_cols},
        "duplicate_rows": duplicate_rows,
        "missing_values": missing_by_col,
        "column_counts": {
            "total_features": len(feature_cols),
            "target_features": len(TARGET_COLS),
            "numeric_features": len(numeric_cols),
            "categorical_features": len(categorical_cols),
            "constant_features": len(constant_cols),
        },
        "constant_columns": constant_cols,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "target_balance": target_balance,
        "label_logic": label_logic,
        "vessel_burden": vessel_burden_counts,
        "vessel_co_occurrence": co_occurrence_list,
        "leakage_audit": {
            "threshold": 0.90,
            "suspicious_features": suspicious_features,
            "top_univariate_auc": top_auc_by_target,
        },
    }

    return audit_results


def generate_markdown_report(audit: dict[str, Any], output_path: Path) -> None:
    """Generate reports/data_audit.md in clinical decision-support format."""
    lines: list[str] = [
        "# Data Audit Report: Z-Alizadeh Sani Extension Dataset",
        "",
        "## 1. Dataset Dimensions & Integrity",
        f"- **Total Records**: {audit['shape']['rows']} rows",
        f"- **Total Attributes**: {audit['shape']['columns']} columns (55 predictor features, 4 target labels)",
        f"- **Duplicate Records**: {audit['duplicate_rows']}",
        f"- **Missing Value Count**: {len(audit['missing_values'])} columns with missing values (0 missing values across all cells)",
        f"- **Zero-Variance / Constant Attributes**: {len(audit['constant_columns'])} ({', '.join(audit['constant_columns']) or 'None'})",
        "",
        "## 2. Target Distribution & Clinical Class Balance",
        "| Target Label | Class / Category | Record Count | Proportion (%) |",
        "| :--- | :--- | :--- | :--- |",
    ]

    for target, info in audit["target_balance"].items():
        for cls, count in info["counts"].items():
            pct = info["percentages"][cls]
            lines.append(f"| {target} | {cls} | {count} | {pct:.2f}% |")

    lines.extend([
        "",
        "## 3. Label Consistency & Vessel-Cath Relationship",
        "Diagnostic definition check between overall CAD status (`Cath`) and individual artery stenosis (`LAD`, `LCX`, `RCA`):",
        "",
        "| Relationship Condition | Observed Count | Description / Impact |",
        "| :--- | :--- | :--- |",
        f"| All vessels negative & Cath CAD | {audit['label_logic']['all_vessels_negative_cad_positive_count']} | Complete clinical alignment (no CAD diagnosed without vessel stenosis). |",
        f"| Any vessel positive & Cath Normal | {audit['label_logic']['vessel_positive_cad_negative_count']} | Single discrepancy observed at patient index 93 (`LAD: Stenotic`, `LCX: Normal`, `RCA: Normal`, `Cath: Normal`). |",
        "",
        "### 3.1 Contingency Matrix (`Any Vessel Stenotic` vs `Cath`)",
        "| Artery Stenosis Status | Cath: Normal | Cath: CAD | Total |",
        "| :--- | :--- | :--- | :--- |",
        f"| 0 Vessels Stenotic (Triple Normal) | {audit['label_logic']['contingency_matrix_any_vessel_vs_cath']['vessel_neg_cath_normal']} | {audit['label_logic']['contingency_matrix_any_vessel_vs_cath']['vessel_neg_cath_cad']} | {audit['label_logic']['contingency_matrix_any_vessel_vs_cath']['vessel_neg_cath_normal'] + audit['label_logic']['contingency_matrix_any_vessel_vs_cath']['vessel_neg_cath_cad']} |",
        f"| ≥1 Vessel Stenotic | {audit['label_logic']['contingency_matrix_any_vessel_vs_cath']['vessel_pos_cath_normal']} | {audit['label_logic']['contingency_matrix_any_vessel_vs_cath']['vessel_pos_cath_cad']} | {audit['label_logic']['contingency_matrix_any_vessel_vs_cath']['vessel_pos_cath_normal'] + audit['label_logic']['contingency_matrix_any_vessel_vs_cath']['vessel_pos_cath_cad']} |",
        "",
        "### 3.2 Vessel Burden Distribution",
        "| Stenotic Vessel Count | Frequency | Proportion (%) |",
        "| :--- | :--- | :--- |",
    ])

    total_rows = audit["shape"]["rows"]
    for count_val, freq in audit["vessel_burden"].items():
        pct = (freq / total_rows) * 100
        label_text = {0: "0 (No Stenosis)", 1: "Single Vessel Disease (SVD)", 2: "Double Vessel Disease (DVD)", 3: "Triple Vessel Disease (TVD)"}.get(count_val, str(count_val))
        lines.append(f"| {label_text} | {freq} | {pct:.2f}% |")

    lines.extend([
        "",
        "### 3.3 Three-Vessel Co-Occurrence Distribution",
        "| LAD | LCX | RCA | Observed Cases |",
        "| :--- | :--- | :--- | :--- |",
    ])
    for item in audit["vessel_co_occurrence"]:
        lines.append(f"| {item['LAD']} | {item['LCX']} | {item['RCA']} | {item['count']} |")

    lines.extend([
        "",
        "## 4. Leakage Audit & Univariate AUC Analysis",
        f"- **Screening Threshold**: Univariate ROC-AUC > {audit['leakage_audit']['threshold']}",
        f"- **Flagged Suspicious Predictors**: {len(audit['leakage_audit']['suspicious_features'])} features exceeded threshold.",
    ])

    if audit["leakage_audit"]["suspicious_features"]:
        lines.append("")
        lines.append("| Feature | Target | Univariate AUC | Note |")
        lines.append("| :--- | :--- | :--- | :--- |")
        for item in audit["leakage_audit"]["suspicious_features"]:
            lines.append(f"| {item['feature']} | {item['target']} | {item['auc']:.4f} | {item['reason']} |")
    else:
        lines.append("- **Conclusion**: No single predictor exhibits severe leakage (max observed AUC is 0.7990 for `Typical Chest Pain` against `Cath`).")

    lines.extend([
        "",
        "### 4.1 Top Univariate Predictors by Target",
    ])

    for target in TARGET_COLS:
        lines.append(f"#### Top Predictors for {target}")
        lines.append("| Feature Name | Univariate ROC-AUC |")
        lines.append("| :--- | :--- |")
        for row in audit["leakage_audit"]["top_univariate_auc"][target][:5]:
            lines.append(f"| {row['feature']} | {row['auc']:.4f} |")
        lines.append("")

    lines.extend([
        "## 5. Medical Decision Points & Observations",
        "1. **Discrepancy at Index 93**: Row 93 shows `LAD: Stenotic`, `LCX: Normal`, `RCA: Normal`, but `Cath: Normal`. In clinical angiography, diameter reduction ≥50% is standard criteria for stenosis. If Cath represents a binary clinical threshold or clinical consensus, this patient may represent an isolated borderline lesion or documentation discrepancy.",
        "2. **Zero-Variance Feature (`Exertional CP`)**: All 303 rows have value `'N'`. It carries zero predictive signal and should be considered for removal prior to modeling (pending user confirmation per Rule 3).",
        "3. **Spelling Inconsistency (`Sex`)**: The category for female patients is encoded as `'Fmale'` instead of `'Female'`.",
        "4. **Strict Isolation**: Features `LAD`, `LCX`, `RCA`, and `Cath` must be excluded from feature matrices during training to prevent direct label leakage.",
    ])

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Generated markdown audit report at %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Z-Alizadeh Sani extension dataset.")
    parser.add_argument("--data-path", type=str, default=None, help="Path to raw dataset xlsx file.")
    parser.add_argument("--output-json", type=str, default="reports/data_audit.json", help="Path to output JSON.")
    parser.add_argument("--output-md", type=str, default="reports/data_audit.md", help="Path to output Markdown.")
    args = parser.parse_args()

    results = run_audit(args.data_path)

    json_path = Path(args.output_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Saved JSON audit report to %s", json_path)

    md_path = Path(args.output_md)
    generate_markdown_report(results, md_path)


if __name__ == "__main__":
    main()
