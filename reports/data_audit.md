# Data Audit Report: Z-Alizadeh Sani Extension Dataset

## 1. Dataset Dimensions & Integrity
- **Total Records**: 303 rows
- **Total Attributes**: 59 columns (55 predictor features, 4 target labels)
- **Duplicate Records**: 0
- **Missing Value Count**: 0 columns with missing values (0 missing values across all cells)
- **Zero-Variance / Constant Attributes**: 1 (Exertional CP)

## 2. Target Distribution & Clinical Class Balance
| Target Label | Class / Category | Record Count | Proportion (%) |
| :--- | :--- | :--- | :--- |
| Cath | CAD | 216 | 71.29% |
| Cath | Normal | 87 | 28.71% |
| LAD | Stenotic | 177 | 58.42% |
| LAD | Normal | 126 | 41.58% |
| LCX | Normal | 184 | 60.73% |
| LCX | Stenotic | 119 | 39.27% |
| RCA | Normal | 189 | 62.38% |
| RCA | Stenotic | 114 | 37.62% |

## 3. Label Consistency & Vessel-Cath Relationship
Diagnostic definition check between overall CAD status (`Cath`) and individual artery stenosis (`LAD`, `LCX`, `RCA`):

| Relationship Condition | Observed Count | Description / Impact |
| :--- | :--- | :--- |
| All vessels negative & Cath CAD | 0 | Complete clinical alignment (no CAD diagnosed without vessel stenosis). |
| Any vessel positive & Cath Normal | 1 | Single discrepancy observed at patient index 93 (`LAD: Stenotic`, `LCX: Normal`, `RCA: Normal`, `Cath: Normal`). |

### 3.1 Contingency Matrix (`Any Vessel Stenotic` vs `Cath`)
| Artery Stenosis Status | Cath: Normal | Cath: CAD | Total |
| :--- | :--- | :--- | :--- |
| 0 Vessels Stenotic (Triple Normal) | 86 | 0 | 86 |
| ≥1 Vessel Stenotic | 1 | 216 | 217 |

### 3.2 Vessel Burden Distribution
| Stenotic Vessel Count | Frequency | Proportion (%) |
| :--- | :--- | :--- |
| 0 (No Stenosis) | 86 | 28.38% |
| Single Vessel Disease (SVD) | 87 | 28.71% |
| Double Vessel Disease (DVD) | 67 | 22.11% |
| Triple Vessel Disease (TVD) | 63 | 20.79% |

### 3.3 Three-Vessel Co-Occurrence Distribution
| LAD | LCX | RCA | Observed Cases |
| :--- | :--- | :--- | :--- |
| Normal | Normal | Normal | 86 |
| Normal | Normal | Stenotic | 17 |
| Normal | Stenotic | Normal | 13 |
| Normal | Stenotic | Stenotic | 10 |
| Stenotic | Normal | Normal | 57 |
| Stenotic | Normal | Stenotic | 24 |
| Stenotic | Stenotic | Normal | 33 |
| Stenotic | Stenotic | Stenotic | 63 |

## 4. Leakage Audit & Univariate AUC Analysis
- **Screening Threshold**: Univariate ROC-AUC > 0.9
- **Flagged Suspicious Predictors**: 0 features exceeded threshold.
- **Conclusion**: No single predictor exhibits severe leakage (max observed AUC is 0.7990 for `Typical Chest Pain` against `Cath`).

### 4.1 Top Univariate Predictors by Target
#### Top Predictors for Cath
| Feature Name | Univariate ROC-AUC |
| :--- | :--- |
| Typical Chest Pain | 0.7990 |
| Age | 0.7239 |
| Atypical | 0.7120 |
| EF-TTE | 0.6870 |
| Region RWMA | 0.6717 |

#### Top Predictors for LAD
| Feature Name | Univariate ROC-AUC |
| :--- | :--- |
| Typical Chest Pain | 0.7323 |
| EF-TTE | 0.7084 |
| Age | 0.6772 |
| Region RWMA | 0.6752 |
| Atypical | 0.6653 |

#### Top Predictors for LCX
| Feature Name | Univariate ROC-AUC |
| :--- | :--- |
| Age | 0.6809 |
| Typical Chest Pain | 0.6494 |
| BP | 0.6169 |
| PLT | 0.6012 |
| FBS | 0.5910 |

#### Top Predictors for RCA
| Feature Name | Univariate ROC-AUC |
| :--- | :--- |
| Typical Chest Pain | 0.6497 |
| Neut | 0.6350 |
| Age | 0.6349 |
| Lymph | 0.6284 |
| DM | 0.6275 |

## 5. Medical Decision Points & Observations
1. **Discrepancy at Index 93**: Row 93 shows `LAD: Stenotic`, `LCX: Normal`, `RCA: Normal`, but `Cath: Normal`. In clinical angiography, diameter reduction ≥50% is standard criteria for stenosis. If Cath represents a binary clinical threshold or clinical consensus, this patient may represent an isolated borderline lesion or documentation discrepancy.
2. **Zero-Variance Feature (`Exertional CP`)**: All 303 rows have value `'N'`. It carries zero predictive signal and should be considered for removal prior to modeling (pending user confirmation per Rule 3).
3. **Spelling Inconsistency (`Sex`)**: The category for female patients is encoded as `'Fmale'` instead of `'Female'`.
4. **Strict Isolation**: Features `LAD`, `LCX`, `RCA`, and `Cath` must be excluded from feature matrices during training to prevent direct label leakage.