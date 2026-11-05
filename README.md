# VTE in pregnancy: XGBoost prediction models

The exported package contains the final **10-feature model** and the **9-feature
sensitivity model without lower-limb edema**, each with its fitted sigmoid mapping.
They predict VTE diagnosed after obstetric admission and before the end of pregnancy.
These are the full-training-set final models, not an ensemble of ten fold models.

## Use

Install the recorded inference dependencies in a separate Python environment:

```bash
python -m pip install -r requirements-prediction.txt
python predict_vte.py --input new_records.csv --model 10 --output predictions_10.csv
python predict_vte.py --input new_records.csv --model 9 --output predictions_9.csv
```

Use the exact Chinese column names in `input_template_10_features.csv` or
`input_template_9_features.csv`. Column order is handled automatically.
`feature_dictionary.csv` defines each input. In particular:

- `妊娠周数` is **0 for <28 weeks and 1 for >=28 weeks**, not a numeric week count.
- `分娩次数` is **0 for <3 deliveries and 1 for >=3**, not a raw count.
- BMI is **admission BMI**, not prepregnancy BMI. Other categorical inputs use 0/1.
- Use one row per pregnancy record and the same clinical definitions as the study.

Inputs must already be coded and complete, including BMI.

Outputs preserve input row order and include calibrated probability (0–1), risk
percentage, and calibrated log-odds. Optional `--threshold 0.005` adds a high-risk
indicator at 0.5%; it is not a treatment recommendation. Optional `--id-column ID`
copies that column into the prediction output. Existing output files are not overwritten.

`manifest.json` stores feature order, sigmoid coefficients and model checksums.
`verification.json` records the export/reload comparison against Notebook predictions.
Software reports distinguish saved training-time versions from the current environment.
No participant records, outcomes or individual predictions are included in the export.
The model package supports prediction; it does not reproduce training or cross-validation.
External calibration was imperfect; clinical implementation requires further evaluation.

## Files

| File | Description |
|---|---|
| `README.md` | Model overview, input requirements, and usage instructions. |
| `feature_dictionary.csv` | Predictor definitions and coding rules. |
| `input_template_10_features.csv` | Input template for the 10-feature model. |
| `input_template_9_features.csv` | Input template for the 9-feature model excluding lower-limb edema. |
| `xgboost_10_features.json` | Fitted 10-feature XGBoost model. |
| `xgboost_9_features.json` | Fitted 9-feature XGBoost sensitivity model excluding lower-limb edema. |
| `manifest.json` | Feature order, sigmoid calibration parameters, prediction settings, and model checksums. |
| `predict_vte.py` | Script for generating calibrated predictions using either model. |
| `requirements-prediction.txt` | Python packages and versions required for prediction. |
| `verification.json` | Results of model export and prediction consistency checks. |
