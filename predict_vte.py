"""Predict VTE in pregnancy using exported, frozen XGBoost + sigmoid models.

No training, recalibration, outcome column, or original Notebook is required.
"""
import argparse
import csv
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
import xgboost as xgb


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream), [])
    if not header or len(header) != len(set(header)):
        raise ValueError("The input CSV has no header or has duplicate column names.")
    return pd.read_csv(path, encoding="utf-8-sig")


def _cpu_booster(model_path):
    booster = xgb.Booster()
    booster.load_model(str(model_path))
    if int(xgb.__version__.split(".")[0]) >= 2:
        booster.set_param({"device": "cpu", "nthread": 1})
    else:
        booster.set_param({"predictor": "cpu_predictor", "nthread": 1})
    return booster


def _prepare_frame(data, spec, bmi_imputer_path=None):
    if not isinstance(data, pd.DataFrame) or data.empty or not data.columns.is_unique:
        raise ValueError("Input must be a nonempty DataFrame with unique column names.")
    features = spec["features"]
    absent = [name for name in features if name not in data.columns]
    if absent:
        raise ValueError("Missing required columns: " + ", ".join(absent))
    frame = data.loc[:, features].copy()
    for name in features:
        frame[name] = pd.to_numeric(frame[name], errors="raise")
    if np.isinf(frame.to_numpy(float)).any():
        raise ValueError("Infinite feature values are not allowed.")
    for name, description in spec["feature_definitions"].items():
        if description["type"] == "binary":
            if not frame[name].isin([0, 1]).all():
                raise ValueError(f"{name}: expected the study's 0/1 coding, with no missing values.")
    if frame["年龄"].isna().any() or (frame["年龄"] < 18).any():
        raise ValueError("Age must be available and at least 18 years.")
    if frame["BMI"].isna().any():
        if bmi_imputer_path is None:
            raise ValueError("BMI is missing. Supply complete inputs or a trusted fitted BMI imputer; "
                             "the prediction script never fits an imputer on new patients.")
        import joblib
        import sklearn
        # This optional PRIVATE file contains KNN donor data. Only load a trusted file.
        fitted = joblib.load(bmi_imputer_path)
        if fitted.get("format") != "vte-private-bmi-imputer-v1":
            raise ValueError("Unrecognized BMI imputer file.")
        if fitted["sklearn_version"] != sklearn.__version__:
            raise ValueError("Use the scikit-learn version recorded in the private imputer file: "
                             + fitted["sklearn_version"])
        if spec["model_sha256"] not in fitted["verified_model_sha256"]:
            raise ValueError("This imputer was not verified against this exported model package.")
        columns = fitted["columns"]
        transformed = fitted["imputer"].transform(frame.loc[:, columns])
        missing = frame["BMI"].isna()
        frame.loc[missing, "BMI"] = transformed[missing.to_numpy(), columns.index("BMI")]
    if not np.isfinite(frame.to_numpy(float)).all() or (frame["BMI"] <= 0).any():
        raise ValueError("All required inputs must be finite; BMI must be positive.")
    return frame


def predict_dataframe(data, package_dir, model="10", threshold=None, bmi_imputer_path=None):
    """Return probabilities in the input row order; threshold is a proportion (0.005 = 0.5%)."""
    if str(model) not in {"10", "9"}:
        raise ValueError("model must be '10' or '9'.")
    if threshold is not None and (not np.isfinite(threshold) or not 0 < threshold < 1):
        raise ValueError("threshold must be strictly between 0 and 1; use 0.005 for 0.5%.")
    root = Path(package_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != "vte-xgb-sigmoid-v1":
        raise ValueError("Unrecognized model package format.")
    spec = manifest["models"][str(model)]
    filename = spec["model_file"]
    if Path(filename).name != filename:
        raise ValueError("Invalid model filename.")
    model_path = root / filename
    if sha256_file(model_path) != spec["model_sha256"]:
        raise ValueError("Model checksum does not match the manifest.")
    if manifest["export_environment"]["xgboost"] != xgb.__version__:
        warnings.warn("XGBoost version differs from the verified export environment. "
                      "Use requirements-prediction.txt for reproducibility.")
    frame = _prepare_frame(data, spec, bmi_imputer_path)
    booster = _cpu_booster(model_path)
    if booster.num_features() != len(spec["features"]):
        raise ValueError("Feature count and model disagree.")
    matrix = xgb.DMatrix(frame.to_numpy(dtype=np.float32, copy=True),
                         feature_names=booster.feature_names, nthread=1)
    margin = np.asarray(booster.predict(matrix, output_margin=True,
                        iteration_range=(0, spec["rounds"])), dtype=float)
    mapping = spec["sigmoid"]
    log_odds = mapping["intercept"] + mapping["slope"] * margin
    probability = expit(log_odds)
    if not np.isfinite(log_odds).all() or not np.isfinite(probability).all():
        raise ValueError("Non-finite model predictions.")
    result = pd.DataFrame({"row_position": np.arange(len(frame)),
                           "model": "XGBoost_" + str(model) + "_features",
                           "predicted_probability": probability,
                           "predicted_risk_percent": 100 * probability,
                           "calibrated_log_odds": log_odds}, index=data.index.copy())
    if threshold is not None:
        result["threshold_probability"] = threshold
        result["classified_high_risk"] = (probability >= threshold).astype(int)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--input", required=True, help="UTF-8 CSV; see input_template_*.csv")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", choices=["10", "9"], default="10")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Optional probability, e.g. 0.005. No treatment recommendation.")
    parser.add_argument("--bmi-imputer", default=None, help="Optional trusted PRIVATE joblib file")
    parser.add_argument("--id-column", default=None, help="Optional input ID copied to output")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Output already exists; choose a new filename.")
    frame = _read_csv(args.input)
    if args.id_column is not None and args.id_column not in frame:
        raise ValueError("ID column not found.")
    result = predict_dataframe(frame, args.package, args.model, args.threshold, args.bmi_imputer)
    if args.id_column is not None:
        result.insert(1, "input_id", frame[args.id_column].to_numpy())
    with output.open("x", encoding="utf-8-sig", newline="") as stream:
        result.to_csv(stream, index=False)
    print(f"Saved {len(result)} predictions to {output.resolve()}")


if __name__ == "__main__":
    main()
