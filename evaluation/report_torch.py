"""Generate the PyTorch classifier's calibration table, and complete the
combined XGBoost-vs-PyTorch calibration chart if report_xgboost.py has
already run. See report_xgboost.py's module docstring for why this is a
separate script rather than one that imports both frameworks.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import torch

from data.config import PROCESSED_DIR
from evaluation.metrics import FEATURE_COLUMNS, calibration_table, time_based_split
from evaluation.plots import ARTIFACT_CAL_DIR, plot_calibration
from models.torch_classifier import CAT_COLUMNS, DelayNet, Vocab

ARTIFACT_DIR = ARTIFACT_CAL_DIR.parent


def main() -> None:
    frame = pl.read_parquet(PROCESSED_DIR / "features.parquet")
    _train, test = time_based_split(frame)

    ckpt = torch.load(ARTIFACT_DIR / "torch_classifier.pt", weights_only=False)
    vocabs: dict[str, Vocab] = {}
    for col in CAT_COLUMNS:
        v = Vocab.__new__(Vocab)
        v.token_to_idx = ckpt["vocab_tokens"][col]
        v.size = ckpt["vocab_sizes"][col]
        vocabs[col] = v

    model = DelayNet(ckpt["vocab_sizes"], n_dense=len(FEATURE_COLUMNS))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    means, stds = ckpt["means"], ckpt["stds"]
    raw = test.select(list(FEATURE_COLUMNS)).to_numpy().astype(np.float64)
    raw = np.where(np.isnan(raw), means, raw)
    dense = torch.from_numpy(((raw - means) / stds).astype(np.float32))
    cat = {col: torch.from_numpy(vocabs[col].encode(test[col])) for col in CAT_COLUMNS}

    with torch.no_grad():
        logits = model(cat, dense)
    proba = 1.0 / (1.0 + np.exp(-logits.numpy()))
    y = test["is_delayed"].to_numpy()

    torch_cal = calibration_table(y, proba, n_bins=10)
    torch_cal.write_parquet(ARTIFACT_CAL_DIR / "calibration_pytorch.parquet")
    print("[report_torch] wrote calibration_pytorch.parquet")

    xgb_cal_path = ARTIFACT_CAL_DIR / "calibration_xgboost.parquet"
    if xgb_cal_path.exists():
        xgb_cal = pl.read_parquet(xgb_cal_path)
        plot_calibration({"XGBoost": xgb_cal, "PyTorch": torch_cal}, "calibration_classifiers.png")
        print("[report_torch] wrote calibration_classifiers.png")
    else:
        print("[report_torch] calibration_xgboost.parquet not found yet -- run report_xgboost.py to complete the combined chart")


if __name__ == "__main__":
    main()
