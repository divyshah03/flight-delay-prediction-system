"""PyTorch neural net classifier: embeddings for airport/carrier/route
concatenated with dense numeric features, feeding a small MLP.

Categorical vocabularies are built from the TRAIN split only; any category
seen only in the test split (e.g. a route that didn't appear in training)
maps to a reserved "unknown" index. This isn't a leakage concern in the
temporal sense (route/carrier/airport identity doesn't depend on when it's
observed), but building the vocabulary from the full dataset would still be
sloppy practice -- a model shouldn't get credit for having "seen" a category
that a real deployment, trained only on past data, wouldn't have.
"""

from __future__ import annotations

import random
from pathlib import Path

import mlflow
import numpy as np
import polars as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data.config import PROCESSED_DIR
from evaluation.metrics import FEATURE_COLUMNS, classification_metrics, time_based_split

ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)

SEED = 42

CAT_COLUMNS = ["Origin", "Reporting_Airline", "route"]
EMBED_DIMS = {"Origin": 6, "Reporting_Airline": 6, "route": 12}
UNK_INDEX = 0


class Vocab:
    def __init__(self, values: list[str]) -> None:
        uniq = sorted(set(values))
        self.token_to_idx = {v: i + 1 for i, v in enumerate(uniq)}  # 0 reserved for unknown
        self.size = len(uniq) + 1

    def encode(self, values: pl.Series) -> np.ndarray:
        return np.array([self.token_to_idx.get(v, UNK_INDEX) for v in values.to_list()], dtype=np.int64)


class DelayNet(nn.Module):
    def __init__(self, vocab_sizes: dict[str, int], n_dense: int, hidden: tuple[int, ...] = (128, 64)) -> None:
        super().__init__()
        self.embeddings = nn.ModuleDict(
            {col: nn.Embedding(vocab_sizes[col], EMBED_DIMS[col]) for col in CAT_COLUMNS}
        )
        embed_total = sum(EMBED_DIMS.values())
        layers: list[nn.Module] = []
        in_dim = embed_total + n_dense
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(0.2)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, cat_inputs: dict[str, torch.Tensor], dense: torch.Tensor) -> torch.Tensor:
        embedded = [self.embeddings[col](cat_inputs[col]) for col in CAT_COLUMNS]
        x = torch.cat(embedded + [dense], dim=1)
        return self.mlp(x).squeeze(-1)


def _dense_matrix(frame: pl.DataFrame, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    raw = frame.select(list(FEATURE_COLUMNS)).to_numpy().astype(np.float64)
    raw = np.where(np.isnan(raw), means, raw)
    return ((raw - means) / stds).astype(np.float32)


def train_classifier(
    features_path: Path = PROCESSED_DIR / "features.parquet",
    epochs: int = 15,
    batch_size: int = 2048,
    lr: float = 1e-3,
    seed: int = SEED,
) -> dict:
    # Fixed seed for reproducibility: covers weight init (nn.Linear/nn.Embedding
    # draw from the global RNG at construction time) and the train DataLoader's
    # shuffle order, both of which are otherwise nondeterministic run-to-run.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    frame = pl.read_parquet(features_path)
    train, test = time_based_split(frame)

    vocabs = {col: Vocab(train[col].to_list()) for col in CAT_COLUMNS}

    raw_train = train.select(list(FEATURE_COLUMNS)).to_numpy().astype(np.float64)
    means = np.nanmean(raw_train, axis=0)
    means = np.where(np.isnan(means), 0.0, means)
    stds = np.nanstd(raw_train, axis=0)
    stds = np.where((stds == 0) | np.isnan(stds), 1.0, stds)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def make_loader(split: pl.DataFrame, shuffle: bool) -> DataLoader:
        cat_tensors = {col: torch.from_numpy(vocabs[col].encode(split[col])) for col in CAT_COLUMNS}
        dense = torch.from_numpy(_dense_matrix(split, means, stds))
        y = torch.from_numpy(split["is_delayed"].to_numpy().astype(np.float32))
        dataset = TensorDataset(cat_tensors["Origin"], cat_tensors["Reporting_Airline"], cat_tensors["route"], dense, y)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    train_loader = make_loader(train, shuffle=True)
    test_loader = make_loader(test, shuffle=False)

    model = DelayNet({col: vocabs[col].size for col in CAT_COLUMNS}, n_dense=len(FEATURE_COLUMNS)).to(device)
    pos_weight = torch.tensor([(train["is_delayed"] == 0).sum() / max((train["is_delayed"] == 1).sum(), 1)])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for origin, carrier, route, dense, y in train_loader:
            origin, carrier, route = origin.to(device), carrier.to(device), route.to(device)
            dense, y = dense.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model({"Origin": origin, "Reporting_Airline": carrier, "route": route}, dense)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
        print(f"epoch {epoch + 1}/{epochs} loss={total_loss / len(train):.4f}")

    model.eval()
    all_logits = []
    with torch.no_grad():
        for origin, carrier, route, dense, _y in test_loader:
            origin, carrier, route, dense = origin.to(device), carrier.to(device), route.to(device), dense.to(device)
            logits = model({"Origin": origin, "Reporting_Airline": carrier, "route": route}, dense)
            all_logits.append(logits.cpu().numpy())
    proba = 1.0 / (1.0 + np.exp(-np.concatenate(all_logits)))
    y_test = test["is_delayed"].to_numpy()
    report = classification_metrics(y_test, proba)

    with mlflow.start_run(run_name="pytorch_classifier"):
        mlflow.log_params({"epochs": epochs, "batch_size": batch_size, "lr": lr, "hidden": "128,64"})
        mlflow.log_metrics(
            {
                "roc_auc": report.roc_auc,
                "pr_auc": report.pr_auc,
                "brier": report.brier,
                "catch_rate_at_20pct_fpr": report.catch_rate_at_fpr,
            }
        )
        model_path = ARTIFACT_DIR / "torch_classifier.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "vocab_sizes": {col: vocabs[col].size for col in CAT_COLUMNS},
                "vocab_tokens": {col: vocabs[col].token_to_idx for col in CAT_COLUMNS},
                "means": means,
                "stds": stds,
            },
            model_path,
        )
        mlflow.log_artifact(str(model_path))

    print(f"PyTorch classifier: ROC-AUC={report.roc_auc:.4f} PR-AUC={report.pr_auc:.4f}")
    return {"model": model, "report": report, "test": test, "y_test": y_test, "proba": proba}


if __name__ == "__main__":
    train_classifier()
