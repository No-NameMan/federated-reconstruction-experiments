from __future__ import annotations
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve
import numpy as np
import pandas as pd
import torch
from fedrecon.data.client_dataset import ClientDataset

MOVIELENS_1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


@dataclass(frozen=True)
class MovieLensData:
    train_clients: dict[int, ClientDataset]
    val_clients: dict[int, ClientDataset]
    test_clients: dict[int, ClientDataset]
    num_items: int
    num_users: int
    user_id_map: dict[int, int]
    item_id_map: dict[int, int]


def download_movielens_1m(data_dir: str | Path) -> Path:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    extracted_dir = data_dir / "ml-1m"
    if extracted_dir.exists():
        return extracted_dir
    zip_path = data_dir / "ml-1m.zip"
    if not zip_path.exists():
        print(f"Downloading MovieLens 1M to {zip_path}...")
        urlretrieve(MOVIELENS_1M_URL, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_dir)
    return extracted_dir


def read_ratings_ml1m(data_dir: str | Path) -> pd.DataFrame:
    ml_dir = download_movielens_1m(data_dir)
    ratings_path = ml_dir / "ratings.dat"
    return pd.read_csv(
        ratings_path,
        sep="::",
        engine="python",
        names=["user_id", "item_id", "rating", "timestamp"],
        encoding="latin-1",
    )


def _encode_ids(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, int], dict[int, int]]:
    user_values = sorted(df["user_id"].unique().tolist())
    item_values = sorted(df["item_id"].unique().tolist())
    user_id_map = {raw_id: idx for idx, raw_id in enumerate(user_values)}
    item_id_map = {raw_id: idx for idx, raw_id in enumerate(item_values)}
    out = df.copy()
    out["client_id"] = out["user_id"].map(user_id_map).astype(int)
    out["item_idx"] = out["item_id"].map(item_id_map).astype(int)
    return out, user_id_map, item_id_map


def _build_clients(df: pd.DataFrame) -> dict[int, ClientDataset]:
    clients: dict[int, ClientDataset] = {}
    for client_id, group in df.groupby("client_id"):
        group = group.sort_values("timestamp")
        clients[int(client_id)] = ClientDataset(
            client_id=int(client_id),
            item_ids=torch.tensor(group["item_idx"].to_numpy(), dtype=torch.long),
            ratings=torch.tensor(group["rating"].to_numpy(), dtype=torch.float32),
            timestamps=torch.tensor(group["timestamp"].to_numpy(), dtype=torch.long),
        )
    return clients


def load_movielens_1m_clients(
    data_dir: str | Path,
    min_ratings_per_user: int = 20,
    user_split: dict[str, float] | None = None,
    seed: int = 42,
) -> MovieLensData:
    if user_split is None:
        user_split = {"train": 0.8, "val": 0.1, "test": 0.1}

    df = read_ratings_ml1m(data_dir)
    user_counts = df.groupby("user_id").size()
    eligible_users = user_counts[user_counts >= min_ratings_per_user].index
    df = df[df["user_id"].isin(eligible_users)].reset_index(drop=True)
    df, user_id_map, item_id_map = _encode_ids(df)

    all_client_ids = np.array(sorted(df["client_id"].unique().tolist()))
    rng = np.random.default_rng(seed)
    rng.shuffle(all_client_ids)

    n = len(all_client_ids)
    n_train = int(round(n * user_split["train"]))
    n_val = int(round(n * user_split["val"]))

    train_ids = set(all_client_ids[:n_train].tolist())
    val_ids = set(all_client_ids[n_train:n_train + n_val].tolist())
    test_ids = set(all_client_ids[n_train + n_val:].tolist())

    return MovieLensData(
        train_clients=_build_clients(df[df["client_id"].isin(train_ids)]),
        val_clients=_build_clients(df[df["client_id"].isin(val_ids)]),
        test_clients=_build_clients(df[df["client_id"].isin(test_ids)]),
        num_items=len(item_id_map),
        num_users=len(user_id_map),
        user_id_map=user_id_map,
        item_id_map=item_id_map,
    )
