"""Formato único dos CSV de resultados (results/raw/*.csv).

Colunas: dataset,method,setting,seed,train_acc,test_acc,worst_group_acc,partition_purity
Células sem valor ficam vazias. Valores de exatidão em [0, 1].
"""
import csv
from pathlib import Path

COLUMNS = ["dataset", "method", "setting", "seed", "train_acc", "test_acc",
           "worst_group_acc", "partition_purity"]
RAW_DIR = Path(__file__).resolve().parents[1] / "results" / "raw"


def write_rows(name: str, rows: list[dict], append: bool = False) -> Path:
    """Escreve results/raw/<name>.csv. `rows` = lista de dicts com (um subconjunto de) COLUMNS."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{name}.csv"
    new = not (append and path.exists())
    with open(path, "a" if append else "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore", restval="")
        if new:
            w.writeheader()
        w.writerows(rows)
    return path


def read_all():
    """Junta todos os CSV de results/raw/ num DataFrame."""
    import pandas as pd
    files = sorted(RAW_DIR.glob("*.csv"))
    return pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
