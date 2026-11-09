"""Descarrega o Waterbirds via pacote wilds para data/ (fora do git).

Uso:
    python waterbirds/download_data.py

Resultado: data/waterbirds_v1.0/{metadata.csv, <pastas de imagens>}
Se o download do wilds falhar: usar o tarball do group_DRO
(https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz)
e extrair para data/waterbird_complete95_forest2water2/  (ver --root em extract_features.py).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def main():
    from wilds import get_dataset

    DATA.mkdir(exist_ok=True)
    ds = get_dataset(dataset="waterbirds", root_dir=str(DATA), download=True)
    print("OK:", ds.data_dir)
    for split in ("train", "val", "test"):
        print(split, len(ds.get_subset(split)))


if __name__ == "__main__":
    main()
