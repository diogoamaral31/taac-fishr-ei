"""Extrai features (2048-d, antes da camada de classificação) da ResNet-50
pré-treinada no ImageNet para treino/validação/teste do Waterbirds.

Saída em data/waterbirds_feats/:
    {split}_x.npy      float32 (N, 2048)
    {split}_y.npy      int64   0 = pássaro terrestre, 1 = pássaro aquático
    {split}_place.npy  int64   0 = fundo terra, 1 = fundo água
    {split}_g.npy      int64   grupo = 2*y + place   (convenção do group_DRO)
        g0 = terrestre/terra, g1 = terrestre/água (minoritário),
        g2 = aquático/terra (minoritário), g3 = aquático/água

Pré-processamento igual ao do DFR/group_DRO: Resize(256) -> CenterCrop(224) -> normalização ImageNet.

Uso:
    python waterbirds/extract_features.py
    python waterbirds/extract_features.py --root data/waterbird_complete95_forest2water2   # tarball group_DRO
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights, resnet50

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "waterbirds_feats"
SPLITS = {0: "train", 1: "val", 2: "test"}


class Imgs(Dataset):
    def __init__(self, root, df, tf):
        self.root, self.df, self.tf = root, df.reset_index(drop=True), tf

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        path = self.root / self.df.loc[i, "img_filename"]
        return self.tf(Image.open(path).convert("RGB"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "data" / "waterbirds_v1.0"))
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    root = Path(args.root)
    meta = pd.read_csv(root / "metadata.csv")
    # colunas esperadas: img_filename, y, split (0/1/2), place
    for c in ("img_filename", "y", "split", "place"):
        assert c in meta.columns, f"metadata.csv sem coluna {c}: {list(meta.columns)}"

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev)
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity()
    model.eval().to(dev)

    tf = T.Compose([
        T.Resize(256), T.CenterCrop(224), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    OUT.mkdir(parents=True, exist_ok=True)
    for sid, name in SPLITS.items():
        df = meta[meta["split"] == sid]
        dl = DataLoader(Imgs(root, df, tf), batch_size=args.batch_size,
                        num_workers=args.workers, shuffle=False)
        feats = []
        with torch.no_grad(), torch.autocast(dev, enabled=(dev == "cuda")):
            for k, xb in enumerate(dl):
                feats.append(model(xb.to(dev)).float().cpu().numpy())
                if k % 10 == 0:
                    print(f"{name}: {k * args.batch_size}/{len(df)}", flush=True)
        y = df["y"].to_numpy().astype(np.int64)
        place = df["place"].to_numpy().astype(np.int64)
        np.save(OUT / f"{name}_x.npy", np.concatenate(feats).astype(np.float32))
        np.save(OUT / f"{name}_y.npy", y)
        np.save(OUT / f"{name}_place.npy", place)
        np.save(OUT / f"{name}_g.npy", 2 * y + place)
        print(name, "grupos:", np.bincount(2 * y + place, minlength=4))


if __name__ == "__main__":
    main()
