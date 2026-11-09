"""ERM / Fishr (ambientes verdadeiros) / Fishr (ambientes aleatórios) sobre features fixas
da ResNet-50 no Waterbirds. O classificador é uma regressão logística (última camada).

Gradiente por exemplo na última camada (forma fechada, sem BackPACK):
    g_i = (p_i - y_i) * [z_i, 1]          (z_i = feature padronizada, 1 = bias)
Penalização Fishr: variância por dimensão de g_i dentro de cada ambiente, v_e;
    pen = média_e  mean_dim (v_e - v_medio)^2
Perda: média_e NLL_e + lambda * pen   (com o anneal do repo oficial: lambda=1 antes de
`anneal` passos, depois lambda e a perda é dividida por lambda; o otimizador é reiniciado).

Ambientes de treino (ver README.md desta pasta para a regra completa):
    true   : grupos maioritários (g0,g3) divididos 50/50; grupos minoritários (g1,g2)
             com fração `--minority_frac_env2` (0.9) no ambiente 2 e o resto no 1.
    random : o treino todo baralhado e dividido em duas metades (sem usar grupos).

Seleção de modelo: lambda e l2 escolhidos pela worst-group accuracy da VALIDAÇÃO
(média das seeds). O teste nunca entra na escolha.

Uso:
    python waterbirds/train_last_layer.py                  # grelha completa, 5 seeds
    python waterbirds/train_last_layer.py --seeds 0 1 --quick
"""
import argparse
import csv
import itertools
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
FEATS = ROOT / "data" / "waterbirds_feats"
RAW = ROOT / "results" / "raw"
GRID_CSV = Path(__file__).resolve().parent / "grid_val.csv"
FINAL_CSV = RAW / "waterbirds.csv"
COLS = ["dataset", "method", "setting", "seed", "train_acc", "test_acc",
        "worst_group_acc", "partition_purity"]

try:  # lista de seeds do grupo, se já existir
    import sys
    sys.path.insert(0, str(ROOT))
    from shared.seeds import SEEDS_WATERBIRDS as DEFAULT_SEEDS  # noqa
except Exception:
    DEFAULT_SEEDS = [0, 1, 2, 3, 4]

DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------------- dados
def load(split):
    x = np.load(FEATS / f"{split}_x.npy")
    return x, np.load(FEATS / f"{split}_y.npy"), np.load(FEATS / f"{split}_g.npy")


def make_envs(g, seed, rule, minority_frac_env2):
    """Devolve array env (0/1) com o ambiente de cada exemplo de treino."""
    rng = np.random.RandomState(1000 + seed)
    env = np.zeros(len(g), dtype=np.int64)
    if rule == "random":
        perm = rng.permutation(len(g))
        env[perm[len(g) // 2:]] = 1
        return env
    assert rule == "true"
    for grp in range(4):
        idx = rng.permutation(np.where(g == grp)[0])
        frac_env2 = minority_frac_env2 if grp in (1, 2) else 0.5
        env[idx[: int(round(frac_env2 * len(idx)))]] = 1  # 1 = "ambiente 2"
    return env


def describe_envs(g, env):
    out = []
    for e in (0, 1):
        ge = g[env == e]
        n = len(ge)
        minority = np.isin(ge, (1, 2)).mean()
        out.append(f"env{e + 1}: n={n}, grupos={np.bincount(ge, minlength=4).tolist()}, "
                   f"corr. atalho={1 - minority:.3f}")
    return " | ".join(out)


# ----------------------------------------------------------------------------- modelo
def fishr_penalty(w, b, xs, ys):
    """xs, ys: listas (uma entrada por ambiente) de tensores do mini-batch."""
    vs = []
    for x, y in zip(xs, ys):
        p = torch.sigmoid(x @ w + b)
        r = (p - y)[:, None]
        g = torch.cat([r * x, r], dim=1)          # (B, d+1) gradientes por exemplo
        vs.append(g.var(dim=0, unbiased=False))   # variância por dimensão
    v = torch.stack(vs)
    return ((v - v.mean(0, keepdim=True)) ** 2).mean(dim=1).mean()


def train_one(X, y, env, method, lam, l2, seed, steps, anneal, bs, lr):
    torch.manual_seed(seed)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    d = X.shape[1]
    w = (0.01 * torch.randn(d, device=DEV)).requires_grad_()
    b = torch.zeros((), device=DEV, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr)
    envs_idx = [torch.where(env == e)[0] for e in (0, 1)]
    n = len(y)

    for step in range(steps):
        if method == "erm":  # amostra do treino todo (uniforme), mesmo orçamento que Fishr
            idx = torch.randint(0, n, (2 * bs,), generator=gen).to(DEV)
            xb, yb = X[idx], y[idx]
            loss = F.binary_cross_entropy_with_logits(xb @ w + b, yb) + l2 * (w ** 2).sum()
        else:
            xs, ys, nlls = [], [], []
            for ei in envs_idx:
                idx = ei[torch.randint(0, len(ei), (bs,), generator=gen).to(DEV)]
                xs.append(X[idx]); ys.append(y[idx])
                nlls.append(F.binary_cross_entropy_with_logits(X[idx] @ w + b, y[idx]))
            pen = fishr_penalty(w, b, xs, ys)
            pw = lam if step >= anneal else 1.0
            loss = torch.stack(nlls).mean() + l2 * (w ** 2).sum() + pw * pen
            if pw > 1.0:
                loss = loss / pw
            if step == anneal:  # como no repo oficial: reinicia o otimizador
                opt = torch.optim.Adam([w, b], lr=lr)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return w.detach(), b.detach()


@torch.no_grad()
def group_acc(w, b, X, y, g):
    pred = ((X @ w + b) > 0).long()
    correct = (pred == y.long()).float().cpu().numpy()
    g = g.cpu().numpy()
    return np.array([correct[g == k].mean() for k in range(4)]), correct.mean()


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--anneal", type=int, default=300)
    ap.add_argument("--bs", type=int, default=256, help="exemplos por ambiente (ERM: 2*bs no total)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--minority_frac_env2", type=float, default=0.9)
    ap.add_argument("--lambdas", type=float, nargs="+", default=[1e1, 1e2, 1e3, 1e4, 1e5])
    ap.add_argument("--l2s", type=float, nargs="+", default=[1e-4, 1e-3, 1e-2])
    ap.add_argument("--quick", action="store_true", help="grelha mínima para testar o pipeline")
    args = ap.parse_args()
    if args.quick:
        args.lambdas, args.l2s, args.steps, args.anneal = [1e2], [1e-3], 200, 60

    xtr, ytr, gtr = load("train")
    xva, yva, gva = load("val")
    xte, yte, gte = load("test")
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-6          # padronização só com o treino
    t = lambda a, dt=torch.float32: torch.tensor(a, dtype=dt, device=DEV)  # noqa
    Xtr, Xva, Xte = t((xtr - mu) / sd), t((xva - mu) / sd), t((xte - mu) / sd)
    Ytr, Yva, Yte = t(ytr), t(yva), t(yte)
    Gva, Gte = torch.tensor(gva), torch.tensor(gte)
    train_w = np.bincount(gtr, minlength=4) / len(gtr)   # pesos dos grupos no treino
    print("device:", DEV, "| grupos treino:", np.bincount(gtr, minlength=4).tolist())

    # configurações: (método, regra de ambientes, lambda, l2)
    configs = [("erm", "none", 0.0, l2) for l2 in args.l2s]
    configs += [("fishr", "true", lam, l2) for lam, l2 in itertools.product(args.lambdas, args.l2s)]
    configs += [("fishr_random_envs", "random", lam, l2)
                for lam, l2 in itertools.product(args.lambdas, args.l2s)]

    rows = []  # uma linha por (config, seed), com val_wga para a seleção
    for seed in args.seeds:
        envs = {r: torch.tensor(make_envs(gtr, seed, r, args.minority_frac_env2), device=DEV)
                for r in ("true", "random")}
        envs["none"] = envs["random"]
        print(f"[seed {seed}] true  -> {describe_envs(gtr, envs['true'].cpu().numpy())}")
        print(f"[seed {seed}] rand. -> {describe_envs(gtr, envs['random'].cpu().numpy())}")
        for method, rule, lam, l2 in configs:
            t0 = time.time()
            w, b = train_one(Xtr, Ytr, envs[rule], method, lam, l2, seed,
                             args.steps, args.anneal, args.bs, args.lr)
            _, tr_acc = group_acc(w, b, Xtr, Ytr, torch.tensor(gtr))
            va_g, _ = group_acc(w, b, Xva, Yva, Gva)
            te_g, _ = group_acc(w, b, Xte, Yte, Gte)
            rows.append(dict(
                dataset="waterbirds", method=method,
                setting=f"lambda={lam:g},l2={l2:g}" if method != "erm" else f"l2={l2:g}",
                seed=seed, train_acc=tr_acc,
                test_acc=float((te_g * train_w).sum()),   # média ponderada pela distribuição do treino
                worst_group_acc=float(te_g.min()), partition_purity="",
                val_wga=float(va_g.min()), test_acc_unweighted=float(
                    ((((Xte @ w + b) > 0).float() == Yte).float().mean()).item())))
            print(f"  seed={seed} {method:18s} {rows[-1]['setting']:22s} "
                  f"val_wga={rows[-1]['val_wga']:.3f} test_wga={rows[-1]['worst_group_acc']:.3f} "
                  f"({time.time() - t0:.1f}s)", flush=True)

    # guarda toda a grelha (inclui val_wga; fica na pasta waterbirds/, não em results/)
    with open(GRID_CSV, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)

    # seleção: por método, o setting com maior val_wga médio nas seeds
    final = []
    for method in ("erm", "fishr", "fishr_random_envs"):
        by_setting = {}
        for r in rows:
            if r["method"] == method:
                by_setting.setdefault(r["setting"], []).append(r)
        best = max(by_setting, key=lambda s: np.mean([r["val_wga"] for r in by_setting[s]]))
        print(f"selecionado {method}: {best}  (val_wga médio="
              f"{np.mean([r['val_wga'] for r in by_setting[best]]):.3f})")
        final += by_setting[best]

    RAW.mkdir(parents=True, exist_ok=True)
    with open(FINAL_CSV, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        wr.writeheader(); wr.writerows(final)
    print("escrito:", FINAL_CSV)


if __name__ == "__main__":
    main()
