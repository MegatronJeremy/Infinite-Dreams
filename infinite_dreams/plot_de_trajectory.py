import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FEATS = ["tile_m", "tile_n", "tile_k", "tm", "tn"]


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def pca_fit_transform(X: np.ndarray) -> np.ndarray:
    # X: [N,5], already normalized
    Xc = X - X.mean(axis=0, keepdims=True)
    C = (Xc.T @ Xc) / max(1, (Xc.shape[0] - 1))
    w, V = np.linalg.eigh(C)  # ascending eigenvalues
    V2 = V[:, -2:]  # top-2
    return Xc @ V2  # [N,2]


def mean_pairwise_l2(X: np.ndarray) -> float:
    # X: [P,5] normalized
    P = X.shape[0]
    if P <= 1:
        return 0.0
    # compute efficiently: sum_{i<j} ||xi-xj|| / num_pairs
    s = 0.0
    cnt = 0
    for i in range(P):
        d = X[i + 1:] - X[i]
        s += np.sqrt((d * d).sum(axis=1)).sum()
        cnt += (P - i - 1)
    return float(s / max(1, cnt))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/eval_log.csv")
    ap.add_argument("--outdir", default="results/de_plots")
    ap.add_argument("--phase", default="search", choices=["search", "confirm", "all"])
    ap.add_argument("--ms_max", type=float, default=1e8, help="filter sentinel/invalid ms")
    ap.add_argument("--pop_mode", default="best_per_gen_popi",
                    choices=["best_per_gen_popi", "all_rows"],
                    help="How to choose population points per generation.\n"
                         "best_per_gen_popi: keep min(ms) for each (gen,pop_i)\n"
                         "all_rows: use all rows (noisy, but shows all tries)")
    args = ap.parse_args()

    ensure_dir(args.outdir)
    df = pd.read_csv(args.csv)

    # Phase filter
    if args.phase != "all":
        df = df[df["phase"].astype(str).str.startswith(args.phase)].copy()

    # Filter invalid
    df = df[df["ms"] < args.ms_max].copy()

    # Require generation info
    if "gen" not in df.columns:
        raise RuntimeError("CSV must contain 'gen' column to plot DE trajectory.")
    if "pop_i" not in df.columns:
        raise RuntimeError("CSV must contain 'pop_i' column to plot DE trajectory.")

    # Keep only rows with gen>=0 (exclude init/confirm if you logged them with -1)
    df = df[df["gen"] >= 0].copy()

    # Pick representative population point per (gen, pop_i)
    if args.pop_mode == "best_per_gen_popi":
        df = df.sort_values("ms").groupby(["gen", "pop_i"], as_index=False).first()

    # Build normalized feature matrix for PCA and diversity
    X = df[FEATS].to_numpy(dtype=np.float32)
    Xn = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-6)
    Y = pca_fit_transform(Xn)

    gens = df["gen"].to_numpy(dtype=np.int32)
    ms = df["ms"].to_numpy(dtype=np.float32)

    # ---------------------------
    # PCA trajectory scatter
    # ---------------------------
    plt.figure()
    sc = plt.scatter(Y[:, 0], Y[:, 1], c=gens, s=18, alpha=0.7)
    plt.colorbar(sc, label="generation")
    plt.xlabel("PC1 (normalized 5D)")
    plt.ylabel("PC2")
    plt.title("DE population trajectory (PCA projection, colored by generation)")
    out = os.path.join(args.outdir, "de_pca_trajectory.png")
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()

    # ---------------------------
    # 2) Best/median/worst per gen
    # ---------------------------
    stats = df.groupby("gen")["ms"].agg(["min", "median", "max"]).reset_index()
    plt.figure()
    plt.plot(stats["gen"].to_numpy(), stats["min"].to_numpy(), label="best (min)")
    plt.plot(stats["gen"].to_numpy(), stats["median"].to_numpy(), label="median")
    plt.plot(stats["gen"].to_numpy(), stats["max"].to_numpy(), label="worst (max)")
    plt.xlabel("generation")
    plt.ylabel("ms")
    plt.title("Population fitness vs generation")
    plt.legend()
    out = os.path.join(args.outdir, "de_fitness_stats.png")
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()

    # ---------------------------
    # 3) Diversity vs generation
    # ---------------------------
    # Compute on normalized 5D, per generation
    div = []
    for g in sorted(df["gen"].unique()):
        sub = df[df["gen"] == g]
        Xg = sub[FEATS].to_numpy(dtype=np.float32)
        Xg = (Xg - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-6)
        div.append((g, mean_pairwise_l2(Xg)))

    div = np.array(div, dtype=np.float32)
    plt.figure()
    plt.plot(div[:, 0], div[:, 1])
    plt.xlabel("generation")
    plt.ylabel("mean pairwise distance (normalized 5D)")
    plt.title("Population diversity vs generation")
    out = os.path.join(args.outdir, "de_diversity.png")
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()

    print("Wrote:", args.outdir)


if __name__ == "__main__":
    main()
