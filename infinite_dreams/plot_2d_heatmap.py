import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/eval_log.csv")
    ap.add_argument("--outdir", default="results/tmtn_heatmaps")
    ap.add_argument("--ms_max", type=float, default=1e8, help="filter invalid/sentinel ms")
    ap.add_argument("--phase", type=str, default="search", choices=["search", "confirm", "all"])
    ap.add_argument("--collapse_by_cfg", action="store_true",
                    help="Collapse duplicates by cfg_id keeping best ms per cfg_id first (optional).")
    args = ap.parse_args()

    ensure_dir(args.outdir)
    df = pd.read_csv(args.csv)

    # Optional phase filter
    if args.phase != "all":
        df = df[df["phase"] == args.phase].copy()

    # Filter invalid/sentinel
    df = df[df["ms"] < args.ms_max].copy()

    # Optional collapse duplicates per cfg_id
    if args.collapse_by_cfg:
        if "cfg_id" not in df.columns:
            raise RuntimeError("--collapse_by_cfg requires cfg_id column")
        df = df.sort_values("ms").groupby("cfg_id", as_index=False).first()

    required = {"tile_m", "tile_n", "tile_k", "tm", "tn", "ms"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"CSV missing columns: {sorted(missing)}")

    tms = sorted(df["tm"].unique())
    tns = sorted(df["tn"].unique())

    for tm in tms:
        for tn in tns:
            sub = df[(df.tm == tm) & (df.tn == tn)].copy()
            if sub.empty:
                continue

            # For each (tile_m, tile_n, tile_k): keep best ms (if repeated evals exist)
            sub = sub.groupby(["tile_m", "tile_n", "tile_k"], as_index=False)["ms"].min()

            # For each (tile_m, tile_n): find best tile_k (argmin) and best ms
            idx = sub.groupby(["tile_m", "tile_n"])["ms"].idxmin()
            best = sub.loc[idx, ["tile_m", "tile_n", "tile_k", "ms"]].copy()

            # Pivot tables
            piv_ms = best.pivot(index="tile_m", columns="tile_n", values="ms").sort_index().sort_index(axis=1)
            piv_k = best.pivot(index="tile_m", columns="tile_n", values="tile_k").sort_index().sort_index(axis=1)

            # ---------------------------
            # 1) Min-ms heatmap
            # ---------------------------
            plt.figure()
            im = plt.imshow(piv_ms.to_numpy(), aspect="auto", interpolation="nearest")
            plt.colorbar(im, label="min ms over tile_k")
            plt.xticks(np.arange(piv_ms.shape[1]), piv_ms.columns.to_numpy(), rotation=90)
            plt.yticks(np.arange(piv_ms.shape[0]), piv_ms.index.to_numpy())
            plt.xlabel("tile_n");
            plt.ylabel("tile_m")
            plt.title(f"tm={tm}, tn={tn}: min(ms) over tile_k")
            out_ms = os.path.join(args.outdir, f"heatmap_tm{tm}_tn{tn}_min_ms.png")
            plt.savefig(out_ms, dpi=180, bbox_inches="tight")
            plt.close()

            # ---------------------------
            # Argmin tile_k heatmap
            # ---------------------------
            # Use a discrete colormap by showing tile_k as numbers.
            # (We avoid custom colors; the colorbar will still show distinct bands.)
            k_vals = piv_k.to_numpy()
            plt.figure()
            im2 = plt.imshow(k_vals, aspect="auto", interpolation="nearest")
            cbar = plt.colorbar(im2, label="argmin tile_k")
            plt.xticks(np.arange(piv_k.shape[1]), piv_k.columns.to_numpy(), rotation=90)
            plt.yticks(np.arange(piv_k.shape[0]), piv_k.index.to_numpy())
            plt.xlabel("tile_n");
            plt.ylabel("tile_m")
            plt.title(f"tm={tm}, tn={tn}: tile_k that minimizes ms")

            out_k = os.path.join(args.outdir, f"heatmap_tm{tm}_tn{tn}_argmin_tilek.png")
            plt.savefig(out_k, dpi=180, bbox_inches="tight")
            plt.close()

    print("Wrote heatmaps to:", args.outdir)


if __name__ == "__main__":
    main()
