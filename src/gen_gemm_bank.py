# gen_gemm_bank.py
import math
from collections import defaultdict
from itertools import product
from pathlib import Path

# ---------- knobs ----------
SMEM_CAP = 48 * 1024  # per-block shared limit (conservative)
CAP = 10000  # set None for no cap

# Tile dimensions: multiples of 8
tile_ms = list(range(8, 257, 8))
tile_ns = list(range(8, 257, 8))
tile_ks = list(range(8, 257, 8))

# Register blocking: power of 2
vals = [1, 2, 4, 8]
tm_tn = [(tm, tn) for tm in vals for tn in vals]

# Optional: forbid super tiny tiles that are almost always bad
MIN_TILE_AREA = 16 * 16

# Prevent registers from exploding
MAX_THREAD_TILE = 16

# --- Bank sizing knobs ---
TARGET_CFGS_PER_BANK = 100

# Output folder for generated CUDA files/includes/headers
OUT_DIR = Path("src/generated")


# ---------- end knobs ----------


def smem_bytes(Tm, Tn, Tk) -> int:
    # As: Tm*Tk half, Bs: Tk*Tn half  => total half count = Tk*(Tm+Tn)
    return 2 * (Tk * (Tm + Tn))  # sizeof(half)=2


def valid(Tm, Tn, Tk, tm, tn) -> bool:
    # divisibility (so blockDim is integral)
    if Tm % tm != 0 or Tn % tn != 0:
        return False

    # threads per block
    threads = (Tm // tm) * (Tn // tn)
    if threads <= 0 or threads > 1024:
        return False

    # shared memory per block
    if smem_bytes(Tm, Tn, Tk) > SMEM_CAP:
        return False

    # avoid pathological tiny tiles
    if (Tm * Tn) < MIN_TILE_AREA:
        return False

    if (tm * tn) > MAX_THREAD_TILE:
        return False

    return True


def score(cfg):
    """Heuristic: prefer bigger output tiles, moderate K, moderate per-thread work."""
    Tm, Tn, Tk, tm, tn = cfg
    tile_area = Tm * Tn
    micro = tm * tn
    return (tile_area, -Tk, -micro)


def flatten_round_robin(cfgs):
    """
    Deterministic 'good spread' ordering:
    - Bucket by (Tm,Tn,Tk)
    - Sort within bucket by score
    - Flatten round-robin across buckets
    """
    buckets = defaultdict(list)
    for c in cfgs:
        buckets[(c[0], c[1], c[2])].append(c)

    for k in buckets:
        buckets[k].sort(key=score, reverse=True)

    keys = sorted(buckets.keys(), key=lambda k: (k[0] * k[1], k[2]), reverse=True)

    flat = []
    i = 0
    while True:
        progressed = False
        for k in keys:
            if i < len(buckets[k]):
                flat.append(buckets[k][i])
                progressed = True
        if not progressed:
            break
        i += 1

    return flat


def compute_num_banks(cfgs, target_cfgs_per_bank: int) -> int:
    n = int(math.ceil(len(cfgs) / target_cfgs_per_bank))
    return n


def pack_banks_adaptive(flat_cfgs, num_banks: int):
    """
    Capacity-constrained packing:
    - each bank has a maximum compile complexity budget
    - if all banks are full → create a new bank
    This prevents NVCC ICE crashes.
    """

    banks = [[]]
    loads = [0.0]

    for c in flat_cfgs:
        w = cfg_compile_weight(c)

        # find first bank that fits
        placed = False
        for i in range(len(banks)):
            if loads[i] + w <= MAX_BANK_COMPILE_WEIGHT:
                banks[i].append(c)
                loads[i] += w
                placed = True
                break

        # if none fits → create new bank
        if not placed:
            banks.append([c])
            loads.append(w)

    return banks, loads


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # enumerate feasible cfgs
    cfgs = []
    for Tm, Tn, Tk, (tm, tn) in product(tile_ms, tile_ns, tile_ks, tm_tn):
        if valid(Tm, Tn, Tk, tm, tn):
            cfgs.append((Tm, Tn, Tk, tm, tn))

    # deterministic base order
    cfgs.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4]))

    # cap with diversity
    if CAP is not None and len(cfgs) > CAP:
        buckets = defaultdict(list)
        for c in cfgs:
            buckets[(c[0], c[1], c[2])].append(c)

        for k in buckets:
            buckets[k].sort(key=score, reverse=True)

        selected = []
        keys = sorted(buckets.keys(), key=lambda k: (k[0] * k[1], k[2]), reverse=True)

        # first pass: take up to 2 per bucket
        for k in keys:
            selected.extend(buckets[k][:2])
            if len(selected) >= CAP:
                break

        # fill remaining round-robin
        if len(selected) < CAP:
            i = 2
            while len(selected) < CAP:
                progressed = False
                for k in keys:
                    if i < len(buckets[k]):
                        selected.append(buckets[k][i])
                        progressed = True
                        if len(selected) >= CAP:
                            break
                if not progressed:
                    break
                i += 1

        cfgs = sorted(set(selected), key=lambda x: (x[0], x[1], x[2], x[3], x[4]))

    num_cfgs = len(cfgs)
    print("Generated", num_cfgs, "configs")

    # derive number of banks from cfgs (and compile-weight)
    NUM_BANKS = compute_num_banks(cfgs, TARGET_CFGS_PER_BANK)

    # adaptive split into banks (balanced by compile-weight)
    flat = flatten_round_robin(cfgs)
    banks, loads = pack_banks_adaptive(flat, NUM_BANKS)
    NUM_BANKS = len(banks)

    avg_cfgs = (num_cfgs / NUM_BANKS) if NUM_BANKS > 0 else 0.0
    avg_load = (sum(loads) / NUM_BANKS) if NUM_BANKS > 0 else 0.0
    min_cfgs = min((len(b) for b in banks), default=0)
    max_cfgs = max((len(b) for b in banks), default=0)
    min_load = min(loads) if loads else 0.0
    max_load = max(loads) if loads else 0.0

    print(
        f"Using {NUM_BANKS} banks | "
        f"cfgs/bank avg={avg_cfgs:.1f} min={min_cfgs} max={max_cfgs} | "
        f"load/bank avg={avg_load:.3f} min={min_load:.3f} max={max_load:.3f}"
    )

    # write extern declarations include
    decl_path = OUT_DIR / "gemm_dispatch_decl_gen.inc"
    with open(decl_path, "w", newline="\n") as f:
        f.write("// AUTO-GENERATED by gen_gemm_bank.py\n")
        f.write("// Declarations for all bank entry points (include at namespace scope)\n\n")
        for i in range(NUM_BANKS):
            f.write(
                f'extern "C" bool gemm_try_bank_{i:03d}('
                f'const half* A, const half* B, half* C, '
                f'int M, int N, int K, '
                f'int tile_m, int tile_n, int tile_k, int tm, int tn, '
                f'cudaStream_t stream);\n'
            )

    # write dispatcher call-chain include (Option B: statement list; avoids MSVC parser overflow)
    calls_path = OUT_DIR / "gemm_dispatch_calls_gen.inc"
    with open(calls_path, "w", newline="\n") as f:
        f.write("// AUTO-GENERATED by gen_gemm_bank.py\n")
        f.write("// Try banks in order until one matches (short-circuit via if(!launched))\n\n")

        f.write("bool launched = false;\n")
        for i in range(NUM_BANKS):
            f.write(
                f"if (!launched) launched = gemm_try_bank_{i:03d}"
                f"(A,B,C,M,N,K,tile_m,tile_n,tile_k,tm,tn,stream);\n"
            )

    # write C++ menu header for cfg_id path
    menu_path = OUT_DIR / "gemm_cfg_menu.h"
    with open(menu_path, "w", newline="\n") as f:
        f.write("// AUTO-GENERATED by gen_gemm_bank.py\n")
        f.write("#pragma once\n")
        f.write("#include <vector>\n\n")
        f.write("struct GemmCfg { int tile_m, tile_n, tile_k, tm, tn; };\n\n")
        f.write("static inline const std::vector<GemmCfg>& gemm_cfg_menu() {\n")
        f.write("    static const std::vector<GemmCfg> kMenu = {\n")
        for Tm, Tn, Tk, tm, tn in cfgs:
            f.write(f"        {{{Tm}, {Tn}, {Tk}, {tm}, {tn}}},\n")
        f.write("    };\n")
        f.write("    return kMenu;\n")
        f.write("}\n")

    # write bank .cu files
    for i, bank_cfgs in enumerate(banks):
        bank_path = OUT_DIR / f"gemm_bank_{i:03d}.cu"
        with open(bank_path, "w", newline="\n") as f:
            f.write("// AUTO-GENERATED by gen_gemm_bank.py\n")
            f.write('#include "../gemm_kernels.cuh"\n\n')
            f.write(f'extern "C" bool gemm_try_bank_{i:03d}(\n')
            f.write("    const half* A,\n")
            f.write("    const half* B,\n")
            f.write("    half* C,\n")
            f.write("    int M, int N, int K,\n")
            f.write("    int tile_m, int tile_n, int tile_k,\n")
            f.write("    int tm, int tn,\n")
            f.write("    cudaStream_t stream\n")
            f.write(") {\n")
            for (Tm, Tn, Tk, tm, tn) in bank_cfgs:
                f.write(f"    TRY_CFG({Tm}, {Tn}, {Tk}, {tm}, {tn});\n")
            f.write("    return false;\n")
            f.write("}\n")


if __name__ == "__main__":
    main()
