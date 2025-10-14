import random
import galois
import numpy as np
import pulp as pl
import time
import math
from collections.abc import Iterable
from typing import Dict, Any, List, Tuple
from collections import Counter
import os
cplex_bin = os.environ["CPEX_BIN"]
threads = int(os.environ.get("SLURM_CPUS_PER_TASK", "16"))
def solve_mod2_mip_weighted_per_spans(
    encoding_parity_matrix,      # A: list of m rows, each length k (ints, mod-2)
    original_message_bits,       # full length n = k + m (list of 0/1)
    original_bits_weights,       # length n; per-bit positions (e.g., 0..bit_size-1)
    message_indices,             # length k; indices into [0..n-1]
    parity_indices,              # length m; indices into [0..n-1]
    number_spans,                # list[dict] with {"start": int, "end": int, ...} for each number in THIS CHUNK
    time_limit_sec=30,
    use_cplex=True,
    warm_start=None,
    cplex_path=r"C://Program Files/IBM/ILOG/CPLEX_Studio2211/cplex/bin/x64_win64/cplex.exe",
):
    """
    Minimize sum_t D_t subject to:
        For each span t with [start_t, end_t):
            S_t = sum_{i in span_t} ( (1<<original_bits_weights[i]) * (y_i - orig_i) )
            -D_t <= S_t <= D_t

        GF(2) parity constraints:
            For each row r:  sum_j A[r][j]*y[message_indices[j]] - y[parity_indices[r]] = 2*z_r

    Returns:
        mutated_bits (list[int] length n), objective_value (float), solver_status (str)
        or (None, None, status) on failure.
    """
    # print("Available solvers:", pl.listSolvers())
    # ---------- Validate & normalize ----------
    # orig = [int(b) & 1 for b in original_message_bits]
    orig = original_message_bits
    n = len(orig)

    # if len(original_bits_weights) != n:
    #     raise ValueError("original_bits_weights must have same length as original_message_bits.")

    # significance per bit index: 2^(bit_position)
    w = [float(1 << int(p)) for p in original_bits_weights]

    k = len(message_indices)
    m = len(parity_indices)
    # if k <= 0 or m <= 0 or n != k + m:
    #     raise ValueError(f"Expected n = k + m with k>0,m>0; got n={n}, k={k}, m={m}.")

    # Distinct index sets, in-bounds
    # if any(i < 0 or i >= n for i in message_indices) or any(i < 0 or i >= n for i in parity_indices):
    #     raise ValueError("message_indices / parity_indices contain out-of-bounds indices.")
    # if set(message_indices) & set(parity_indices):
    #     raise ValueError("message_indices and parity_indices must be disjoint.")
    # if len(set(message_indices)) != k or len(set(parity_indices)) != m:
    #     raise ValueError("message_indices / parity_indices contain duplicates.")

    # Validate spans
    # if not isinstance(number_spans, list) or not number_spans:
    #     raise ValueError("number_spans must be a non-empty list of span dicts.")
    # for t, rec in enumerate(number_spans):
    #     if not isinstance(rec, dict) or "start" not in rec or "end" not in rec:
    #         raise ValueError(f"Span {t} must have 'start' and 'end'.")
    #     s, e = int(rec["start"]), int(rec["end"])
    #     if not (0 <= s <= e <= n):
    #         raise ValueError(f"Span {t} out of bounds or invalid: [{s}, {e}).")

    # Parity matrix A
    A2 = encoding_parity_matrix
    # if len(A) != m:
    #     raise ValueError(f"Parity matrix row count must be m={m}, got {len(A)}.")
    # for r in range(m):
    #     if len(A[r]) != k:
    #         raise ValueError(f"Row {r} of A must have length k={k}, got {len(A[r])}.")
    # Reduce A mod-2 (robustness)
    # A2 = [[int(a) & 1 for a in row] for row in A]

    # ---------- Model ----------
    model = pl.LpProblem("weighted_mod2_parity_mip_per_spans", pl.LpMinimize)

    # y[i] = mutated bit (binary) for the full vector
    y = [pl.LpVariable(f"y_{i}", lowBound=0, upBound=1, cat=pl.LpBinary) for i in range(n)]

    # z_r integers for mod-2 parity equalities
    z = []
    for r in range(m):
        row_sum = sum(A2[r])  # max sum when all xm_j=1
        z_lb, z_ub = 0, int(math.ceil(row_sum / 2))  # safe static bound
        z.append(pl.LpVariable(f"z_{r}", lowBound=z_lb, upBound=z_ub, cat=pl.LpInteger))

    # Parity constraints: Σ_j A[r][j]*y[msg_j] - y[par_r] = 2*z_r
    for r in range(m):
        lhs = pl.lpSum(A2[r][j] * y[message_indices[j]] for j in range(k)) - y[parity_indices[r]]
        model += (lhs == 2 * z[r])

    # Per-number absolute deviations using spans
    D = []
    for t, rec in enumerate(number_spans):
        s, e = int(rec["start"]), int(rec["end"])
        D_t = pl.LpVariable(f"D_{t}", lowBound=0, cat=pl.LpContinuous)
        D.append(D_t)
        S_t = pl.lpSum(w[i] * (y[i] - orig[i]) for i in range(s, e))
        model += D_t >=  S_t
        model += D_t >= -S_t

    # Objective: minimize sum_t D_t
    model += pl.lpSum(D)

    # ---------- Warm start (optional) ----------
    if warm_start is not None:
        if len(warm_start) != n:
            raise ValueError(f"warm_start length {len(warm_start)} != n {n}")
        ws = [int(b) & 1 for b in warm_start]
        for i in range(n):
            y[i].setInitialValue(ws[i])
        # derive z from warm start
        for r in range(m):
            s_val = sum(A2[r][j] * ws[message_indices[j]] for j in range(k))
            z0 = (s_val - ws[parity_indices[r]]) // 2
            # clamp to bounds
            z_lb = int(z[r].lowBound) if z[r].lowBound is not None else None
            z_ub = int(z[r].upBound) if z[r].upBound is not None else None
            if z_lb is not None: z0 = max(z0, z_lb)
            if z_ub is not None: z0 = min(z0, z_ub)
            z[r].setInitialValue(z0)

    # ---------- Solve ----------
    if use_cplex:
        # cplex_bin = '/home/vkamineni/MILP_graph_CPLEX/CPLEX/cplex/bin/x86-64_linux/cplex'
        # cplex_bin = '/opt/ibm/ILOG/CPLEX_Studio2211/cplex/bin/x86-64_linux/cplex'
        # cplex_bin = "/apps/cplex/12.9/cplex/bin/x86-64_linux/cplex"
        # opts = [
        #     "set parallel 1",
        #     f"set threads {threads}",
        #     "set mip display 0",
        #     "set mip tolerances mipgap 0.01",
        #     "set workmem 4096",
        #     "set mip limits treememory 64000",    MB for search tree (≈ half of --mem)
        #     # Optional spill:
        #     # "set mip strategy file 1",
        # ]
        # print('threads',threads)
        options = [
            "set parallel -1",                   # 1=deterministic, 0=opportunistic
            f"set threads {threads}",
            "set mip display 0",                # 0..5; 3 is informative without flooding
            "set mip tolerances mipgap 0.01",   # 1% target gap (adjust)
            # "set workmem 8192",                 # MB per worker for node processing
            # "set mip strategy heuristiceffort 1",  # slightly more primal heuristics
            # Optional spill:
            # "set mip strategy file 1",  # write compressed node files to disk
            "set output clonelog -1",
            "set logfile *", 
        ]
        solver = pl.CPLEX_CMD(path=cplex_bin, msg=False, timeLimit=time_limit_sec, options=options)
        # solver = pl.CPLEX_PY(msg=False)
    else:
        solver = pl.PULP_CBC_CMD(msg=False, timeLimit=time_limit_sec, warmStart=True)
    status = model.solve(solver)
    status_str = pl.LpStatus[status]

    y_vals = [pl.value(v) for v in y]
    obj_val = pl.value(model.objective)
    if any(v is None for v in y_vals) or obj_val is None:
        return None, None, status_str

    mutated = [int(round(v)) for v in y_vals]
    return mutated, float(obj_val), status_str