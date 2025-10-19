# dynamic_parallel.py
import os, json, time, math, pathlib
import multiprocessing as mp

import random
import galois
import numpy as np
import pulp as pl
import time
import os
import math
from collections.abc import Iterable
from typing import Dict, Any, List, Tuple
from collections import Counter
import json
import sys

from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.make_list import make_list
from utils.synthesize_from_distribution import synthesize_from_distribution
from utils.convert_to_binary import convert_to_binary
from utils.messageSliceBasedOnChunkSize import messageSliceBasedOnChunkSize
from utils.reconstruct_numbers_from_chunks import reconstruct_numbers_from_chunks


# from implementations.ParityOverwriteByTopWeightsEncode import ParityOverwriteByTopWeightsEncode
from implementations.OptimizedParityFittingWeightsEncodeAndDecode import OptimizedParityFittingWeightsEncodeAndDecode

NandTvaluesToKvalues = {
    63 : {
        1 : 57, 2: 51, 3: 45, 4: 39, 5: 36, 6: 30,  7: 24,  8: 18
    },
    127 : {
        4 : 99, 5: 92, 6: 85, 7: 78, 8: 71, 9: 71, 10: 64, 11: 57, 12: 50
    },
    255 : {
        8 : 191, 9: 187, 10: 179, 11: 171, 12: 163, 13: 155, 14: 147, 15: 139, 16: 131
    }
}

# ---------------------------
# Tunables
# ---------------------------
MEMMAP_PATH = "/home/vkamineni/Projects/RECC/pipeline_data/all_weights.int8.npy"
OUT_DIR     = "/home/vkamineni/Projects/RECC/pipeline_data/chunk_outputs"      # per-process JSONL files here
P = 24                   # processes
message_parity_size = CHUNK_SIZE = 63
message_size = 45

def _assign_affinity_for_worker(p: int, cpus_per_worker: int = 3):
    # Get the cpuset Slurm gave this job step (48 logical CPUs here)
    avail = sorted(os.sched_getaffinity(0))          # e.g., [0,1,2,...,47]
    # Partition into groups of 4 and pick the p-th group
    start = p * cpus_per_worker
    end = start + cpus_per_worker
    if end > len(avail):
        # fallback: modulo wrap if P*4 > available (shouldn't happen if SLURM matches)
        group = [avail[i % len(avail)] for i in range(start, end)]
    else:
        group = avail[start:end]
    os.sched_setaffinity(0, set(group))
    return group

# ---------------------------
# Atomic allocator
# ---------------------------
def claim_next(next_idx: mp.Value, lock: mp.Lock, N: int, chunk_size: int):
    with lock:
        start = next_idx.value
        if start >= N:
            return None
        next_idx.value = start + chunk_size
    end = min(start + chunk_size, N) - 1
    return (start, end)

# ---------------------------
# Your compute kernel (edit me)
# ---------------------------
def process_payload(values):
    values = values.tolist()
    message_bits = convert_to_binary(values, bit_size=8)
    chunks = messageSliceBasedOnChunkSize(message_bits, chunk_size=message_parity_size)
    mutated_chunks2 = []
    count=0
    for chunk in chunks:
        mutated_chunk2 = OptimizedParityFittingWeightsEncodeAndDecode(
                    chunk,
                    message_parity_size=message_parity_size,
                    message_size=message_size,
                    # warm_start=mutated_chunk["sliced_message_bits"],
                    solver='cpsat'
                )
        # mutated_chunk2 = chunk
        mutated_chunks2.append(mutated_chunk2)

        reconstructed_chunks2 = reconstruct_numbers_from_chunks(mutated_chunks2)
        mutated_nums2 = [reconstructed_chunks2[i]['original_number'] for i in range(len(reconstructed_chunks2))]
        
        mutated_nums_neg = (np.array(mutated_nums2) - 128).tolist()

        distorsion2 = sum([abs(mutated_nums2[i]-values[i]) for i in range(len(mutated_nums2))])/len(values)
        
    return mutated_nums_neg, distorsion2

# ---------------------------
# Worker
# ---------------------------
def worker(p: int, next_idx: mp.Value, lock: mp.Lock, N: int, chunk_size: int, batch_claim: int = 1):
    grp = _assign_affinity_for_worker(p, cpus_per_worker=3)
    if p == 0: print(f"[affinity] worker {p} -> CPUs {grp}")

    arr = np.load(MEMMAP_PATH, mmap_mode="r")   # dtype=int8, shape=(N,), no RAM copy

    # per-process sink
    pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    out_path = pathlib.Path(OUT_DIR) / f"chunks_p{p}.jsonl"

    # line-buffered writes; each record flushed on '\n'
    with open(out_path, "w", buffering=1) as f:
        while True:    
            rng = claim_next(next_idx, lock, N, chunk_size)
            if rng is None:
                break
            start, end = rng

            # slice values directly from memmap (zero-copy view)
            values = arr[start:end+1]          # memmap view
            mutated_values, distorsion = process_payload(values)
            # serialize to disk for later processing
            rec = {
                "p": p,
                "start": int(start),
                "end": int(end),
                "count": int(end - start + 1),
                "values": mutated_values,      # write actual int8s; change if you prefer summaries
                "distorsion":distorsion,
                "status": "ok"
            }
            if(next_idx.value%(63*1000)==0):
                print("Progress",time.time(),rec)

            f.write(json.dumps(rec) + "\n")
            

# ---------------------------
# Validator
# ---------------------------
def validate_coverage(N: int, log_dir: str):
    intervals = []
    if not os.path.exists(log_dir):
        raise RuntimeError(f"Log directory '{log_dir}' not found.")
    for name in os.listdir(log_dir):
        if not name.endswith(".jsonl"): 
            continue
        with open(os.path.join(log_dir, name), "r") as fh:
            for line in fh:
                if not line.strip(): 
                    continue
                rec = json.loads(line)
                if rec.get("status") != "ok":
                    continue
                intervals.append((rec["start"], rec["end"]))

    if not intervals and N > 0:
        raise RuntimeError("No completed intervals found.")

    intervals = list(set(intervals))
    intervals.sort()
    cur = 0
    # print(intervals)
    for (s, e) in intervals:
        if s != cur:
            raise RuntimeError(f"Gap or overlap detected at index {cur}; next interval starts at {s}.")
        cur = e + 1
    if cur != N:
        raise RuntimeError(f"Did not reach N={N}. Last covered index is {cur-1}.")
    print(f"[OK] Coverage validated: exactly [0, {N}) with {len(intervals)} chunks.")

# ---------------------------
# Entry point for standalone scripts
# ---------------------------
if __name__ == "__main__":
    # Read length from header once to set N robustly
    header_view = np.load(MEMMAP_PATH, mmap_mode="r")
    N = int(header_view.shape[0])
    
    # os.makedirs(OUT_DIR, exist_ok=True)
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    for f in Path(OUT_DIR).glob("chunks_p*.jsonl"):
        try:
            f.unlink()   # remove old per-process outputs
        except FileNotFoundError:
            pass

    ctx = mp.get_context("spawn")   # works both in notebook & script
    next_idx = ctx.Value('q', 0)
    lock = ctx.Lock()

    print(f"MEMMAP_PATH={MEMMAP_PATH}")
    print(f"N={N:,}  processes={P}  chunk_size={CHUNK_SIZE}")
    
    procs = []
    for p in range(P):
        proc = ctx.Process(target=worker, args=(p, next_idx, lock, N, CHUNK_SIZE, 1))
        proc.start()
        procs.append(proc)
    for proc in procs:
        proc.join()

    validate_coverage(N, OUT_DIR)