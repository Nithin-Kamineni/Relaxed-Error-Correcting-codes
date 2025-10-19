import json
from math import prod
from pathlib import Path
import torch
import numpy as np
import os

from dynamic_parallel_payload_process import process_payload, validate_coverage

# -------- paths (edit if needed) --------
MEMMAP_PATH = "/home/vkamineni/Projects/RECC/pipeline_data/all_weights.int8.npy"
OUT_DIR = Path("/home/vkamineni/Projects/RECC/pipeline_data/chunk_outputs")
ORIG_CHECKPOINT = "/home/vkamineni/Projects/RECC/code/weights/model_int8_ptq.pth"
OUT_CKPT = "/home/vkamineni/Projects/RECC/pipeline_data/new_quant_info.pth"

def add_chunks(missed, start, end, chunk=63):
    """Append inclusive [start, end] as chunk-sized intervals (last one may be shorter)."""
    x = start
    while x <= end:
        y = min(x + chunk - 1, end)
        missed.append((x, y))
        x = y + 1

arr = np.load(MEMMAP_PATH, mmap_mode="r")   # dtype=int8, shape=(N,), no RAM copy
N = len(arr)

# -------- read chunk outputs and reassemble in order --------
intervals = []
if not os.path.exists(OUT_DIR):
    raise RuntimeError(f"Log directory '{OUT_DIR}' not found.")
for name in os.listdir(OUT_DIR):
    if not name.endswith(".jsonl"): 
        continue
    with open(os.path.join(OUT_DIR, name), "r") as fh:
        for line in fh:
            if not line.strip(): 
                continue
            rec = json.loads(line)
            if rec.get("status") != "ok":
                continue
            intervals.append((rec["start"], rec["end"]))


# sort by start; ensure full coverage with no gaps/overlaps
intervals.sort(key=lambda t: t[0])

missed_intervals = []
cur = 0
for (s, e) in intervals:
    if s != cur:
        add_chunks(missed_intervals, cur, s - 1)
    cur = e + 1
if cur != N:
    add_chunks(missed_intervals, cur, N - 1)

print("len(missed_intervals)", len(missed_intervals))
if not missed_intervals:
    print(f"[OK] Coverage validated: exactly [0, {N}) with {len(intervals)} chunks. (no need to add)")

else:
    out_path = Path(OUT_DIR) / f"chunks_p30.jsonl"

    # open file in append mode
    with open(out_path, "a", buffering=1) as f:
        for missed_interval in missed_intervals:
            start, end = missed_interval
            values = np.array(arr[start:end+1], copy=True)  # make it writable
            mutated_values, distorsion = process_payload(values)
            rec = {
                    "p": 'NotInJobs',
                    "start": int(start),
                    "end": int(end),
                    "count": int(end - start + 1),
                    "values": mutated_values,      # write actual int8s; change if you prefer summaries
                    "distorsion":distorsion,
                    "status": "ok"
                    }
            
            f.write(json.dumps(rec) + "\n")

    validate_coverage(N, OUT_DIR)