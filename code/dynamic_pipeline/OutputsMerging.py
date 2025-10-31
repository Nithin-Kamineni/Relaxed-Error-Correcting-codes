import json
from math import prod
from pathlib import Path
import torch
import numpy as np
import os

ARTIFACT_PATH = os.getenv("ARTIFACT_PATH", "cifar10/resnet18/model_int8_ptq.pth")
art_path = Path(ARTIFACT_PATH)
dir_path = str(art_path.parent)
file_stem  = art_path.stem
base = Path("/home/vkamineni/Projects/RECC/pipeline_data/artifact_loaded_data")
target_dir = base / dir_path
QUANT_INFO_JSON = target_dir / f"{file_stem}.quant_info.json"

message_parity_size = CHUNK_SIZE = int(os.getenv("CODEWORD", 63))
t_value = int(os.getenv("Tvalue", 2))
OUT_DIR_BASE     = Path("/home/vkamineni/Projects/RECC/pipeline_data/chunk_outputs")      # per-process JSONL files here
MessageEncoding = f"M{message_parity_size}_t{t_value}"
Approch = os.getenv("Approch", "parfit")
Approch = 'no' if Approch not in ('parfit','replace','no') else Approch
OUT_DIR     = OUT_DIR_BASE/dir_path/MessageEncoding/Approch

ORIG_CHECKPOINT = f"/home/vkamineni/Projects/RECC/code/trainAndQuantize/shell-scripts/artifacts/models/{ARTIFACT_PATH}"

output_ckpt_base = Path("/home/vkamineni/Projects/RECC/pipeline_data/processed_payload")      # per-process JSONL files here
OUT_CKPT_directory = output_ckpt_base/dir_path/MessageEncoding/Approch
OUT_CKPT_directory.mkdir(parents=True, exist_ok=True)
OUT_CKPT = output_ckpt_base/dir_path/MessageEncoding/Approch/art_path.name


# -------- paths (edit if needed) --------
# OUT_DIR = Path("/home/vkamineni/Projects/RECC/pipeline_data/chunk_outputs")
# QUANT_INFO_JSON = "/home/vkamineni/Projects/RECC/pipeline_data/quant_info.json"
# ORIG_CHECKPOINT = "/home/vkamineni/Projects/RECC/code/weights/model_int8_ptq.pth"
# OUT_CKPT = "/home/vkamineni/Projects/RECC/pipeline_data/new_quant_info.pth"

# -------- load quant_info metadata (shapes, scales, layer order) --------
with open(QUANT_INFO_JSON, "r", encoding="utf-8") as f:
    job = json.load(f)
quant_info_records = job["quant_info_records"]   # list of dicts with layername, shape, scale

# Expected total elements
expected_total = sum(prod(r["shape"]) for r in quant_info_records)

# -------- read chunk outputs and reassemble in order --------
records = []
for fp in OUT_DIR.glob("chunks_p*.jsonl"):
    with fp.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("status") != "ok":
                continue
            start = int(rec["start"])
            end = int(rec["end"])
            vals = rec["values"]
            # sanity checks per record
            cnt = int(rec["count"])
            if cnt != (end - start + 1):
                raise ValueError(f"Bad record count @ {fp}: start={start} end={end} count={cnt}")
            if cnt != len(vals):
                raise ValueError(f"Value length mismatch @ {fp}: count={cnt} len(values)={len(vals)}")
            records.append((start, end, vals))

if not records:
    raise RuntimeError(f"No valid 'ok' records found under {OUT_DIR}")

def load_original_quant(path: str):
    payload = torch.load(path, map_location="cpu")
    return payload

# sort by start; ensure full coverage with no gaps/overlaps
records.sort(key=lambda t: t[0])
cur = 0
stitched = []
for (s, e, vals) in records:
    if s != cur:
        raise RuntimeError(f"Gap/overlap detected at index {cur}; next interval starts at {s}")
    stitched.extend(vals)
    cur = e + 1

total = len(stitched)
if total != expected_total:
    raise RuntimeError(f"Reassembled length {total} != expected {expected_total}")

# Convert to int8 ndarray (your JSON values are already in -128..127)
all_weights_np = np.asarray(stitched, dtype=np.int8, order="C")

# -------- rebuild per-layer tensors in new_quant_info --------
# load original checkpoint to copy structure (meta etc.)
new_quant_info = torch.load(ORIG_CHECKPOINT, map_location="cpu")
# ensure required keys exist
new_quant_info.setdefault("qstate_dict", {})
new_quant_info.setdefault("meta", {})
new_quant_info["meta"].setdefault("scales", {})
new_quant_info["meta"]["num_bits"] = 8  # keep consistent

idx = 0
for rec in quant_info_records:
    layername = rec["layername"]
    shape = rec["shape"]
    scale = rec["scale"]

    num = prod(shape)
    layer_vals = all_weights_np[idx: idx + num]     # view slice
    if layer_vals.size != num:
        raise RuntimeError(f"Layer {layername}: need {num}, got {layer_vals.size}")

    # reshape and convert to torch.int8 tensor
    arr = np.array(layer_vals, dtype=np.int8, copy=False).reshape(shape, order="C")
    t = torch.from_numpy(arr)  # shares memory; dtype int8

    new_quant_info["qstate_dict"][layername] = t
    new_quant_info["meta"]["scales"][layername] = scale

    idx += num

assert idx == total, f"Stitching mismatch: consumed {idx}, total {total}"

# --- save for reuse elsewhere ---
torch.save(new_quant_info, OUT_CKPT)
print(f"Saved: {OUT_CKPT}")