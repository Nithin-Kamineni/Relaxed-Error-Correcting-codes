import os, json, math, time
from pathlib import Path
from typing import Dict, Tuple

import torch
import numpy as np

# -----------------------------
# Config paths
# -----------------------------
SRC = "/home/vkamineni/Projects/RECC/code/weights/model_int8_ptq.pth"
NEW = "/home/vkamineni/Projects/RECC/pipeline_data/new_quant_info.pth"
DST = "/home/vkamineni/Projects/RECC/pipeline_data/model_int8_ptq_edit.pth"

# -----------------------------
# Utils
# -----------------------------
def to_np_int8(t):
    if isinstance(t, torch.Tensor):
        return t.detach().to("cpu").contiguous().view(-1).to(torch.int8).numpy()
    elif isinstance(t, np.ndarray):
        return t.astype(np.int8, copy=False).reshape(-1)
    else:
        return np.asarray(t, dtype=np.int8).reshape(-1)

def load_quant_info_dict_from_ckpt(path: str) -> Dict[str, Tuple[torch.Tensor, float]]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if "quant_info" in ck:
        return ck["quant_info"]
    # If your src file stores it under another key, add a fallback here:
    # elif "checkpoint" in ck and "quant_info" in ck["checkpoint"]: return ck["checkpoint"]["quant_info"]
    raise KeyError(f"'quant_info' not found in {path}")

def compare_quant_infos(q1: Dict[str, Tuple[torch.Tensor, float]],
                        q2: Dict[str, Tuple[torch.Tensor, float]],
                        atol_scale: float = 1e-8) -> None:
    keys1, keys2 = set(q1.keys()), set(q2.keys())
    only1 = sorted(keys1 - keys2)
    only2 = sorted(keys2 - keys1)
    if only1:
        print(f"[WARN] Layers only in SRC : {len(only1)} -> {only1[:10]}{' ...' if len(only1)>10 else ''}")
    if only2:
        print(f"[WARN] Layers only in NEW : {len(only2)} -> {only2[:10]}{' ...' if len(only2)>10 else ''}")

    common = sorted(keys1 & keys2)
    total_elems = 0
    total_mismatch = 0
    per_layer_report = []

    for k in common:
        t1, s1 = q1[k]
        t2, s2 = q2[k]

        a1 = to_np_int8(t1)
        a2 = to_np_int8(t2)

        if a1.shape != a2.shape:
            per_layer_report.append((k, "SHAPE_MISMATCH", a1.shape, a2.shape, None, None))
            continue

        # compare scales
        scale_close = (abs(float(s1) - float(s2)) <= atol_scale)

        # compare values
        diff = (a1 != a2)
        mism = int(diff.sum())
        n = a1.size

        total_elems += n
        total_mismatch += mism

        status = "OK" if (mism == 0 and scale_close) else ("VAL_MISMATCH" if mism > 0 else "SCALE_MISMATCH")
        # find a few examples if mismatched
        examples = None
        if mism > 0:
            idxs = np.flatnonzero(diff)[:5]
            examples = [(int(i), int(a1[i]), int(a2[i])) for i in idxs]

        per_layer_report.append((k, status, n, mism, bool(scale_close), examples))

    # Print concise report
    print("\n=== Layer-by-layer comparison (common layers) ===")
    for k, status, n_or_shape, mism, scale_ok, examples in per_layer_report:
        if status == "SHAPE_MISMATCH":
            print(f"[{status:15}] {k}: a.shape={n_or_shape} vs b.shape={mism}")
        else:
            line = f"[{status:15}] {k}: elems={n_or_shape}, mismatches={mism}, scale_equal={scale_ok}"
            if examples:
                line += f", ex={examples}"
            print(line)

    print("\n=== Summary ===")
    print(f"Common layers      : {len(common)}")
    print(f"Only in SRC        : {len(only1)}")
    print(f"Only in NEW        : {len(only2)}")
    print(f"Total elems compared: {total_elems}")
    print(f"Total mismatches    : {total_mismatch}")
    if total_elems > 0:
        print(f"Mismatch rate       : {total_mismatch/total_elems:.6f}")

# -----------------------------
# 1) Load both quant_info dicts and compare
# -----------------------------
src_q = load_quant_info_dict_from_ckpt(SRC)
new_q = load_quant_info_dict_from_ckpt(NEW)

print(">>> Comparing SRC quant_info vs NEW quant_info (per layer values & scales)")
compare_quant_infos(src_q, new_q)

# -----------------------------
# 2) Accuracy comparison (last)
#     - Evaluate original SRC checkpoint
#     - Create DST by copying SRC, then inject NEW quant_info and evaluate
# -----------------------------
from utils.weights_extractor_functions import (
    cifar_test_loader, build_model, load_any_checkpoint,
    copy_quant_checkpoint, evaluate, save_updated_quant_checkpoint
)

loader, nc = cifar_test_loader("CIFAR10", 128)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# A) Acc of SRC as-is
model_src = build_model("resnet18", nc).to(device)
_ = load_any_checkpoint(model_src, SRC, map_location=device, strict=True)
model_src.eval()
with torch.inference_mode():
    acc_src = evaluate(model_src, loader, device)
print(f"\nACC (SRC): {acc_src:.4f}")

# B) Acc after injecting NEW into a copy of SRC
copy_quant_checkpoint(SRC, DST)
save_updated_quant_checkpoint(DST, new_q)

model_new = build_model("resnet18", nc).to(device)
_ = load_any_checkpoint(model_new, DST, map_location=device, strict=True)
model_new.eval()
with torch.inference_mode():
    acc_new = evaluate(model_new, loader, device)
print(f"ACC (NEW injected): {acc_new:.4f}")

print(f"\nΔACC (NEW - SRC): {acc_new - acc_src:+.4f}")
