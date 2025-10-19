
import json, gzip, os
from typing import Dict, Tuple, List
import torch
import numpy as np
import sys

from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.weights_extractor_functions import cifar_test_loader, build_model, load_any_checkpoint, count_learnable_layers, param_counts, evaluate
from utils.load_quant_info_from_checkpoint import load_quant_info_from_checkpoint

def change_quant_info_format(quant_info: Dict[str, Tuple[torch.Tensor, float]]) -> str:
    records: List[dict] = []
    for layername, (t, scale) in quant_info.items():
        # ensure CPU, contiguous, int8
        t_cpu = t.detach().to("cpu").contiguous()
        if t_cpu.dtype != torch.int8:
            # if you truly need original dtype, replace with str(t_cpu.dtype) and adjust loader
            t_cpu = t_cpu.to(torch.int8)
        flat = t_cpu.view(-1)
        rec = {
            "layername": layername,
            "tensor_flattened": flat.tolist(),      # 1-D list of ints
            "shape": list(t.shape),                 # original shape
            "scale": scale,                  # python float
        }
        records.append(rec)
    return records

def checkMaxMinValues(payload):
    max_val = -9999
    min_val = 9999
    for layer in range(len(payload)):
        lst = payload['quant_info_records'][layer]['tensor_flattened']
        if(max_val<max(lst)):
            max_val = max(lst)
        if(min_val>min(lst)):
            min_val = min(lst)
    
    assert -128<=min_val and max_val<=127

def should_skip(layername: str, tokens) -> bool:
    return any(tok in layername for tok in tokens)

SKIP_TOKENS = [
    'running_var', 
    # 'conv'
    # add your own:
    # 'num_batches_tracked', 'running_mean', ...
]

# --- loading data ---
quant = 8 # 8 16 32
src = f"/home/vkamineni/Projects/RECC/code/weights/model_int{quant}_ptq.pth"
quant_info, num_bits, scales = load_quant_info_from_checkpoint(src)
# editable = clone_quant_info(quant_info)

ordered_keys = [k for k in quant_info.keys() if not should_skip(k, SKIP_TOKENS)]
quant_info_key_to_index = {k:i for i,k in enumerate(quant_info.keys())}

filtered_quant_info = {k: quant_info[k] for k in ordered_keys}

quant_info_records = change_quant_info_format(filtered_quant_info)

# --- build sizes and allocate ---
jobs = 12
single_process = 63

AllWeights = []
for i in range(len(quant_info_records)):
    print(quant_info_records[i]['layername'], len(quant_info_records[i]['tensor_flattened']))
    AllWeights.extend(quant_info_records[i]['tensor_flattened'])
    print('len(AllWeights)',len(AllWeights))

npy_out_path  = '/home/vkamineni/Projects/RECC/pipeline_data/all_weights.int8.npy'
all_weights_np = np.asarray(AllWeights, dtype=np.int8)
all_weights_np_u8 = (all_weights_np.astype(np.int16) + 128).astype(np.uint8)
np.save(npy_out_path, all_weights_np_u8, allow_pickle=False)

payload = {
    "quant_info_records": quant_info_records,     # records is a list: [{}, {}...]
}

checkMaxMinValues(payload)

json_out_path  = '/home/vkamineni/Projects/RECC/pipeline_data/quant_info.json'
with open(json_out_path, "w", encoding="utf-8") as f:
    # separators to reduce size; ensure ascii-safe
    json.dump(payload, f, separators=(",", ":"))

print(f"Saved JSON: {json_out_path}")
print(f"Saved NPY (int): {npy_out_path}  size={all_weights_np.size}")