
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

def split_into_job_sizes(layers: list[int], jobs: int, block: int) -> list[int]:
    """
    Give each job an integer number of `block`-sized chunks as evenly as possible,
    and put the <block leftover on the last job.
    """
    total = sum(layers)
    total_blocks = total // block
    tail = total % block
    base = total_blocks // jobs
    extra = total_blocks % jobs
    sizes = [(base + (1 if i < extra else 0)) * block for i in range(jobs)]
    if tail:
        sizes[-1] += tail
    return sizes


def allocate_jobs(layers: list[int], sizes: list[int], layerNames, quant_info_key_to_index):
    """
    Map each job size onto consecutive layers.
    Records both layer-relative and global flat half-open ranges.
    """
    total = sum(layers)
    assert sum(sizes) == total, f"sum(sizes)={sum(sizes)} != total={total}"

    job_map = []
    layer_i, off_in_layer, flat = 0, 0, 0

    for j, need in enumerate(sizes):
        if need == 0:
            continue

        start_layer, start_off, flat_start = layer_i, off_in_layer, flat
        left = need

        while left > 0:
            avail = layers[layer_i] - off_in_layer
            take = min(avail, left)
            off_in_layer += take
            flat += take
            left -= take

            # move to next layer if current is done
            if off_in_layer == layers[layer_i] and layer_i<len(layers)-1:
                layer_i += 1
                off_in_layer = 0

        # start_layer_name = layerNames[start_layer]
        # end_layer_name = layerNames[layer_i]

        # start_layer_index = quant_info_key_to_index[start_layer_name]
        # end_layer_index = quant_info_key_to_index[end_layer_name]

        job_map.append({
            "job_index": j,
            "size": need,
            # layer-relative (end is exclusive)
            "start_layer": start_layer,
            # "start_layer": start_layer_index,
            "start_offset_in_layer": start_off,
            "end_layer": layer_i,
            # "end_layer": end_layer_index,
            "end_offset_in_layer": off_in_layer,
            # global flat (half-open)
            "flat_start": flat_start,
            "flat_end": flat_start + need,
        })

    assert flat == total, "Not all work assigned"
    return job_map

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
src_filename = f"model_int{quant}_ptq.pth"
src = f"/home/vkamineni/Projects/RECC/code/weights/"+src_filename
quant_info, num_bits, scales = load_quant_info_from_checkpoint(src)
# editable = clone_quant_info(quant_info)

ordered_keys = [k for k in quant_info.keys() if not should_skip(k, SKIP_TOKENS)]
quant_info_key_to_index = {k:i for i,k in enumerate(quant_info.keys())}

filtered_quant_info = {k: quant_info[k] for k in ordered_keys}

quant_info_records = change_quant_info_format(filtered_quant_info)

# --- build sizes and allocate ---
jobs = 12
single_process = 63
layers = [quant_info[k][0].numel() for k in ordered_keys]  # lengths per layer

sizes = split_into_job_sizes(layers, jobs, single_process)
job_map = allocate_jobs(layers, sizes, ordered_keys, quant_info_key_to_index)
skip_layers = {quant_info_key_to_index[k]:k for k in quant_info.keys() if should_skip(k, SKIP_TOKENS)}

payload = {
    "quant_info_records": quant_info_records,     # records is a list: [{}, {}...]
    "job_map": job_map, # dict: {job_index: {...}}
    "skip_layers": skip_layers
}

checkMaxMinValues(payload)

out_path = '/home/vkamineni/Projects/RECC/pipeline_data/quant_and_job_info.json'
with open(out_path, "w", encoding="utf-8") as f:
    # separators to reduce size; ensure ascii-safe
    json.dump(payload, f, separators=(",", ":"))

print(f"Saved: {out_path}")
print()
for j in job_map:
    sL, sO = j["start_layer"], j["start_offset_in_layer"]
    eL, eO = j["end_layer"],   j["end_offset_in_layer"]
    print(f"Job {j['job_index']:02d}  size={j['size']:5d} layers [{sL}:{sO}] -> [{eL}:{eO}]   ")