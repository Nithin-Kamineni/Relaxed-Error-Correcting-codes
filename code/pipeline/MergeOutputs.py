import json
from math import prod
from pathlib import Path
import torch
import numpy as np

def load_original_quant(path: str):
    payload = torch.load(path, map_location="cpu")
    return payload

path = "/home/vkamineni/Projects/RECC/pipeline_data/quant_and_job_info.json"
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Access fields
quant_info_records = data["quant_info_records"]

# read list of numbers from different files and combine them in a single list
nums = []
dir_path = Path("/home/vkamineni/Projects/RECC/pipeline_data/ResultingWeightsProcessed")
print(dir_path.glob("*.json"))
print(sorted(dir_path.glob("*.json"), key=lambda x:int(x.stem)))
for fp in sorted(dir_path.glob("*.json"), key=lambda x:int(x.stem)):  # sorted for deterministic order
    with open(fp, "r", encoding="utf-8") as f:
        data = json.load(f)
        nums.extend(data)

weights_argument_original = "/home/vkamineni/Projects/RECC/code/weights/model_int8_ptq.pth"
# new_quant_info = {'qstate_dict':{}, 'meta':{'num_bits':8, 'scales':{}}}
new_quant_info = load_original_quant(weights_argument_original)

print('len(nums)',len(nums))
complete_index=0
for i in range(len(quant_info_records)):
    num_ele_in_layer = prod(quant_info_records[i]['shape'])
    weights_in_layer = nums[complete_index:num_ele_in_layer+complete_index]
    weights_in_layer_np = np.array(weights_in_layer)
    
    print(num_ele_in_layer, len(weights_in_layer), weights_in_layer_np.shape, quant_info_records[i]['shape'], "aa")

    weights_in_layer_np = weights_in_layer_np.reshape(quant_info_records[i]['shape'])
    weights_in_layer_np = np.asarray(weights_in_layer_np, dtype=np.int8, order="C")
    t = torch.from_numpy(weights_in_layer_np)        # dtype=torch.int8, shares memory

    scale = quant_info_records[i]['scale']
    layername = quant_info_records[i]['layername']

    new_quant_info['meta']['scales'][layername] = scale
    new_quant_info['qstate_dict'][layername] = t

    complete_index+=num_ele_in_layer

# --- save for reuse elsewhere ---
out_ckpt = "/home/vkamineni/Projects/RECC/pipeline_data/new_quant_info.pth"
torch.save(new_quant_info, out_ckpt)
print(f"Saved: {out_ckpt}")