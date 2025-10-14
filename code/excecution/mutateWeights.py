import random
import galois
import numpy as np
import pulp as pl
import time
import math
from collections.abc import Iterable
from typing import Dict, Any, List, Tuple
from collections import Counter
import numpy as np
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.weights_extractor_functions import cifar_test_loader, build_model, load_any_checkpoint, count_learnable_layers, param_counts, evaluate

# read the weights on each layer in the Neural Network seperatly -> original weights in each layer

# original weights in each layer -> Save the format shape of those weights -> flat list of original weights

# flat list of original weights -> input those weights into MIP solver -> flat list of mutated weights

# flat list of mutated weights -> convert to orginal format shape -> mutated weights in each layer

# mutated weights in each layer -> update and save those new weights in the neural network

dataset_argument = "CIFAR10"
batch_size_argument = 128
device_argument = "cuda" if torch.cuda.is_available() else "cpu"
arch_argument = "resnet18"

print('came here', device_argument)
quant = 8 # 8 16 32
bit_change = 3  # 1 2 3 4

src_filename = f"\model_int{quant}_ptq.pth"
dst_filename = f"\model_int{quant}_ptq_edit.pth"
src = "/home/vkamineni/Projects/RECC/code/weights"+src_filename
dst = "/home/vkamineni/Projects/RECC/code/weights"+dst_filename


# 1) copy
copy_quant_checkpoint(src, dst)

# 2) load and clone
quant_info, num_bits, scales = load_quant_info_from_checkpoint(dst)
editable = clone_quant_info(quant_info)
# print("num_bits:", num_bits)
# print("layers available:", list(editable.keys())[:5], "...", len(editable.keys()))

# 3) make some dummy edits (example: only layers with 'conv' in name)
editable2, TotalCount,SkippedCount = apply_dummy_edits(editable, bit_change=bit_change, select_substr=None)

# 4) save back into the copied checkpoint
save_updated_quant_checkpoint(dst, editable, tag_suffix="_edited")
# print("Saved edited quantized checkpoint to:", dst)

loader, nc = cifar_test_loader("CIFAR10", 128)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model("resnet18", nc).to(device)
tag, quant_info = load_any_checkpoint(model, dst, map_location=device, strict=True)