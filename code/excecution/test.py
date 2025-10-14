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
weights_argument = "/home/vkamineni/Projects/RECC/code/weights/model_int8_ptq.pth"

print('came here', device_argument)

# import os, certifi
# os.environ["SSL_CERT_FILE"] = certifi.where()

start_time = time.time()

loader, nc = cifar_test_loader(dataset_argument, batch_size_argument)
device = torch.device(device_argument)
model = build_model(arch_argument, nc).to(device)

tag, quant_info = load_any_checkpoint(model, weights_argument, map_location=device, strict=True)
print(f"[Loaded] {weights_argument} → {tag}\n")

layers = count_learnable_layers(model)
total_p, train_p = param_counts(model)
print(f"\n[Layers] learnable modules: {layers}")
print(f"[Params] total: {total_p:,}  trainable: {train_p:,}")

acc = evaluate(model, loader, device)
print(f"\n[Test accuracy] {acc:.2f}% on {dataset_argument}")

print('duration', time.time()-start_time)



