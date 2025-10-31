import torch
import sys
import numpy as np
import time
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.weights_extractor_functions import cifar_test_loader, build_model, load_any_checkpoint, copy_quant_checkpoint, count_learnable_layers, param_counts, evaluate, save_updated_quant_checkpoint

import os, certifi
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("CURL_CA_BUNDLE", certifi.where())


import ssl
ssl._create_default_https_context = ssl._create_stdlib_context


dataset_argument = "CIFAR10"
batch_size_argument = 128
device_argument = "cuda" if torch.cuda.is_available() else "cpu"
arch_argument = "resnet18"

# print('came here', device_argument)

start_time = time.time()

# weights_argument = "/home/vkamineni/Projects/RECC/code/weights/model_int8_ptq.pth"
ARTIFACT_PATH = os.getenv("ARTIFACT_PATH", "cifar10/resnet18/model_int8_ptq.pth")
weights_argument = f"/home/vkamineni/Projects/RECC/code/trainAndQuantize/shell-scripts/artifacts/models/{ARTIFACT_PATH}"

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

print('duration 1', time.time()-start_time)

start_time = time.time()

# weights_argument = "/home/vkamineni/Projects/RECC/pipeline_data/resnet18_parityM6T1_All_MIP.pth"

# weights_argument = "/home/vkamineni/Projects/RECC/pipeline_data/resnet18_parityM127T1_All_MIP.pth"
# weights_argument = "/home/vkamineni/Projects/RECC/pipeline_data/resnet18_replaceM127T3.pth"

# weights_argument = "/home/vkamineni/Projects/RECC/pipeline_data/resnet18_parityM6T6_All_MIP.pth"
EvalModel = os.getenv("EvalModel", "cifar10/resnet18/M63_t2/no/model_int8_ptq.pth")
weights_argument = "/home/vkamineni/Projects/RECC/pipeline_data/processed_payload/"+EvalModel
# weights_argument = "/home/vkamineni/Projects/RECC/code/weights/model_int8_ptq.pth"

loader, nc = cifar_test_loader(dataset_argument, batch_size_argument)
device = torch.device(device_argument)
model = build_model(arch_argument, nc).to(device)

tag, quant_info = load_any_checkpoint(model, weights_argument, map_location=device, strict=True)
print(f"[Loaded] {weights_argument} → {tag}\n")

layers = count_learnable_layers(model)
acc_lst = []

total_p, train_p = param_counts(model)
print(f"\n[Layers] learnable modules: {layers}")
print(f"[Params] total: {total_p:,}  trainable: {train_p:,}")

acc = evaluate(model, loader, device)
acc_lst.append(acc)
print(f"\n[Test accuracy] {acc:.2f}% on {dataset_argument}")

print('avg(acc)',sum(acc_lst)/len(acc_lst))
print('duration 2', time.time()-start_time)