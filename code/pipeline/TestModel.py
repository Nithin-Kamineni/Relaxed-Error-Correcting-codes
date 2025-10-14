import torch
import sys
import numpy as np
import time
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.weights_extractor_functions import cifar_test_loader, build_model, load_any_checkpoint, copy_quant_checkpoint, count_learnable_layers, param_counts, evaluate, save_updated_quant_checkpoint


dataset_argument = "CIFAR10"
batch_size_argument = 128
device_argument = "cuda" if torch.cuda.is_available() else "cpu"
arch_argument = "resnet18"

# print('came here', device_argument)

start_time = time.time()

weights_argument = "/home/vkamineni/Projects/RECC/code/weights/model_int8_ptq.pth"

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

weights_argument = "/home/vkamineni/Projects/RECC/pipeline_data/new_quant_info.pth"
# weights_argument = "/home/vkamineni/Projects/RECC/code/weights/model_int8_ptq.pth"

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

print('duration 2', time.time()-start_time)