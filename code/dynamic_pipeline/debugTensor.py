import torch
import sys
import numpy as np
import time
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.weights_extractor_functions import cifar_test_loader, build_model, load_any_checkpoint, copy_quant_checkpoint, count_learnable_layers, param_counts, evaluate, save_updated_quant_checkpoint
from utils.load_quant_info_from_checkpoint import load_quant_info_from_checkpoint, clone_quant_info, revert_back_some_layers

dataset_argument = "CIFAR10"
batch_size_argument = 128
device_argument = "cuda" if torch.cuda.is_available() else "cpu"
arch_argument = "resnet18"

# print('came here', device_argument)

start_time = time.time()

weights_argument_original = "/home/vkamineni/Projects/RECC/code/weights/model_int8_ptq.pth"

loader, nc = cifar_test_loader(dataset_argument, batch_size_argument)
device = torch.device(device_argument)
model = build_model(arch_argument, nc).to(device)

tag, quant_info = load_any_checkpoint(model, weights_argument_original, map_location=device, strict=True)
print(f"[Loaded] {weights_argument_original} → {tag}\n")

layers = count_learnable_layers(model)
total_p, train_p = param_counts(model)
print(f"\n[Layers] learnable modules: {layers}")
print(f"[Params] total: {total_p:,}  trainable: {train_p:,}")

acc = evaluate(model, loader, device)
print(f"\n[Test accuracy] {acc:.2f}% on {dataset_argument}")

print('duration 1', time.time()-start_time)

print()

for _ in range(10):

    start_time = time.time()

    weights_argument = "/home/vkamineni/Projects/RECC/pipeline_data/new_quant_info.pth"
    # weights_argument = "/home/vkamineni/Projects/RECC/code/weights/model_int8_ptq.pth"

    weights_argument_copy_dst = "/home/vkamineni/Projects/RECC/pipeline_data/new_quant_info_copy_dsp.pth"

    # 1) copy
    copy_quant_checkpoint(weights_argument, weights_argument_copy_dst)

    # 2) load and clone
    original, _, _ = load_quant_info_from_checkpoint(weights_argument_original)
    quant_info, num_bits, scales = load_quant_info_from_checkpoint(weights_argument_copy_dst)
    editable = clone_quant_info(quant_info)

    keys = list(editable.keys())

    # 3) make some dummy edits (example: only layers with 'conv' in name)
    # editable2, TotalCount,SkippedCount = apply_dummy_edits(editable, select_substr=None)
    # select_layers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 22]
    # select_layers = [i for i in range(122)]
    # select_layers = [4, 10, 16, 22, 28, 34, 40, 46, 52, 58, 64, 70, 76, 82, 88, 94, 100, 106, 112, 118]
    select_layers = []
    editable, TotalCount, SkippedCount = revert_back_some_layers(editable, original, select_layers=select_layers)

    print('SkippedCount',SkippedCount)
    print('TotalCount',TotalCount)
    print('SkippedCount/TotalCount',SkippedCount/TotalCount)

    # 4) save back into the copied checkpoint
    save_updated_quant_checkpoint(weights_argument_copy_dst, editable, tag_suffix="_edited")

    loader, nc = cifar_test_loader(dataset_argument, batch_size_argument)
    device = torch.device(device_argument)
    model = build_model(arch_argument, nc).to(device)

    tag, quant_info = load_any_checkpoint(model, weights_argument_copy_dst, map_location=device, strict=True)
    print(f"[Loaded] {weights_argument_copy_dst} → {tag}")

    layers = count_learnable_layers(model)
    total_p, train_p = param_counts(model)
    print(f"[Layers] learnable modules: {layers}")
    print(f"[Params] total: {total_p:,}  trainable: {train_p:,}")

    acc = evaluate(model, loader, device)

    print(f"[Test accuracy] {acc:.2f}% on {dataset_argument}")
    print()