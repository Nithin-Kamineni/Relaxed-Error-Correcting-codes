from __future__ import annotations
import os, math, time, argparse
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from torchvision import datasets, transforms, models
import sys

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# ------------------------- Utils -------------------------

def log(rank: int, *msg: Any):
    if rank == 0:
        print(*msg, flush=True)


def pick_device(arg_device: str, local_rank: int) -> torch.device:
    if arg_device.lower().startswith("cuda") and torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def maybe_compile(model: nn.Module, use_compile: bool) -> nn.Module:
    if use_compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, fullgraph=False, mode="max-autotune")
        except Exception:
            pass
    return model


def strip_prefix_from_state_dict(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in sd.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        if nk.startswith("_orig_mod."):
            nk = nk[len("_orig_mod."):]
        out[nk] = v
    return out


# ------------------------- Data -------------------------

def cifar_mean_std(dataset: str):
    if dataset.lower() == "cifar10":
        return [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]
    else:  # CIFAR100
        return [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]


def build_transforms(dataset: str, train: bool):
    ds = dataset.lower()
    if ds in ("cifar10", "cifar100"):
        mean, std = cifar_mean_std(ds)
        if train:
            return transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
    elif ds == "mnist":
        mean, std = [0.1307]*3, [0.3081]*3
        base = [
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.expand(3, -1, -1)),
            transforms.Normalize(mean, std),
        ]
        return transforms.Compose(base)
    else:
        raise ValueError(f"Unknown dataset {dataset}")


def get_datasets(dataset: str, data_root: str):
    ds = dataset.lower()
    tf_tr = build_transforms(ds, train=True)
    tf_te = build_transforms(ds, train=False)
    if ds == "cifar10":
        tr = datasets.CIFAR10(data_root, train=True, transform=tf_tr, download=True)
        te = datasets.CIFAR10(data_root, train=False, transform=tf_te, download=True)
        nc = 10
    elif ds == "cifar100":
        tr = datasets.CIFAR100(data_root, train=True, transform=tf_tr, download=True)
        te = datasets.CIFAR100(data_root, train=False, transform=tf_te, download=True)
        nc = 100
    elif ds == "mnist":
        tr = datasets.MNIST(data_root, train=True, transform=tf_tr, download=True)
        te = datasets.MNIST(data_root, train=False, transform=tf_te, download=True)
        nc = 10
    else:
        raise ValueError(dataset)
    return tr, te, nc


def get_dataloaders(dataset: str, data_root: str, batch_size: int, num_workers: int, dist_mode: bool,
                     seed: int = 42):
    tr_set, te_set, nc = get_datasets(dataset, data_root)
    sampler = DistributedSampler(tr_set, shuffle=True, seed=seed) if dist_mode else None
    tr_loader = DataLoader(tr_set, batch_size=batch_size, shuffle=(sampler is None),
                           sampler=sampler, num_workers=num_workers, pin_memory=True, drop_last=True)
    te_loader = DataLoader(te_set, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True)
    return tr_loader, te_loader, nc, sampler


# ------------------------- Models -------------------------

class CIFARMLP(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3*32*32, 1024), nn.ReLU(True),
            nn.Linear(1024, 512), nn.ReLU(True),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        return self.net(x)


def adapt_resnet(m: nn.Module, num_classes: int):
    
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def adapt_vgg16(m: nn.Module, num_classes: int):
    
    in_features = 512 * 7 * 7
    m.classifier = nn.Sequential(
        nn.Linear(in_features, 512), nn.ReLU(True), nn.Dropout(0.5),
        nn.Linear(512, 512), nn.ReLU(True), nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )
    return m


def adapt_alexnet(m: nn.Module, num_classes: int):
    
    m.features[0] = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
    m.classifier = nn.Sequential(nn.Linear(256*6*6 if hasattr(m, 'avgpool') else 512, num_classes))
    return m


def build_model(arch: str, num_classes: int) -> nn.Module:
    a = arch.lower()
    if a in ("resnet18", "resnet34", "resnet50", "resnet101"):
        m = getattr(models, a)(weights=None)
        return adapt_resnet(m, num_classes)
    elif a == "vgg16":
        m = models.vgg16(weights=None)
        return adapt_vgg16(m, num_classes)
    elif a == "alexnet":
        m = models.alexnet(weights=None)
        return adapt_alexnet(m, num_classes)
    elif a == "mlp":
        return CIFARMLP(num_classes)
    else:
        raise ValueError(arch)




@dataclass
class TrainCfg:
    epochs: int = 120
    lr: float = 0.1
    weight_decay: float = 5e-4
    optim: str = "sgd"
    momentum: float = 0.9
    scheduler: str = "cosine"  # or step
    step_size: int = 60
    gamma: float = 0.2
    label_smoothing: float = 0.0
    warmup_epochs: int = 0


def build_optimizer(model: nn.Module, cfg: TrainCfg):
    if cfg.optim.lower() == "sgd":
        return optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay, nesterov=True)
    elif cfg.optim.lower() == "adamw":
        return optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        raise ValueError(cfg.optim)


def build_scheduler(optimizer, cfg: TrainCfg, steps_per_epoch: int):
    if cfg.scheduler == "cosine":
        def lr_lambda(epoch):
            if cfg.warmup_epochs > 0 and epoch < cfg.warmup_epochs:
                return float(epoch + 1) / float(cfg.warmup_epochs)
            t = (epoch - cfg.warmup_epochs) / max(1, (cfg.epochs - cfg.warmup_epochs))
            return 0.5 * (1.0 + math.cos(math.pi * t))
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif cfg.scheduler == "step":
        return optim.lr_scheduler.StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)
    else:
        return None


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total


# ------------------------- PTQ (INT4/8/16) -------------------------

# Simple per-tensor symmetric quantization into integer tensors + global scale per tensor

def quantize_tensor(t: torch.Tensor, num_bits: int):
    assert t.dtype in (torch.float32, torch.float16, torch.bfloat16)
    t = t.detach().to(torch.float32)
    qmin = -(2**(num_bits-1))
    qmax = (2**(num_bits-1)) - 1
    max_abs = t.abs().max().item() + 1e-12
    scale = max_abs / qmax
    q = torch.clamp(torch.round(t / scale), qmin, qmax)
    if num_bits <= 8:
        q = q.to(torch.int8)
    else:
        q = q.to(torch.int16)
    return q, scale


def dequantize_tensor(q: torch.Tensor, scale: float):
    return q.to(torch.float32) * float(scale)


def save_quantized_checkpoint(model: nn.Module, path: str, num_bits: int):
    qsd: Dict[str, torch.Tensor] = {}
    scales: Dict[str, float] = {}
    for k, p in model.state_dict().items():
        if p.dtype.is_floating_point:
            q, s = quantize_tensor(p, num_bits)
            qsd[k] = q.cpu()
            scales[k] = float(s)
        else:
            # keep non-float tensors as-is
            qsd[k] = p.cpu()
            scales[k] = None
    payload = {
        "qstate_dict": qsd,
        "meta": {"num_bits": num_bits, "scales": scales}
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)


def load_quantized_into_model(model: nn.Module, dataset: str, arch: str, num_bits: int, qtag: str,
                              map_location: torch.device):
    ck = f"artifacts/models/{dataset.lower()}/{arch.lower()}/model_int{num_bits}_{qtag}.pth"
    payload = torch.load(ck, map_location=map_location)
    qsd = payload["qstate_dict"]
    scales = payload["meta"]["scales"]
    dsd = {}
    for k, v in qsd.items():
        s = scales.get(k, None)
        if s is None:
            dsd[k] = v
        else:
            dsd[k] = dequantize_tensor(v, s)
    dsd = strip_prefix_from_state_dict(dsd)
    model.load_state_dict(dsd, strict=True)
    return payload["meta"]


# ------------------------- Checkpoint I/O -------------------------

def ckpt_path(dataset: str, arch: str, tag: str) -> str:
    return f"artifacts/models/{dataset.lower()}/{arch.lower()}/model_{tag}.pth"


def save_float_checkpoint(model: nn.Module, dataset: str, arch: str, tag: str, extra: Dict[str, Any] | None = None):
    path = ckpt_path(dataset, arch, tag)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"state_dict": model.state_dict()}
    if extra: payload.update(extra)
    torch.save(payload, path)
    return path


def load_float_into_model(model: nn.Module, dataset: str, arch: str, tag: str, map_location: torch.device):
    path = ckpt_path(dataset, arch, tag)
    payload = torch.load(path, map_location=map_location)
    sd = payload.get("state_dict", payload)
    sd = strip_prefix_from_state_dict(sd)
    model.load_state_dict(sd, strict=True)


# ------------------------- Train / Eval / Quantize / Inspect -------------------------

def cmd_train(args):
    use_ddp = args.dist and dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if use_ddp else 0
    local_rank = int(os.environ.get("LOCAL_RANK", 0)) if use_ddp else 0
    device = pick_device(args.device, local_rank)
    log(rank, f"[Device] {device} (DDP={use_ddp}, rank={rank}, local_rank={local_rank}, world={(dist.get_world_size() if use_ddp else 1)})")

    train_loader, test_loader, nc, train_sampler = get_dataloaders(
        args.dataset, args.data_root, args.batch_size, args.workers, use_ddp)

    model = build_model(args.arch, nc).to(device)
    model = maybe_compile(model, args.compile)

    if use_ddp:
        model = DDP(model, device_ids=[device.index] if device.type == 'cuda' else None)

    cfg = TrainCfg(
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        optim=args.optim,
        momentum=args.momentum,
        scheduler=args.scheduler,
        step_size=args.step_size,
        gamma=args.gamma,
        label_smoothing=args.label_smoothing,
        warmup_epochs=args.warmup_epochs,
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=len(train_loader))

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    best_acc = 0.0
    t0 = time.time()
    for epoch in range(1, cfg.epochs+1):
        if use_ddp:
            train_sampler.set_epoch(epoch)
        model.train()
        run_loss = 0.0
        corr = tot = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type=='cuda')):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None and args.scheduler_step == 'step':
                scheduler.step()
            run_loss += loss.item() * x.size(0)
            corr += (logits.argmax(1) == y).sum().item()
            tot += y.size(0)
        if scheduler is not None and args.scheduler_step == 'epoch':
            scheduler.step()
        train_acc = 100.0 * corr / tot
        test_acc = evaluate(model if not use_ddp else model.module, test_loader, device)
        log(rank, f"[Epoch {epoch:03d}/{cfg.epochs}] loss={run_loss/tot:.4f} train_acc={train_acc:.2f}% test_acc={test_acc:.2f}%")
        if test_acc > best_acc and rank == 0:
            path = save_float_checkpoint(model.module if use_ddp else model, args.dataset, args.arch, "float32")
            log(rank, f"[Checkpoint] Saved best float32 → {path} (acc={test_acc:.2f}%)")
            best_acc = test_acc
    if rank == 0:
        elapsed = time.time() - t0
        # size report
        fp32_path = ckpt_path(args.dataset, args.arch, "float32")
        fp32_size = os.path.getsize(fp32_path)/1e6 if os.path.exists(fp32_path) else 0
        log(rank, f"[Train Done] Best test acc: {best_acc:.2f}%, elapsed {elapsed:.1f}s")
        log(rank, f"[Size] float32: {fp32_size:.2f} MB")


def cmd_quantize(args):
    device = pick_device(args.device, local_rank=int(os.environ.get("LOCAL_RANK", 0)))
    # Load float32 model
    _, _, nc = get_datasets(args.dataset, args.data_root)
    model = build_model(args.arch, nc)
    load_float_into_model(model, args.dataset, args.arch, "float32", map_location="cpu")
    # Save quantized
    out = ckpt_path(args.dataset, args.arch, f"int{args.bits}_ptq")
    save_quantized_checkpoint(model, out, args.bits)
    # size report
    fp32_path = ckpt_path(args.dataset, args.arch, "float32")
    fp32_size = os.path.getsize(fp32_path)/1e6 if os.path.exists(fp32_path) else 0
    q_size = os.path.getsize(out)/1e6 if os.path.exists(out) else 0
    print(f"[Quantize] Saved INT{args.bits} checkpoint → {out}")
    print(f"[Size] float32: {fp32_size:.2f} MB")
    label = "int8" if args.bits==8 else ("int4" if args.bits==4 else "int16")
    print(f"[Size] {label}: {q_size:.2f} MB")


def cmd_eval(args):
    device = pick_device(args.device, local_rank=0)
    print(f"[Device] {device}")
    # build model
    _, test_loader, nc, _ = get_dataloaders(args.dataset, args.data_root, args.batch_size, args.workers, dist_mode=False)
    model = build_model(args.arch, nc).to(device)
    if args.bits is None:
        # float32
        load_float_into_model(model, args.dataset, args.arch, "float32", map_location=device)
        tag = "float32"
    else:
        # quantized
        meta = load_quantized_into_model(model, args.dataset, args.arch, args.bits, args.qtag, map_location=device)
        tag = f"int{meta['num_bits']}_{args.qtag}"
    acc = evaluate(model, test_loader, device)
    print(f"[Eval] {tag} accuracy = {acc:.2f}%")



@torch.no_grad()
def quick_eval_for_inspector(model: nn.Module, dataset: str, data_root: str, device: torch.device, workers: int, batch_size: int) -> float:
    # Use the same test loader as eval
    _, test_loader, _, _ = get_dataloaders(dataset, data_root, batch_size, workers, dist_mode=False)
    return evaluate(model, test_loader, device)


def show_structure(model: nn.Module):
    print(model)


def show_float_weights(model: nn.Module, layer_filter: Optional[List[str]] = None, max_vals: int = 16):
    print("\n[Float Weights Preview]")
    for n, p in model.named_parameters():
        if (layer_filter is None) or any(f in n for f in layer_filter):
            vals = p.detach().flatten()[:max_vals].tolist()
            print(f"{n:30s}  shape={tuple(p.shape)}  vals={[round(v,6) for v in vals]}")


def show_quantized_preview(qstate: Dict[str, Tuple[torch.Tensor, Optional[float]]], layer_filter: Optional[List[str]] = None, max_vals: int = 16):
    if not qstate:
        print("\n[No quantized tensors]")
        return
    print("\n[Quantized INT Weights]")
    for n, (q, s) in qstate.items():
        if (layer_filter is None) or any(f in n for f in layer_filter):
            vals = q.detach().flatten()[:max_vals].tolist()
            print(f"{n:30s}  dtype={q.dtype}  scale={s}  vals={vals}")


def cmd_inspect(args):
    device = pick_device(args.device, local_rank=0)
    # Build model for correct num_classes
    _, _, nc = get_datasets(args.dataset, args.data_root)
    model = build_model(args.arch, nc).to(device)

    # Load checkpoint (float or quantized)
    qinfo: Dict[str, Tuple[torch.Tensor, Optional[float]]] = {}
    tag = "float32"
    if args.weights:
        payload = torch.load(args.weights, map_location=device)
        if "state_dict" in payload:
            sd = strip_prefix_from_state_dict(payload["state_dict"])
            model.load_state_dict(sd, strict=True)
            tag = "float32"
        elif "qstate_dict" in payload and "meta" in payload and "scales" in payload["meta"]:
            qsd = payload["qstate_dict"]
            sc = payload["meta"]["scales"]
            dsd = {}
            for k, v in qsd.items():
                s = sc.get(k, None)
                qinfo[k] = (v, s)
                dsd[k] = v if s is None else dequantize_tensor(v, s)
            dsd = strip_prefix_from_state_dict(dsd)
            model.load_state_dict(dsd, strict=True)
            nb = payload["meta"].get("num_bits", 8)
            tag = f"int{nb}_ptq"
        else:
            raise ValueError("Unknown checkpoint format")
        print(f"[Loaded] {args.weights} → {tag}")
    else:
        # default: load trained float32
        load_float_into_model(model, args.dataset, args.arch, "float32", map_location=device)

    show_structure(model)

    
    if args.eval_acc:
        acc = quick_eval_for_inspector(model, args.dataset, args.data_root, device=device, workers=args.workers, batch_size=args.batch_size)
        print(f"\n[Test accuracy] {acc:.2f}% on {args.dataset}")

    # Previews
    layer_filter = args.layers if args.layers else None
    show_float_weights(model, layer_filter, args.max_vals)
    if args.show_raw:
        show_quantized_preview(qinfo, layer_filter, args.max_vals)


# ------------------------- Argparse & Main -------------------------


def ddp_init_if_needed(args):
    if args.cmd == "train" and args.dist:
        # torchrun provides envs
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")


def ddp_cleanup_if_needed(args):
    
    if args.cmd == "train" and args.dist:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    # ---- TRAIN (float32) ----
    sys.argv = [
        "main.py",                                      # argv[0] = program name
        "--data-root", "/home/vkamineni/Projects/RECC/code/trainAndQuantize/weights",
        "--device", "cuda",
        "--workers", "4",
        "--batch-size", "256",
        # "--compile",                                   # optional
    
        "train",
        "--dataset", "CIFAR10",
        # "--arch", "alexnet",
        "--arch", "resnet50",
        "--epochs", "300",
        "--lr", "0.001",
        "--weight-decay", "0.0005",
        "--optim", "sgd",
        "--momentum", "0.9",
        "--scheduler", "cosine",
        "--step-size", "60",
        "--gamma", "0.2",
        "--label-smoothing", "0.0",
        "--warmup-epochs", "100",
        "--scheduler-step", "epoch",
        # "--dist",                                      # include if launching with torchrun and you want DDP
    ]

    # ---- EVAL (float32) ----
    # sys.argv = [
    #     "main.py",
    #     "--data-root", "/home/vkamineni/Projects/RECC/code/trainAndQuantize/weights",
    #     "--device", "cuda",
    #     "--workers", "4",
    #     "--batch-size", "256",
    
    #     "eval",
    #     "--dataset", "CIFAR10",
    #     "--arch", "alexnet",
    # ]

    
    # ---- QUANTIZE (INT8) ----
    # sys.argv = [
    #     "main.py",
    #     "--data-root", "/home/vkamineni/Projects/RECC/code/trainAndQuantize/weights",
    
    #     "quantize",
    #     "--dataset", "CIFAR10",
    #     "--arch", "alexnet",
    #     "--bits", "8",
    # ]
    
    # #eval
    # sys.argv = [
    #     "main.py",
    #     "--data-root", "/home/vkamineni/Projects/RECC/code/trainAndQuantize/weights",
    #     "--device", "cuda",
    #     "--workers", "4",
    #     "--batch-size", "256",
    
    #     "eval",
    #     "--dataset", "CIFAR10",
    #     "--arch", "alexnet",
    #     "--bits", "8",
    #     "--qtag", "ptq",
    # ]
    
    p = argparse.ArgumentParser("GPU-First + DDP Trainer / PTQ / Eval / Inspector (fixed)")
    sub = p.add_subparsers(dest="cmd", required=True)

    # shared
    p.add_argument("--data-root", default="./data")
    p.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--compile", action="store_true")

    # train
    tr = sub.add_parser("train")
    tr.add_argument("--dataset", required=True, choices=["CIFAR10", "CIFAR100", "MNIST"])
    tr.add_argument("--arch", required=True, choices=["resnet18","resnet34","resnet50","resnet101","vgg16","alexnet","mlp"])
    tr.add_argument("--epochs", type=int, default=120)
    tr.add_argument("--lr", type=float, default=0.1)
    tr.add_argument("--weight-decay", type=float, default=5e-4)
    tr.add_argument("--optim", default="sgd", choices=["sgd","adamw"])
    tr.add_argument("--momentum", type=float, default=0.9)
    tr.add_argument("--scheduler", default="cosine", choices=["cosine","step","none"])
    tr.add_argument("--step-size", type=int, default=60)
    tr.add_argument("--gamma", type=float, default=0.2)
    tr.add_argument("--label-smoothing", type=float, default=0.0)
    tr.add_argument("--warmup-epochs", type=int, default=0)
    tr.add_argument("--scheduler-step", choices=["epoch","step"], default="epoch")
    tr.add_argument("--dist", action="store_true", help="Enable DDP when launched with torchrun")

    # quantize
    qz = sub.add_parser("quantize")
    qz.add_argument("--dataset", required=True, choices=["CIFAR10", "CIFAR100", "MNIST"])
    qz.add_argument("--arch", required=True, choices=["resnet18","resnet34","resnet50","resnet101","vgg16","alexnet","mlp"])
    qz.add_argument("--bits", type=int, required=True, choices=[4,8,16])

    # eval
    ev = sub.add_parser("eval")
    ev.add_argument("--dataset", required=True, choices=["CIFAR10", "CIFAR100", "MNIST"])
    ev.add_argument("--arch", required=True, choices=["resnet18","resnet34","resnet50","resnet101","vgg16","alexnet","mlp"])
    ev.add_argument("--bits", type=int, choices=[4,8,16])
    ev.add_argument("--qtag", default="ptq")

    # inspect
    ins = sub.add_parser("inspect")
    ins.add_argument("--dataset", required=True, choices=["CIFAR10", "CIFAR100", "MNIST"])
    ins.add_argument("--arch", required=True, choices=["resnet18","resnet34","resnet50","resnet101","vgg16","alexnet","mlp"])
    ins.add_argument("--weights", default="")
    ins.add_argument("--layers", nargs="*")
    ins.add_argument("--max-vals", type=int, default=16)
    ins.add_argument("--show-raw", action="store_true", help="Show raw quantized ints + scale if checkpoint is quantized")
    ins.add_argument("--eval-acc", action="store_true")
    
    args = p.parse_args()

    print(args)
    ddp_init_if_needed(args)
    
    try:
        if args.cmd == "train":
            cmd_train(args)
        elif args.cmd == "quantize":
            cmd_quantize(args)
        elif args.cmd == "eval":
            cmd_eval(args)
        elif args.cmd == "inspect":
            cmd_inspect(args)
        else:
            raise ValueError(args.cmd)
    finally:
        ddp_cleanup_if_needed(args)