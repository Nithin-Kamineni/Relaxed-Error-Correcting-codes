import torch # type: ignore

def load_quant_info_from_checkpoint(path: str):
    """
    Loads the new (copied) quantized checkpoint and returns (quant_info, num_bits, scales_dict).
    quant_info: { name: (q_tensor, scale) }
    """
    payload = torch.load(path, map_location="cpu")
    if not (isinstance(payload, dict) and "qstate_dict" in payload and "meta" in payload and "scales" in payload["meta"]):
        raise ValueError("Checkpoint missing qstate_dict/meta/scales.")
    qsd = payload["qstate_dict"]
    scales = payload["meta"]["scales"]
    num_bits = payload["meta"].get("num_bits", None)

    quant_info = {name: (qt, scales.get(name, None)) for name, qt in qsd.items()}
    return quant_info, num_bits, scales

def clone_quant_info(quant_info: dict):
    editable = {}
    for name, (q, s) in quant_info.items():
        # clone() to decouple from the loaded tensors; keep dtype/device
        editable[name] = (q.clone(), s)
    return editable