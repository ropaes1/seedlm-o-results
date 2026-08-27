"""Phase 2 early probe: does an outlier side-channel help the seed fit MORE
than it helps RTN? Full seed search, one tensor, honest bpw everywhere."""
import sys
sys.path.insert(0, ".")
import torch
from safetensors import safe_open
from phase1_fit import seedlm_fit, with_outliers, rtn_int4, rel_err

TENSOR = "model.layers.11.self_attn.o_proj.weight"
dev = "cuda" if torch.cuda.is_available() else "cpu"

with safe_open("models/originals/Qwen3-0.6B/model.safetensors", framework="pt") as f:
    w = f.get_tensor(TENSOR).to(dev)


def rtn_with_outliers(w, pct):
    k = max(1, int(w.numel() * pct / 100))
    flat = w.flatten().float()
    idx = flat.abs().topk(k).indices
    kept = flat[idx].clone()
    masked = flat.clone()
    masked[idx] = 0.0
    deq = rtn_int4(masked.view(w.shape)).flatten()
    deq[idx] = kept
    bpw = 4.5 + k * (16 + 32) / w.numel()
    return deq.view(w.shape), bpw


print(f"{TENSOR}  {tuple(w.shape)}\n")
print(f"{'variant':<28}{'bpw':>6}{'rel_err':>9}")

deq, bpw = seedlm_fit(w, 8, 3, 65535)
print(f"{'seed only (full search)':<28}{bpw:>6.2f}{rel_err(w, deq):>9.4f}")

for pct in [0.5, 1.0, 2.0]:
    deq, bpw = with_outliers(w, pct, 8, 3, 65535)
    print(f"{f'seed + {pct}% outliers':<28}{bpw:>6.2f}{rel_err(w, deq):>9.4f}")

deq = rtn_int4(w)
print(f"{'rtn4 only':<28}{4.5:>6.2f}{rel_err(w, deq):>9.4f}")
for pct in [1.0]:
    deq, bpw = rtn_with_outliers(w, pct)
    print(f"{f'rtn4 + {pct}% outliers':<28}{bpw:>6.2f}{rel_err(w, deq):>9.4f}")
