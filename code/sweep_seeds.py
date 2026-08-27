"""Diagnostic: does seed-search budget explain the Gate 1b gap?
Sweeps n_seeds on one representative tensor; prints error scaling."""
import sys, time
sys.path.insert(0, ".")
import torch
from safetensors import safe_open
from phase1_fit import seedlm_fit, rtn_int4, rel_err

TENSOR = "model.layers.11.self_attn.o_proj.weight"
dev = "cuda" if torch.cuda.is_available() else "cpu"

with safe_open("models/originals/Qwen3-0.6B/model.safetensors", framework="pt") as f:
    w = f.get_tensor(TENSOR).to(dev)

print(f"{TENSOR}  {tuple(w.shape)}")
print(f"rtn4 (4.5 bpw) rel err : {rel_err(w, rtn_int4(w)):.4f}\n")
print(f"{'n_seeds':>8} {'rel_err':>9} {'time_s':>8}")
for n in [256, 2048, 16384, 65535]:
    t0 = time.time()
    deq, bpw = seedlm_fit(w, 8, 3, n)
    print(f"{n:>8} {rel_err(w, deq):>9.4f} {time.time()-t0:>8.1f}")
