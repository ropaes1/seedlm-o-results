"""Fetch the subject GGUF (unsloth/Qwen3-1.7B-GGUF, Q4_K_M) into this folder.

Exactly the RUNBOOK download line, as a script so it can be backgrounded.
"""

from __future__ import annotations

import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "unsloth/Qwen3-1.7B-GGUF"
FILENAME = "Qwen3-1.7B-Q4_K_M.gguf"

if __name__ == "__main__":
    dest = Path(__file__).parent
    p = hf_hub_download(repo_id=REPO, filename=FILENAME,
                        local_dir=str(dest))
    print(p)
    print(f"{Path(p).stat().st_size:,} bytes", file=sys.stderr)
