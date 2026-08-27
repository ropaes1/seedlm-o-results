"""Move results between the laptop, the pod and the private HF repo.

Phase 3 runs on two machines and they have to exchange exactly two small things:
Leg A's stage-1 results (so the pod learns which variant won) and, at the end,
everything the pod produced.  No cache and no model weights ever cross — fits
are deterministic from ``generator_seed=3407``, so the winner's *identity* is
the entire cross-machine payload.

    python hf_sync.py upload   --repo <your-hf-user>/seedlm-o-results \
        --local results/Qwen3-1.7B --path-in-repo results/Qwen3-1.7B
    python hf_sync.py download --repo <your-hf-user>/seedlm-o-results \
        --allow "results/Qwen3-1.7B/*" --local .laptop

This exists instead of a ``hf``/``huggingface-cli`` call because the CLI's
entry-point name has changed across huggingface_hub releases (``huggingface-cli``
-> ``hf``, and ``huggingface_hub.commands`` was removed in 1.x), while
:class:`huggingface_hub.HfApi` has been stable throughout.  A launcher that
shells out to the wrong name fails at the *end* of a paid pod run, which is the
worst possible moment.

The token is read from ``HF_TOKEN``; nothing is ever written to disk.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def cmd_upload(args: argparse.Namespace) -> int:
    """Upload a folder to the private results repo, creating it if needed.

    Returns:
        Process exit code (0 on success).
    """
    from huggingface_hub import HfApi

    local = Path(args.local)
    if not local.exists():
        print(f"hf_sync: nothing to upload, {local} does not exist",
              file=sys.stderr)
        return 1
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(args.repo, repo_type=args.repo_type, private=True,
                    exist_ok=True)
    url = api.upload_folder(
        folder_path=str(local),
        path_in_repo=args.path_in_repo,
        repo_id=args.repo,
        repo_type=args.repo_type,
        ignore_patterns=list(args.ignore) or None,
        commit_message=args.message,
    )
    print(f"hf_sync: uploaded {local} -> {args.repo}/{args.path_in_repo}")
    print(url)
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    """Download a subset of the results repo into a local directory.

    A missing repo or an empty match is **not** an error: the pod legitimately
    runs before Leg A has uploaded anything, and the launcher decides what to do
    about it (warn and skip, never fail the run).

    Returns:
        0 when something was fetched, 2 when there was nothing to fetch.
    """
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

    try:
        path = snapshot_download(
            repo_id=args.repo,
            repo_type=args.repo_type,
            allow_patterns=list(args.allow) or None,
            local_dir=args.local,
            token=os.environ.get("HF_TOKEN"),
        )
    except (RepositoryNotFoundError, HfHubHTTPError) as exc:
        print(f"hf_sync: nothing downloaded ({type(exc).__name__}: {exc})",
              file=sys.stderr)
        return 2
    got = [p for p in Path(path).rglob("*") if p.is_file()]
    print(f"hf_sync: downloaded {len(got)} file(s) into {path}")
    return 0 if got else 2


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-type", default="model")
    sub = ap.add_subparsers(dest="cmd", required=True)

    up = sub.add_parser("upload", help="push a local folder to the repo")
    up.add_argument("--repo", required=True)
    up.add_argument("--local", required=True)
    up.add_argument("--path-in-repo", required=True)
    up.add_argument("--ignore", action="append", default=[],
                    help="glob to skip; repeatable (ref_logits are huge and "
                         "regenerable, so the launchers skip them)")
    up.add_argument("--message", default="seedlm-o phase 3 results")
    up.set_defaults(func=cmd_upload)

    dn = sub.add_parser("download", help="pull part of the repo locally")
    dn.add_argument("--repo", required=True)
    dn.add_argument("--local", required=True)
    dn.add_argument("--allow", action="append", default=[])
    dn.set_defaults(func=cmd_download)

    args = ap.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
