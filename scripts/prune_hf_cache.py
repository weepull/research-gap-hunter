"""Drop HuggingFace cache revisions and blobs the loader will never use.

Run immediately after pre-caching a model into a container image, in the *same*
Docker layer — Docker layers are additive, so pruning in a later instruction
leaves the bytes in the earlier layer and the image does not shrink at all.

Why this is needed: pre-caching `allenai/specter2_base` pulls two revisions of
roughly 440MB each. One carries `pytorch_model.bin` alongside the config and
tokenizer; the other carries only `model.safetensors`. Only the first is what
`main` resolves to.

That was established empirically, not assumed — each revision was deleted in
turn and the model loaded with the container network disabled. Deleting the
`.bin` revision breaks the load; deleting the safetensors-only revision does
not. This is the reverse of what `transformers.utils.SAFE_WEIGHTS_NAME` would
imply, which is exactly why it was measured.

The rule applied here is deliberately conservative and model-agnostic: keep any
snapshot that has a `config.json` (a revision the loader can actually resolve),
then drop blobs that no surviving snapshot references.
"""

import os
import pathlib
import shutil
import sys


def prune(cache_root: pathlib.Path) -> int:
    """Prune `cache_root`; return bytes reclaimed."""
    reclaimed = 0

    for model_dir in sorted(cache_root.glob("models--*")):
        snapshots = model_dir / "snapshots"
        blobs = model_dir / "blobs"
        if not snapshots.is_dir():
            continue

        for snapshot in sorted(snapshots.iterdir()):
            if snapshot.is_dir() and not (snapshot / "config.json").exists():
                print(f"  pruning unresolvable revision {snapshot.name[:12]}")
                shutil.rmtree(snapshot)

        # Blobs are shared between revisions, so only those nothing points at
        # any more are safe to remove.
        referenced = {
            os.path.realpath(entry)
            for snapshot in snapshots.iterdir()
            if snapshot.is_dir()
            for entry in snapshot.rglob("*")
            if entry.is_symlink()
        }

        if not blobs.is_dir():
            continue
        for blob in sorted(blobs.iterdir()):
            if blob.is_file() and str(blob.resolve()) not in referenced:
                size = blob.stat().st_size
                print(f"  pruning unreferenced blob {blob.name[:12]} ({size / 1e6:.0f} MB)")
                blob.unlink()
                reclaimed += size

    return reclaimed


def main() -> int:
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/opt/hf-cache")
    if not root.is_dir():
        print(f"HF cache {root} does not exist; nothing to prune")
        return 0

    reclaimed = prune(root)
    print(f"reclaimed {reclaimed / 1e6:.0f} MB from {root}")

    # A model that failed to pre-cache must not also fail the build — the
    # runtime can still download it. Report and move on.
    if not any(root.glob("models--*")):
        print("WARNING: no models present in the cache after pruning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
