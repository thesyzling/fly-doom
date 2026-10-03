"""Download a pinned Laya Vision checkpoint without replacing the text Laya package."""

import argparse
import json
from pathlib import Path
import subprocess

from flydoom.calibration import write_json
from flydoom.data import digest


MODEL_REVISION = "f2fe3c12cb6d04c59d8a190250bf3fb40fc828dc"


def prepare(output, source, revision=MODEL_REVISION):
    from huggingface_hub import HfApi, snapshot_download
    output, source = Path(output), Path(source)
    manifest = output / "download.json"
    if manifest.exists():
        raise FileExistsError("A pinned download already exists; use another output directory")
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    repo = "thaitea/laya-vision"
    revision = HfApi().model_info(repo, revision=revision).sha
    print(f"Downloading {repo}@{revision}; source {commit}", flush=True)
    snapshot_download(repo, revision=revision, local_dir=output,
                      allow_patterns=["*.json", "*.safetensors", "processor/*", "backbone/*", "README.md"])
    hashes = {p.relative_to(output).as_posix(): digest(p, "sha256")
              for p in output.rglob("*") if p.is_file() and ".cache" not in p.parts}
    sources = {p.relative_to(source).as_posix(): digest(p, "sha256") for p in (source / "laya").rglob("*.py")}
    write_json(manifest, {"schema": "laya_vision_download_v1", "repo": repo, "revision": revision,
                         "source_commit": commit, "source_sha256": sources, "sha256": hashes})
    print(f"Verified manifest: {manifest}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("models/laya-vision"))
    parser.add_argument("--source", type=Path, default=Path("models/laya-vision-source"))
    parser.add_argument("--revision", default=MODEL_REVISION)
    prepare(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
