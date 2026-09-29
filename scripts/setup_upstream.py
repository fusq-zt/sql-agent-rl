"""Fetch the pinned Agent Lightning source and apply the small runtime overlay.

This installs no Python packages and launches no service or training job.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=Path("vendor/agent-lightning"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    spec = json.loads((root / "runtime/upstream.json").read_text())
    patch = root / "runtime/agent-lightning.patch"
    if hashlib.sha256(patch.read_bytes()).hexdigest() != spec["patch_sha256"]:
        raise RuntimeError("Runtime patch has changed")
    dest = args.dest.resolve()
    if dest.exists():
        raise FileExistsError("Use a new destination; this script will not overwrite an existing checkout")
    dest.mkdir(parents=True)
    def git(*argv):
        return subprocess.check_output(["git", "-C", str(dest), *argv], stderr=subprocess.STDOUT)
    git("init")
    git("config", "core.autocrlf", "false")
    git("fetch", "--depth", "1", spec["repository"], spec["commit"])
    git("checkout", "--detach", "FETCH_HEAD")
    if git("rev-parse", "HEAD").decode().strip() != spec["commit"]:
        raise RuntimeError("Unexpected upstream revision")
    for rel, item in spec["files"].items():
        if hashlib.sha256((dest / rel).read_text(encoding="utf-8").encode()).hexdigest() != item["base_sha256"]:
            raise RuntimeError(f"Unexpected upstream file: {rel}")
    git("apply", "--check", str(patch))
    git("apply", str(patch))
    for rel, item in spec["files"].items():
        if hashlib.sha256((dest / rel).read_text(encoding="utf-8").encode()).hexdigest() != item["sha256"]:
            raise RuntimeError(f"Unexpected runtime result: {rel}")
    print(f"Prepared Agent Lightning {spec['version']} at {dest}")


if __name__ == "__main__":
    main()
