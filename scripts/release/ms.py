"""ModelScope mirror of an already built HF release, preserving repository visibility.

uv run --with modelscope-hub python scripts/release/ms.py build release/v0.3
uv run --with modelscope-hub python scripts/release/ms.py publish release/v0.3 --push
"""
import argparse
from pathlib import Path
import shutil
import subprocess

import hf


FOLDER = Path(hf.ROOT) / "runs/release/modelscope"
MANIFEST = "MANIFEST.modelscope.json"


def model_card(text, cfg):
    repo = cfg["modelscope_repo"]
    text = text.replace('src="assets/logo.png"',
                        f'src="https://modelscope.cn/models/{repo}/resolve/master/assets/logo.png"')
    spec = hf.package_spec(cfg)
    text = text.replace("pip install %s\n" % spec, "pip install %s modelscope-hub\n" % spec)
    text = text.replace(
        f'model = devision.load("{cfg["hub_repo"]}")   # latest release; device defaults to auto',
        'from modelscope_hub import HubApi\n\n'
        f'model_dir = HubApi().download_repo("{repo}", repo_type="model")\n'
        'model = devision.load(model_dir)   # device defaults to auto')
    return text.replace(
        f'devision-serve --checkpoint {cfg["hub_repo"]} --port 8000',
        f'ms-hub download {repo} --local-dir ./devision-model\n'
        'devision-serve --checkpoint ./devision-model --port 8000')


def build(release_dir):
    cfg = hf.release_config(release_dir)
    source = Path(hf.build_dir(cfg))
    if source.resolve() == FOLDER.resolve():
        raise SystemExit("source and ModelScope staging must be different")
    problems = hf.verify(release_dir, str(source), latency=False)
    if problems:
        raise SystemExit(str(problems))
    if FOLDER.exists():
        shutil.rmtree(FOLDER)
    # APFS clone when available; this is staging, never the training checkpoint.
    def copy(src, dst):
        if subprocess.call(["cp", "-c", src, dst], stderr=subprocess.DEVNULL) != 0:
            shutil.copy2(src, dst)
        return dst
    shutil.copytree(source, FOLDER, copy_function=copy)
    (FOLDER / "README.md").write_text(model_card((source / "README.md").read_text(), cfg))
    manifest = hf.manifest(str(FOLDER))
    hf.write_json(str(FOLDER / "MANIFEST.json"), manifest)
    hf.write_json(str(Path(release_dir) / MANIFEST), manifest)
    print(f"built {FOLDER}; weights identical to the HF release")


def check_remote(api, repo, revision, manifest):
    files = {f.path: f for f in api.list_repo_files(repo, "model", revision=revision) if f.type == "blob"}
    for name, expected in manifest.items():
        actual = files.get(name)
        if actual is None or actual.size != expected["size"] or actual.sha256 != expected["sha256"]:
            raise SystemExit(f"remote mismatch at {revision}/{name}")
    print(f"verified {len(manifest)} remote file sizes and SHA256s at {revision}")


def publish(release_dir, push):
    from modelscope_hub import HubApi
    cfg = hf.release_config(release_dir)
    problems = hf.verify(release_dir, str(FOLDER), latency=False, manifest_name=MANIFEST)
    if problems:
        raise SystemExit(str(problems))
    api = HubApi()
    repo = cfg["modelscope_repo"]
    info = api.get_repo(repo, "model")  # existing repository only; do not change visibility
    revisions = {r["Revision"] for r in api.list_repo_revisions(repo, "model")}
    if cfg["version"] in revisions:
        raise SystemExit(f"refusing to overwrite existing version {cfg['version']}")
    print(f"{'PUSH' if push else 'DRY RUN'} {repo}, {cfg['version']}, private={info.private}")
    if not push:
        return
    result = api.upload_folder(repo, "model", str(FOLDER),
                               commit_message=f"deVision {cfg['version']} (github {cfg['git_tag']})",
                               max_workers=4, disable_tqdm=True)
    manifest = hf.load_json(str(Path(release_dir) / MANIFEST))
    check_remote(api, repo, "master", manifest)
    tag = api.create_repo_tag(repo, "model", cfg["version"], revision="master")
    check_remote(api, repo, cfg["version"], manifest)
    hf.write_json(str(FOLDER.parent / "modelscope-upload.json"), {
        "repo": repo, "version": cfg["version"], "private": info.private,
        "upload": result, "tag": tag, "verified_files": len(manifest),
        "weight_sha256": manifest["model.safetensors"]["sha256"]})
    print(f"published https://modelscope.cn/models/{repo}, tag {cfg['version']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["build", "publish"])
    p.add_argument("release_dir")
    p.add_argument("--push", action="store_true")
    args = p.parse_args()
    if args.command == "build":
        build(args.release_dir)
    else:
        publish(args.release_dir, args.push)
