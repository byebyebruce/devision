"""Release a trained checkpoint to the Hugging Face Hub from this repository (manual; release/README.md).

    uv run python scripts/release/hf.py build   release/v0.2      # -> runs/release/v0.2/ (the HF repository layout)
    uv run python scripts/release/hf.py verify  release/v0.2      # files, hashes, card, load + smoke answers on CPU
    uv run python scripts/release/hf.py publish release/v0.2      # dry run: what would be uploaded where
    uv run python scripts/release/hf.py publish release/v0.2 --push   # upload to the private repo, tag the version

release/<version>/release.yaml names the checkpoint (round + stage), the Hub repo, the version tag and the card
metadata; release/card_template.md + release/<version>/notes.md give the card's prose. Every number in the card
comes from the round's evaluation files (runs/<round>/eval/), never typed by hand. Files committed to git under
release/<version>/: results.json (the card's numbers), MANIFEST.json (size and SHA256 of every released file) and
smoke.json (fixed requests on examples/ pictures and the answers the source checkpoint gives; `verify` requires the
release to answer the same). The Hub repository is created private, and publishing stops if it is public.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CHECKPOINT = ["model.safetensors", "encoder", "vision", "tokenizer"]   # + the model config, released as config.json
REQUIRED = CHECKPOINT + ["config.json", "README.md", "LICENSE", "provenance.json", ".gitattributes",
                         "evaluation/results.json", "evaluation/results.md"]
GITATTRIBUTES = "*.safetensors filter=lfs diff=lfs merge=lfs -text\n*.bin filter=lfs diff=lfs merge=lfs -text\n"
TEST_SETS = [("test_exist", "COCO object presence"), ("test_vqa_choice", "VQAv2 multiple choice"),
             ("test_size", "COCO size"), ("bench_pope", "POPE, project filtered set"), ("test_gqa", "GQA val subset"),
             ("test_position", "COCO position"), ("test_vqa_yesno", "VQAv2 yes/no subset"),
             ("test_relation", "COCO relative position"), ("test_vsr", "VSR, project held-out split"),
             ("test_v7w", "Visual7W, project held-out split"), ("test_count_fresh", "Fresh counting test (unseen pictures)")]
LAYA_SETS = [("vqav2_yesno", "VQAv2 yes/no"), ("aokvqa", "A-OKVQA"), ("scienceqa", "ScienceQA with images")]
SMOKE_QUESTIONS = {
    "person": {"type": "noul", "instructions": "Is there a person in the image?"},
    "animal": {"type": "noul", "instructions": "Is there a dog in the image?"},
    "sport": {"type": "choice", "instructions": "Which sport is shown?",
              "criteria": {"tennis": None, "skiing": None, "soccer": None, "surfing": None}},
}
SMOKE_TOLERANCE = 1e-4


def load_json(path: str) -> Any:
    with open(path) as f:
        return json.load(f)


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
        f.write("\n")


def release_config(release_dir: str) -> Dict[str, Any]:
    import yaml
    with open(os.path.join(release_dir, "release.yaml")) as f:
        cfg = yaml.safe_load(f)
    for k in ("version", "git_tag", "model_name", "round", "stage", "hub_repo", "license"):
        if not cfg.get(k):
            raise SystemExit("release.yaml: %s is required" % k)
    return cfg


def build_dir(cfg: Dict[str, Any], root: str = ROOT) -> str:
    return os.path.join(root, "runs", "release", cfg["version"])


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def files_of(folder: str) -> List[str]:
    return sorted(os.path.relpath(os.path.join(d, f), folder) for d, _, fs in os.walk(folder) for f in fs
                  if not f.startswith(".") or f == ".gitattributes")


def manifest(folder: str) -> Dict[str, Dict[str, Any]]:
    return {rel: {"size": os.path.getsize(os.path.join(folder, rel)), "sha256": sha256(os.path.join(folder, rel))}
            for rel in files_of(folder) if rel != "MANIFEST.json"}


def git_commit(root: str = ROOT) -> Optional[str]:
    try:
        return subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


# ---------------------------------------------------------------- the card's numbers

def set_summary(path: str) -> Dict[str, Any]:
    r = load_json(path)
    out = {"questions": r["n"], "accuracy": round(r["accuracy_all"], 4), "nll": round(r["nll"], 4),
           "ece": round(r["ece"], 4)}
    mism = path[:-5] + ".mismatched.json"
    if os.path.exists(mism):
        out["accuracy_mismatched"] = round(load_json(mism)["accuracy_all"], 4)
    if r.get("latency_ms", {}).get("p50"):
        out["latency_ms_p50"] = round(r["latency_ms"]["p50"], 1)
    return out


def results(round_dir: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Every number the card shows, read from the round's evaluation files."""
    ev = os.path.join(round_dir, "eval")
    res: Dict[str, Any] = {"round": os.path.basename(round_dir.rstrip("/")),
                           "temperature": {"choice": config["temperature"][0], "noul": config["temperature"][2],
                                           "by_options": config.get("temperature_by_options", {})},
                           "test_sets": {}, "laya_vision": {}}
    for name, _ in TEST_SETS:
        p = os.path.join(ev, name + ".json")
        if os.path.exists(p):
            res["test_sets"][name] = set_summary(p)
    lv = os.path.join(ev, "compare", "laya_vision.json")
    if os.path.exists(lv):
        d = load_json(lv)
        for key, _ in LAYA_SETS:
            c = d["sets"].get(key, {}).get("all", {}).get("ours_minus_theirs")
            if c:
                res["laya_vision"][key] = {"questions": c["questions"], "laya_vision": round(c["theirs"], 4),
                                           "devision": round(c["ours"], 4), "difference": round(c["difference"], 4),
                                           "ci95": [round(x, 4) for x in c["ci95"]]}
        slice_ = d["sets"].get("scienceqa", {}).get("slices", {}).get("needs_picture&subject=natural science")
        if slice_:
            res["laya_vision"]["scienceqa_natural_needs_picture"] = {
                "questions": slice_["questions"], "laya_vision": round(slice_["theirs"], 4),
                "devision": round(slice_["ours"], 4), "difference": round(slice_["difference"], 4),
                "ci95": [round(x, 4) for x in slice_["ci95"]]}
        res["laya_vision"]["pope"] = {k.replace("pope-", ""): {"laya_vision_published": v["published"],
                                                               "devision": round(v["ours"]["accuracy"], 4)}
                                      for k, v in d.get("pope", {}).items()}
    for v in res["test_sets"].values():     # the evaluations ran on MPS / CUDA: their latencies are not CPU numbers
        v.pop("latency_ms_p50", None)
    return res


def cpu_latency(path: Optional[str], checkpoint: str) -> Optional[Dict[str, Any]]:
    """P50 / P95 of one-question requests from a `devision-eval --device cpu --no-group` result of this checkpoint."""
    if not path:
        return None
    r = load_json(os.path.join(ROOT, path) if not os.path.isabs(path) else path)
    if os.path.normpath(r["run"].get("checkpoint", "")) != os.path.normpath(checkpoint):
        raise SystemExit("cpu_latency %s measured %s, not %s" % (path, r["run"].get("checkpoint"), checkpoint))
    return {"p50": round(r["request_latency_ms"]["p50"]), "p95": round(r["request_latency_ms"]["p95"]),
            "threads": r["environment"]["threads"], "machine": "%s %s" % (r["environment"]["system"],
                                                                         r["environment"]["machine"]), "source": path}


def eval_table(res: Dict[str, Any]) -> str:
    rows = ["| Set | Questions | Accuracy | Mismatched | ECE |", "|---|---:|---:|---:|---:|"]
    for name, label in TEST_SETS:
        v = res["test_sets"].get(name)
        if v:
            rows.append("| %s | %s | %.3f | %s | %.3f |" % (label, format(v["questions"], ","), v["accuracy"],
                        "%.3f" % v["accuracy_mismatched"] if "accuracy_mismatched" in v else "–", v["ece"]))
    return "\n".join(rows)


def laya_table(res: Dict[str, Any]) -> str:
    rows = ["| Set | Questions | Laya Vision | deVision | Difference |", "|---|---:|---:|---:|---|"]
    labels = dict(LAYA_SETS, scienceqa_natural_needs_picture="ScienceQA natural science, needs the picture")
    for key in [k for k, _ in LAYA_SETS] + ["scienceqa_natural_needs_picture"]:
        v = res["laya_vision"].get(key)
        if v:
            rows.append("| %s | %s | %.3f | %.3f | %+.1f [%+.1f, %+.1f] |" % (
                labels[key], format(v["questions"], ","), v["laya_vision"], v["devision"], 100 * v["difference"],
                100 * v["ci95"][0], 100 * v["ci95"][1]))
    return "\n".join(rows)


def pope_line(res: Dict[str, Any]) -> str:
    p = res["laya_vision"].get("pope") or {}
    tiers = [t for t in ("random", "popular", "adversarial") if t in p]
    if not tiers:
        return ""
    return ("On the full POPE random / popular / adversarial sets (3,000 questions each) deVision scores **%s**, "
            "against Laya Vision's published **%s** (aggregate scores only, not paired)." % (
                " / ".join("%.3f" % p[t]["devision"] for t in tiers),
                " / ".join("%.3f" % p[t]["laya_vision_published"] for t in tiers)))


def temperature_table(res: Dict[str, Any]) -> str:
    names = {"choice:2": "Choice, 2 options", "choice:3-5": "Choice, 3–5 options", "choice:6-10": "Choice, 6–10 options",
             "choice:11+": "Choice, 11+ options", "noul:2": "Noul (yes / no)"}
    t = res["temperature"]
    rows = ["| Bucket | Temperature |", "|---|---:|"]
    rows += ["| %s | %.4f |" % (names.get(k, k), v) for k, v in sorted(t["by_options"].items())]
    rows += ["| Choice, any other count | %.4f |" % t["choice"]]
    return "\n".join(rows)


def results_md(res: Dict[str, Any], cfg: Dict[str, Any]) -> str:
    return "\n".join(["# deVision %s evaluation" % cfg["version"], "",
                      "Generated by `scripts/release/hf.py` from the evaluation of this release.", "", eval_table(res), "", "## Laya Vision 201M, same questions", "", laya_table(res),
                      "", pope_line(res), "", "## Temperatures", "", temperature_table(res), ""])


# ---------------------------------------------------------------- card

def notes(release_dir: str) -> Dict[str, str]:
    """notes.md: '## <key>' sections -> text."""
    out, key = {}, None
    for line in open(os.path.join(release_dir, "notes.md")):
        if line.startswith("## "):
            key = line[3:].strip()
            out[key] = ""
        elif key:
            out[key] += line
    return {k: v.strip() for k, v in out.items()}


def front_matter(cfg: Dict[str, Any], res: Dict[str, Any]) -> str:
    """YAML metadata of the card, including model-index entries for the test sets."""
    import yaml
    evals = [{"task": {"type": "visual-question-answering"},
              "dataset": {"name": label, "type": name},
              "metrics": [{"type": "accuracy", "value": v["accuracy"]}, {"type": "ece", "value": v["ece"]}]}
             for name, label in TEST_SETS for v in [res["test_sets"].get(name)] if v]
    meta = {"language": ["en"], "license": cfg["license"], "library_name": "devision",
            "pipeline_tag": "visual-question-answering", "base_model": cfg.get("base_model", []),
            "datasets": cfg.get("datasets", []),
            "tags": ["devision", "visual-decisions", "calibrated-probabilities", "jev", "system-one", "rlcd", "cpu"],
            "model-index": [{"name": "deVision %s" % cfg["version"], "results": evals}]}
    return "---\n" + yaml.safe_dump(meta, sort_keys=False, allow_unicode=True) + "---"


def card(release_dir: str, cfg: Dict[str, Any], res: Dict[str, Any], config: Dict[str, Any], weights_bytes: int) -> str:
    text = open(os.path.join(ROOT, "release", "card_template.md")).read()
    n = notes(release_dir)
    size, shuffle = config.get("image_size", 256), config.get("visual_shuffle", 2)
    fill = {"front_matter": front_matter(cfg, res), "version": cfg["version"],
            "git_tag": cfg["git_tag"], "hub_repo": cfg["hub_repo"], "image_size": str(size), "shuffle": str(shuffle),
            "visual_tokens": str((size // 16 // shuffle) ** 2), "weights_gb": "%.2f" % (weights_bytes / 1e9),
            "training": n.get("training", ""), "limitations": n.get("limitations", ""),
            "temperature_table": temperature_table(res), "eval_table": eval_table(res), "laya_table": laya_table(res),
            "pope_line": pope_line(res),
            "latency": ("P50 %(p50)d ms, P95 %(p95)d ms (%(machine)s, %(threads)d threads, FP32, one question per request,"
                        " warm-up excluded)" % res["cpu_latency"]) if res.get("cpu_latency") else "not measured"}
    for k, v in fill.items():
        text = text.replace("{{%s}}" % k, v)
    left = [w for w in text.split("{{")[1:]]
    if left:
        raise SystemExit("card template placeholders not filled: %s" % [w.split("}}")[0] for w in left])
    return text


# ---------------------------------------------------------------- smoke requests

def smoke_answers(model_dir: str, questions: Dict[str, Any]) -> Dict[str, Any]:
    """The answers a checkpoint gives to the smoke requests on every examples/ picture (CPU)."""
    import devision
    d = devision.load(model_dir, device="cpu")
    out = {}
    for pic in sorted(f for f in os.listdir(os.path.join(ROOT, "examples")) if f.endswith((".jpg", ".png"))):
        out[pic] = d.decide(state="", questions=questions, image=os.path.join(ROOT, "examples", pic))["answers"]
    return out


def same_answers(a: Dict[str, Any], b: Dict[str, Any], tol: float = SMOKE_TOLERANCE) -> List[str]:
    """Differences between two smoke answer sets; [] when they agree within `tol`."""
    diffs = []
    for pic in sorted(set(a) | set(b)):
        for qid in sorted(set(a.get(pic, {})) | set(b.get(pic, {}))):
            x, y = a.get(pic, {}).get(qid), b.get(pic, {}).get(qid)
            if x is None or y is None:
                diffs.append("%s/%s missing" % (pic, qid))
            elif x["type"] == "noul":
                if abs(x["noul"] - y["noul"]) > tol:
                    diffs.append("%s/%s noul %.6f vs %.6f" % (pic, qid, x["noul"], y["noul"]))
            elif x.get("choice") != y.get("choice") or any(
                    abs(x["probabilities"][k] - y["probabilities"].get(k, -1)) > tol for k in x["probabilities"]):
                diffs.append("%s/%s choice %s vs %s" % (pic, qid, x.get("choice"), y.get("choice")))
    return diffs


# ---------------------------------------------------------------- commands

def build(release_dir: str, out: Optional[str] = None) -> str:
    cfg = release_config(release_dir)
    round_dir = os.path.join(ROOT, cfg["round"]) if not os.path.isabs(cfg["round"]) else cfg["round"]
    ckpt = os.path.join(round_dir, cfg["stage"])
    out = out or build_dir(cfg)
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(os.path.join(out, "evaluation"))
    for f in CHECKPOINT:
        src, dst = os.path.join(ckpt, f), os.path.join(out, f)
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        elif subprocess.call(["cp", "-c", src, dst], stderr=subprocess.DEVNULL) != 0:   # APFS clone, no 2 GB copy
            shutil.copy(src, dst)
    # the model config is config.json (the Hub's standard name, which also counts downloads); a checkpoint saved
    # before 2026-10-07 calls it devision_config.json
    from devision.infer import config_path
    config = load_json(config_path(ckpt))
    config["model_name"] = cfg["model_name"]
    write_json(os.path.join(out, "config.json"), config)
    res = results(round_dir, config)
    internal_round = res.pop("round")      # only the version (v0.2) is public; the training round stays in provenance
    res["cpu_latency"] = cpu_latency(cfg.get("cpu_latency"), os.path.join(cfg["round"], cfg["stage"]))
    write_json(os.path.join(out, "evaluation", "results.json"), res)
    with open(os.path.join(out, "evaluation", "results.md"), "w") as f:
        f.write(results_md(res, cfg))
    with open(os.path.join(out, "README.md"), "w") as f:
        f.write(card(release_dir, cfg, res, config, os.path.getsize(os.path.join(out, "model.safetensors"))))
    shutil.copy(os.path.join(ROOT, "LICENSE"), os.path.join(out, "LICENSE"))
    with open(os.path.join(out, ".gitattributes"), "w") as f:
        f.write(GITATTRIBUTES)
    run = load_json(os.path.join(round_dir, "run.json")) if os.path.exists(os.path.join(round_dir, "run.json")) else {}
    write_json(os.path.join(out, "provenance.json"), {
        "version": cfg["version"], "git_tag": cfg["git_tag"], "built_from_commit": git_commit(),
        "round": internal_round, "stage": cfg["stage"], "training_commit": run.get("git_commit"),
        "training_started": run.get("started"), "stages": run.get("stages", []), "training_config": run.get("config"),
        "source_checkpoint_sha256": sha256(os.path.join(ckpt, "model.safetensors"))})
    # committed copies: the card's numbers, the manifest, the smoke answers of the source checkpoint
    write_json(os.path.join(release_dir, "results.json"), res)
    smoke = os.path.join(release_dir, "smoke.json")
    questions = cfg.get("smoke_questions") or SMOKE_QUESTIONS
    write_json(smoke, {"questions": questions, "tolerance": SMOKE_TOLERANCE, "source": cfg["round"] + "/" + cfg["stage"],
                       "answers": smoke_answers(ckpt, questions)})
    m = manifest(out)
    write_json(os.path.join(out, "MANIFEST.json"), m)
    write_json(os.path.join(release_dir, "MANIFEST.json"), m)
    print("built %s (%d files, %.2f GB); card, results, manifest and smoke answers written" % (
        out, len(m), sum(v["size"] for v in m.values()) / 1e9))
    return out


def verify(release_dir: str, folder: Optional[str] = None, latency: bool = True) -> List[str]:
    """Problems of a built (or downloaded) release folder; [] when it is ready."""
    import yaml
    cfg = release_config(release_dir)
    folder = folder or build_dir(cfg)
    problems = ["missing %s" % f for f in REQUIRED if not os.path.exists(os.path.join(folder, f))]
    want = load_json(os.path.join(release_dir, "MANIFEST.json"))
    got = manifest(folder)
    problems += ["%s: %s" % (rel, "missing" if rel not in got else "sha256 differs")
                 for rel, v in want.items() if rel not in got or got[rel]["sha256"] != v["sha256"]]
    problems += ["%s: not in the manifest" % rel for rel in got if rel not in want]
    text = open(os.path.join(folder, "README.md")).read() if os.path.exists(os.path.join(folder, "README.md")) else ""
    meta = yaml.safe_load(text.split("---")[1]) if text.startswith("---") else {}
    for k in ("license", "library_name", "base_model", "pipeline_tag"):
        if not meta.get(k):
            problems.append("card metadata: %s missing" % k)
    if "{{" in text:
        problems.append("card has unfilled placeholders")
    if problems:
        return problems
    smoke = load_json(os.path.join(release_dir, "smoke.json"))
    got_answers = smoke_answers(folder, smoke["questions"])
    problems += ["smoke: %s" % d for d in same_answers(smoke["answers"], got_answers, smoke["tolerance"])]
    config = load_json(os.path.join(folder, "config.json"))
    if config.get("model_name") != cfg["model_name"]:
        problems.append("config.json model_name %r, release says %r" % (config.get("model_name"), cfg["model_name"]))
    if latency and not problems:
        import devision
        d = devision.load(folder, device="cpu")
        pic = os.path.join(ROOT, "examples", sorted(smoke["answers"])[0])
        q = dict(list(smoke["questions"].items())[:1])
        d.decide(state="", questions=q, image=pic)
        times = []
        for _ in range(10):
            t = time.perf_counter()
            d.decide(state="", questions=q, image=pic)
            times.append(1000 * (time.perf_counter() - t))
        print("CPU one-question latency here: P50 %.0f ms (10 requests)" % sorted(times)[5])
    return problems


def publish(release_dir: str, push: bool, folder: Optional[str] = None) -> None:
    from huggingface_hub import HfApi
    cfg = release_config(release_dir)
    folder = folder or build_dir(cfg)
    problems = verify(release_dir, folder, latency=False)
    if problems:
        raise SystemExit("not publishing, verify found:\n  " + "\n  ".join(problems))
    api = HfApi()
    repo = cfg["hub_repo"]
    exists = api.repo_exists(repo)
    print("%s %s -> %s (%s), tag %s; %d files, %.2f GB" % ("PUSH" if push else "DRY RUN", folder, repo,
          "exists" if exists else "will be created private", cfg["version"], len(files_of(folder)),
          sum(os.path.getsize(os.path.join(folder, f)) for f in files_of(folder)) / 1e9))
    if exists and not api.model_info(repo).private:
        raise SystemExit("%s is public: publishing is stopped (making it public is the project owner's call)" % repo)
    tags = {t.name for t in api.list_repo_refs(repo).tags} if exists else set()
    if cfg["version"] in tags:
        raise SystemExit("tag %s already exists on %s: a release is never overwritten" % (cfg["version"], repo))
    if not push:
        print("dry run: nothing uploaded; add --push to upload")
        return
    if not exists:
        api.create_repo(repo, private=True, repo_type="model")
    if not api.model_info(repo).private:
        raise SystemExit("%s is not private after creation; stopping" % repo)
    commit = api.upload_folder(repo_id=repo, folder_path=folder, repo_type="model",
                               commit_message="deVision %s (github %s, %s)" % (cfg["version"], cfg["git_tag"],
                                                                                git_commit() or "?"))
    api.create_tag(repo, tag=cfg["version"], revision=commit.oid, tag_message="github %s" % cfg["git_tag"])
    print("uploaded %s, tagged %s at %s" % (repo, cfg["version"], commit.oid))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["build", "verify", "publish"])
    p.add_argument("release_dir", help="e.g. release/v0.2")
    p.add_argument("--folder", help="the release folder (default runs/release/<version>); verify also takes a "
                                    "downloaded Hub snapshot here")
    p.add_argument("--push", action="store_true", help="publish: really upload (default: dry run)")
    a = p.parse_args(argv)
    if a.command == "build":
        build(a.release_dir, a.folder)
    elif a.command == "verify":
        problems = verify(a.release_dir, a.folder)
        print("verify: OK" if not problems else "verify found:\n  " + "\n  ".join(problems))
        sys.exit(1 if problems else 0)
    else:
        publish(a.release_dir, a.push, a.folder)


if __name__ == "__main__":
    main()
