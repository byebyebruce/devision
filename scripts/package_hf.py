"""Lay out one training round's checkpoint as a Hugging Face model repository (nothing is uploaded).

    uv run --with matplotlib python scripts/package_hf.py --round runs/v2-align-lora --stage stage2b \
        --name devision-v2 --test-prefix v2eval_ --out ../devision-hf

The layout follows convaiinnovations/laya: weights and configs at the root, a model card as README.md,
and eval/ with results.json (every number of the card, from the round's evaluation summaries),
results.md (the same as tables) and plots. Per-question details stay in the code repository's runs/.
Uploading is a separate, deliberate step (`hf upload <repo> <out>`).
"""
import argparse
import json
import os
import shutil
from collections import defaultdict

CHECKPOINT_FILES = ["model.safetensors", "encoder", "tokenizer", "vision"]
TEST_SETS = ["eval_coco_exist", "eval_coco_size", "eval_vqa_choice", "eval_gqa", "eval_coco_position",
             "eval_vqa_yesno", "eval_coco_relation", "eval_pope"]
LAYA_VISION = ["vqav2_yesno", "aokvqa", "scienceqa"]
LAYA_VISION_POPE = {"random": 0.836, "popular": 0.819, "adversarial": 0.777}   # their published scores
# The default .gitattributes of a new Hugging Face model repository (large and binary files through LFS).
GITATTRIBUTES = "".join("%s filter=lfs diff=lfs merge=lfs -text\n" % p for p in [
    "*.7z", "*.arrow", "*.bin", "*.bz2", "*.ckpt", "*.ftz", "*.gz", "*.h5", "*.joblib", "*.lfs.*", "*.mlmodel",
    "*.model", "*.msgpack", "*.npy", "*.npz", "*.onnx", "*.ot", "*.parquet", "*.pb", "*.pickle", "*.pkl", "*.pt",
    "*.pth", "*.rar", "*.safetensors", "saved_model/**/*", "*.tar.*", "*.tar", "*.tflite", "*.tgz", "*.wasm",
    "*.xz", "*.zip", "*.zst", "*tfevents*", "*.png"])


def load(path):
    with open(path) as f:
        return json.load(f)


def rnd(x, n=6):
    return round(x, n) if isinstance(x, float) else x


def summary(path):
    """The card's numbers for one evaluated set, plus its controls when they were run."""
    r = load(path)
    out = {"role": r["run"].get("role"), "questions": r["n"], "pictures": r.get("images"),
           "accuracy": rnd(r["accuracy_all"]), "accuracy_by_type": {k: rnd(v) for k, v in r["accuracy"].items() if v is not None},
           "nll": rnd(r["nll"]), "brier": rnd(r.get("brier")), "ece": rnd(r["ece"])}
    if r.get("by_axis") and set(r["by_axis"]) - {"-"}:
        out["accuracy_by_axis"] = {k: {"questions": v["n"], "accuracy": rnd(v["accuracy"])} for k, v in r["by_axis"].items()}
    if r.get("by_kind") and len(r["by_kind"]) > 1:
        out["accuracy_by_kind"] = {k: {"questions": v["n"], "accuracy": rnd(v["accuracy"])} for k, v in r["by_kind"].items()}
    if r.get("pope"):
        out["pope"] = {k.replace("pope-", ""): {m: rnd(v[m]) for m in ("accuracy", "precision", "recall", "f1", "yes_ratio")}
                       for k, v in r["pope"].items()}
    mism = path[:-5] + ".mismatched.json"
    if os.path.exists(mism):
        out["accuracy_with_mismatched_pictures"] = rnd(load(mism)["accuracy_all"])
    if r.get("latency_ms"):
        out["latency_ms"] = {k: rnd(v, 1) for k, v in r["latency_ms"].items() if k in ("p50", "p95")}
    return out


def results(round_dir, prefix, name, config):
    ev = os.path.join(round_dir, "eval")
    res = {"model": name, "round": os.path.basename(round_dir.rstrip("/")),
           "temperature": {"choice": rnd(config["temperature"][0]), "noul": rnd(config["temperature"][2])},
           "test_sets": {}, "laya_vision": {}}
    for s in TEST_SETS:
        p = os.path.join(ev, "%s%s.json" % (prefix, s))
        if os.path.exists(p):
            res["test_sets"][s[5:]] = summary(p)
    for s in LAYA_VISION:
        for subset in ("all", "unseen"):
            p = os.path.join(ev, "compare", "lv_%s.%s.json" % (s, subset))
            if os.path.exists(p):
                c = load(p)["all"]
                res["laya_vision"].setdefault(s, {})[subset] = {
                    "questions": c["questions"], "laya_vision_201m": rnd(c["accuracy_a"]), "devision": rnd(c["accuracy_b"]),
                    "difference": rnd(c["difference"]), "ci95": [rnd(x) for x in c["ci95"]]}
    p = os.path.join(ev, "lv_pope.json")
    if os.path.exists(p):
        pope = summary(p)["pope"]
        res["laya_vision"]["pope"] = {k: {"devision": pope[k]["accuracy"], "laya_vision_201m_published": v}
                                      for k, v in LAYA_VISION_POPE.items() if k in pope}
    lat = [v["latency_ms"]["p50"] for v in res["test_sets"].values() if v.get("latency_ms")]
    if lat:
        res["latency_ms_p50_one_question_mac_cpu"] = rnd(sorted(lat)[len(lat) // 2], 1)
    return res


def results_md(res):
    lines = ["# %s evaluation" % res["model"], "",
             "Through `decide()` on CPU with the fitted temperatures (choice %.2f, noul %.2f). The test sets hold no picture"
             " any training file used; *mismatched* is the accuracy when every picture is swapped for another one."
             % (res["temperature"]["choice"], res["temperature"]["noul"]), "",
             "| set | role | questions | accuracy | mismatched | NLL | ECE |", "|---|---|---|---|---|---|---|"]
    for s, v in res["test_sets"].items():
        lines.append("| %s | %s | %d | %.3f | %s | %.3f | %.3f |" % (
            s, v["role"], v["questions"], v["accuracy"],
            "%.3f" % v["accuracy_with_mismatched_pictures"] if "accuracy_with_mismatched_pictures" in v else "-",
            v["nll"], v["ece"]))
    axes = {s: v["accuracy_by_axis"] for s, v in res["test_sets"].items() if "accuracy_by_axis" in v}
    if axes:
        lines += ["", "By direction (tb = above/below, lr = left/right):", "", "| set | tb | lr |", "|---|---|---|"]
        for s, a in axes.items():
            lines.append("| %s | %.3f | %.3f |" % (s, a.get("tb", {}).get("accuracy", float("nan")),
                                                   a.get("lr", {}).get("accuracy", float("nan"))))
    if res["laya_vision"]:
        lines += ["", "## Against laya-vision 201M on its published questions", "",
                  "Paired question by question with their published predictions; 95% interval by picture bootstrap.", "",
                  "| set | questions | laya-vision 201M | deVision | difference [95% CI] |", "|---|---|---|---|---|"]
        for s in LAYA_VISION:
            for subset, v in res["laya_vision"].get(s, {}).items():
                lines.append("| %s (%s) | %d | %.3f | %.3f | %+.3f [%+.3f, %+.3f] |" % (
                    s, subset, v["questions"], v["laya_vision_201m"], v["devision"], v["difference"], *v["ci95"]))
        for k, v in res["laya_vision"].get("pope", {}).items():
            lines.append("| POPE %s (their published score) | 3000 | %.3f | %.3f | - |" % (
                k, v["laya_vision_201m_published"], v["devision"]))
    if "latency_ms_p50_one_question_mac_cpu" in res:
        lines += ["", "Latency: P50 %.0f ms per one-question request on a Mac CPU (warm-up excluded)."
                  % res["latency_ms_p50_one_question_mac_cpu"]]
    return "\n".join(lines) + "\n"


def plots(round_dir, prefix, res, out):
    try:
        import matplotlib  # pyright: ignore[reportMissingImports]  (run with `uv run --with matplotlib`)
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]
    except ImportError:
        print("matplotlib not available: no plots (run with `uv run --with matplotlib`)")
        return
    # reliability over every held-out test question
    bins = defaultdict(lambda: [0, 0.0, 0.0])
    for s in TEST_SETS:
        p = os.path.join(round_dir, "eval", "%s%s.details.jsonl" % (prefix, s))
        if not os.path.exists(p) or res["test_sets"].get(s[5:], {}).get("role") != "heldout":
            continue
        for line in open(p):
            r = json.loads(line)
            b = min(9, int(r["p_prediction"] * 10))
            bins[b][0] += 1
            bins[b][1] += r["p_prediction"]
            bins[b][2] += float(r["correct"])
    xs = sorted(bins)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([bins[b][1] / bins[b][0] for b in xs], [bins[b][2] / bins[b][0] for b in xs], "o-", color="#2b6cb0")
    ax.set_xlabel("confidence (probability of the chosen answer)")
    ax.set_ylabel("accuracy")
    ax.set_title("Reliability, held-out test questions (n=%d)" % sum(v[0] for v in bins.values()), fontsize=10)
    lo = min(0.5, min(bins[b][1] / bins[b][0] for b in xs)) - 0.05
    ax.set_xlim(lo, 1.0)
    ax.set_ylim(lo, 1.0)
    ax.plot([lo, 1], [lo, 1], "--", color="grey", lw=1)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "eval", "reliability_heldout.png"), dpi=150)
    plt.close(fig)
    # against laya-vision
    lv = res["laya_vision"]
    if lv:
        labels, ours, theirs = [], [], []
        for s in LAYA_VISION:
            if "all" in lv.get(s, {}):
                labels.append(s)
                ours.append(lv[s]["all"]["devision"])
                theirs.append(lv[s]["all"]["laya_vision_201m"])
        for k, v in lv.get("pope", {}).items():
            labels.append("pope_" + k)
            ours.append(v["devision"])
            theirs.append(v["laya_vision_201m_published"])
        fig, ax = plt.subplots(figsize=(7, 3.6))
        x = range(len(labels))
        ax.bar([i - 0.2 for i in x], theirs, 0.4, label="laya-vision 201M", color="#a0aec0")
        ax.bar([i + 0.2 for i in x], ours, 0.4, label=res["model"], color="#2b6cb0")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=20, fontsize=8)
        ax.set_ylim(0.3, 1.0)
        ax.set_ylabel("accuracy")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "eval", "benchmark_laya_vision.png"), dpi=150)
        plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--round", required=True, help="runs/<round>")
    p.add_argument("--stage", required=True, help="the stage whose checkpoint is published")
    p.add_argument("--name", required=True, help="model_name the API reports, e.g. devision-v2")
    p.add_argument("--test-prefix", default="", help="file prefix of the test-set results in <round>/eval/")
    p.add_argument("--card", default="docs/model-card.md")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    ckpt = os.path.join(a.round, a.stage)
    if os.path.exists(a.out):
        shutil.rmtree(a.out)
    os.makedirs(os.path.join(a.out, "eval"))
    for f in CHECKPOINT_FILES:
        src, dst = os.path.join(ckpt, f), os.path.join(a.out, f)
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        elif os.system("cp -c %s %s 2>/dev/null" % (src, dst)) != 0:   # APFS clone: no second 2 GB copy
            shutil.copy(src, dst)
    config = load(os.path.join(ckpt, "devision_config.json"))
    run = load(os.path.join(a.round, "run.json")) if os.path.exists(os.path.join(a.round, "run.json")) else {}
    config["model_name"] = a.name
    config["training"] = {"round": os.path.basename(a.round.rstrip("/")), "stage": a.stage,
                          "stages": [{k: s.get(k) for k in ("name", "init")} for s in run.get("stages", [])],
                          "git_commit": run.get("git_commit")}
    with open(os.path.join(a.out, "devision_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    res = results(a.round, a.test_prefix, a.name, config)
    with open(os.path.join(a.out, "eval", "results.json"), "w") as f:
        json.dump(res, f, indent=2)
    with open(os.path.join(a.out, "eval", "results.md"), "w") as f:
        f.write(results_md(res))
    plots(a.round, a.test_prefix, res, a.out)
    shutil.copy(a.card, os.path.join(a.out, "README.md"))
    with open(os.path.join(a.out, ".gitattributes"), "w") as f:
        f.write(GITATTRIBUTES)
    for root, _, files in sorted(os.walk(a.out)):
        for f in sorted(files):
            path = os.path.join(root, f)
            print("%10.1f MB  %s" % (os.path.getsize(path) / 2 ** 20, os.path.relpath(path, a.out)))


if __name__ == "__main__":
    main()
