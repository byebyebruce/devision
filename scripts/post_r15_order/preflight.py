"""Validate all real A/B rows and their tokenisation; freeze configs and fingerprints before training."""
import copy
import json
import random
import subprocess
from collections import Counter
from pathlib import Path

import yaml
from transformers import AutoTokenizer

from build import MONITORS, OUT, rows, sha
from report import COUNTS, RULES
from devision.infer import question_item
from devision.model import ModelConfig
from devision.train.pipeline import build_plan
from devision.train.rlcd import plan_batches
from devision.train.samples import read_decision_jsonl

RUN = Path("runs/post-r15-order")
CHECKPOINT = Path("runs/r15-vision/stage2")


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    if (RUN / "frozen.json").exists():
        raise SystemExit("preflight already frozen; do not overwrite after launch")
    manifest = json.loads((OUT / "MANIFEST.json").read_text())
    for rel, expected in manifest["output_sha256"].items():
        assert sha(OUT / rel) == expected, rel
    a, b = [read_decision_jsonl(str(OUT / (arm + ".jsonl"))) for arm in ("a", "b")]
    tok = AutoTokenizer.from_pretrained(CHECKPOINT / "tokenizer", local_files_only=True)
    cfg = ModelConfig.from_dict(json.loads((CHECKPOINT / "config.json").read_text()))
    plans, changed, max_length = {}, 0, 0
    for x, y in zip(a, b, strict=True):
        for key in ("id", "image", "image_id", "group", "gold", "view", "original_id", "block"):
            assert x[key] == y[key], (x["id"], key)
        qx, qy = x["questions"]["q"], y["questions"]["q"]
        assert qx == qy  # Dict equality deliberately ignores criteria enumeration order.
        ix, iy = [question_item(tok, cfg, r.get("state_text", ""), r["questions"]["q"]) for r in (x, y)]
        count = 2 if qx["type"] == "noul" else len(qx["criteria"])
        assert len(ix["markers"]) == len(iy["markers"]) == count, x["id"]
        assert ix["options_distinct"] == iy["options_distinct"] == count, x["id"]
        assert len(ix["ids"]) == len(iy["ids"]), x["id"]
        max_length = max(max_length, len(ix["ids"]))
        if qx["type"] == "choice" and x["view"] == 1:
            assert list(qx["criteria"]) != list(qy["criteria"]), x["id"]
            changed += 1
        else:
            assert ix == iy, x["id"]
    for arm, ss in (("a", a), ("b", b)):
        its = [dict(group=s["group"], qid="q") for s in ss]
        groups = Counter((it["group"], it["qid"]) for it in its)
        assert set(groups.values()) <= {2, 4}
        plan = plan_batches(its, 8, random.Random(1))
        assert len(plan) == 2000 and all(len(batch) == 8 for batch in plan)
        plans[arm] = [[ss[i]["id"] for i in batch] for batch in plan]
    assert plans["a"] == plans["b"]
    (RUN / "batches_seed1.json").write_text(json.dumps(plans["a"]))
    for name, path in MONITORS.items():
        ss = read_decision_jsonl(path)
        assert sum(len(s["questions"]) for s in ss) == COUNTS[name], name
        for s in ss:
            assert (Path("data") / s["image"]).is_file(), s["image"]
    params = dict(epochs=1, micro_batch=8, lr_new=1e-5, lr_head=5e-6, lr_lora=2e-5,
                  warmup=100, eval_every=1000, save_every=500, amp="off", seed=1,
                  monitor_details=True, vision_unfreeze_top=0, distill_weight=0.0,
                  question_conditioning=False, grounding_weight=0.0)
    config_paths = []
    for suffix, arm, seed in (("a", "a", 1), ("b", "b", 1), ("b-seed2", "b", 2)):
        local = dict(params, seed=seed)
        if arm == "b":
            local.update(start_check="runs/post-r15-order-a/stage2/monitor/step0", start_tolerance=1e-5)
        config = dict(name="post-r15-order-" + suffix, device="mps", common={"log_every": 100},
                      stages=[dict(name="stage2", kind="train", init=str(CHECKPOINT),
                                   data=str(OUT / (arm + ".jsonl")), val=MONITORS["val"],
                                   eval={k: v for k, v in MONITORS.items() if k != "val"}, params=local)])
        build_plan(config)
        path = Path("configs/post-r15-order-" + suffix + ".yaml")
        path.write_text("# Frozen by scripts/post_r15_order/preflight.py before the first update.\n" + yaml.safe_dump(config, sort_keys=False))
        config_paths.append(str(path))
    ev = copy.deepcopy(yaml.safe_load(Path("configs/r15-vision.yaml").read_text())["evaluate"])
    ev.update(checkpoint="runs/post-r15-order-b/stage2")
    ev.pop("commands", None)
    ev["history"] += [v for k, v in MONITORS.items() if k != "val"]
    eval_config = dict(name="post-r15-order-final", device="mps", evaluate=ev)
    p = Path("configs/post-r15-order-final.yaml")
    p.write_text(yaml.safe_dump(eval_config, sort_keys=False))
    config_paths.append(str(p))
    checkpoint_sha = sha(CHECKPOINT / "model.safetensors")
    assert checkpoint_sha == "110fea95889947c373258c88a396ad3eec574971815ee437b9c7cc3df8a5d2d5", "r15 release weights differ"
    tracked = [str(OUT / "MANIFEST.json"), *config_paths,
               *[str(p) for p in sorted(Path("scripts/post_r15_order").glob("*.py"))],
               "docs/experiments/2026-10-14-post-r15-training-plan.md",
               *manifest["input_sha256"], *manifest["exclusion_files"],
               *[str(OUT / p) for p in manifest["output_sha256"]],
               *[str(p) for p in sorted(Path("src/devision").rglob("*.py"))],
               *[str(p) for p in sorted(CHECKPOINT.rglob("*.json")) if p.parts[-2] in ("tokenizer", "encoder", "vision")],
               str(CHECKPOINT / "config.json")]
    frozen = dict(checkpoint=str(CHECKPOINT), checkpoint_sha256=checkpoint_sha,
                  code_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  file_sha256={p: sha(p) for p in tracked}, rules=RULES, steps=2000, exposures_per_arm=len(a),
                  changed_choice_views=changed, max_tokens=max_length,
                  batch_sha256=sha(RUN / "batches_seed1.json"), monitor_questions=sum(COUNTS.values()))
    (RUN / "frozen.json").write_text(json.dumps(frozen, indent=2) + "\n")
    print(json.dumps({k: frozen[k] for k in ("steps", "exposures_per_arm", "changed_choice_views", "max_tokens", "monitor_questions")}, indent=2))


if __name__ == "__main__":
    main()
