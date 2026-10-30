"""Pick r23's lr_vision (docs/experiments/2026-10-29-r23-v05-plan.md): 1e-6 (exp135) if it keeps every new use within
2.0 points of exp132 (5e-6), else 5e-6. Rewrites the lr_vision line of configs/r23-v05.yaml and prints the decision."""
import json
import re
import sys

NEW_USES = ["test_chartqa", "test_gui_shop_state", "test_gui_state", "test_gui", "test_ego"]


def accuracy(path):
    rows = [json.loads(line) for line in open(path) if line.strip()]
    return 100.0 * sum(bool(r.get("correct")) for r in rows) / len(rows)


def main():
    a, b = "runs/exp132-v4-offline-distill/eval", "runs/exp135-v4-lowlr-vision/eval"
    worst, lines = 0.0, []
    for name in NEW_USES:
        try:
            d = accuracy(f"{b}/{name}.details.jsonl") - accuracy(f"{a}/{name}.details.jsonl")
        except FileNotFoundError as e:
            lines.append(f"{name}: missing ({e.filename})")
            continue
        worst = min(worst, d)
        lines.append(f"{name}: exp135 - exp132 = {d:+.2f}")
    lr = "1.0e-06" if worst >= -2.0 and not any("missing" in s for s in lines) else "5.0e-06"
    text = open("configs/r23-v05.yaml").read()
    text = re.sub(r"(\n    lr_vision: ).*", r"\g<1>" + lr, text)
    open("configs/r23-v05.yaml", "w").write(text)
    print("\n".join(lines))
    print(f"worst {worst:+.2f} -> lr_vision {lr}")


if __name__ == "__main__":
    sys.exit(main())
