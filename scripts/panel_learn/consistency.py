"""EXP74 step 0 (needs the model-run permission): the frozen K candidate's decision weights (runs/v04-k/cand, loaded as a
plain Decider, reading off -- the reading step is what produced the states) answer the sealed read states
(data/panel_learn/read_states_*.jsonl) with decide(state, question, image=None); every prediction must equal the
sealed evaluation's prediction. Otherwise the transfer check of D2 is not this decision path: invalid, stop."""
import json
import os
import sys

CAND = "runs/v04-k/cand"
SETS = ("sqa_temp", "sqa_force", "sqa_temp_mism", "sqa_force_mism")


def main(out: str, device: str = "auto") -> int:
    from devision.infer import Decider
    d = Decider.load(CAND, device=device)
    res, bad = {}, []
    for name in SETS:
        sealed = {(r["sample_id"], r["qid"]): r["prediction"] for r in map(json.loads, open("runs/v04-k/eval/extra_%s.details.jsonl" % name))}
        hits = n = 0
        for row in map(json.loads, open("data/panel_learn/read_states_%s.jsonl" % name)):
            ans = d.decide(row["state_text"], row["questions"], image=None)["answers"]
            for qid, a in ans.items():
                n += 1
                if a["choice"] != sealed[(row["id"], qid)]:
                    bad.append((name, row["id"], qid, a["choice"], sealed[(row["id"], qid)]))
                g = row["gold"][qid]["probabilities"]
                hits += a["choice"] == max(g, key=lambda o: g[o])
        res[name] = {"questions": n, "accuracy": hits / n}
    res["mismatches"] = bad
    res["consistent"] = not bad
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res, indent=1))
    return 0 if not bad else 11


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
