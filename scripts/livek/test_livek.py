# pyright: reportAttributeAccessIssue=false, reportArgumentType=false, reportOptionalMemberAccess=false, reportOptionalSubscript=false
"""scripts/livek/livek.py without a model (LIVE-K-RUNNER-PREP-02): START binds PROTECT (pinned, complete, A / B / C true) and
writes the K bundle -- EXP71's components, the K rule, another identity than PROTECT's no-K bundle -- before any forward;
the live requests are tune_P's rows without their tables (the stem re-serialised with EXP56's table is the P state again);
the six reused panels hold no reading-routed question (science would be refused); the text fidelity is checked against
PROTECT's registered C_P_new; with a fake reader the report checks each read question's state against its own picture's
table (+ K) and walks the branches -- met, transfer not met, missing, a tampered state; the run's order and limit."""
import importlib.util
import json
import os
import shutil

import pytest

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(os.path.dirname(HERE))
spec = importlib.util.spec_from_file_location("livek", os.path.join(HERE, "livek.py"))
assert spec is not None and spec.loader is not None
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
has = all(os.path.exists(os.path.join(ROOT, p)) for p in ("runs/protect/report.json", "runs/exp71/report.json", "data/protect_p/P_K.jsonl"))
data = pytest.mark.skipif(not has, reason="PROTECT / EXP71 runs or data/protect_p not present")


@pytest.fixture(autouse=True)
def in_root(monkeypatch):
    monkeypatch.chdir(ROOT)


@pytest.fixture
def started(tmp_path):
    import time
    out = str(tmp_path / "livek")
    m.stage_start(int(time.time()) + 1800, out=out)
    return out


@data
def test_start_binds_protect_and_writes_the_k_bundle(started, monkeypatch):
    from devision.reading import K_RULE
    st = m.check_start(started, code=False)
    kb = json.load(open(os.path.join(started, "bundle_k", "reading.json")))
    assert kb["k_rule"] == json.loads(json.dumps(K_RULE)) and st["k_bundle"]["reading.json"] != st["protect"]["no_k_bundle"]
    assert {c: kb["sha256"][c] for c in ("selector.pt", "head.pt", "corrector.pt")} == {c: st["exp71"]["bundle_on"][c] for c in ("selector.pt", "head.pt", "corrector.pt")}
    assert kb["sha256"]["model.safetensors"] == m.NEW_SHA["model.safetensors"] and st["candidate_eligible"] is False
    monkeypatch.setitem(m.PROTECT, "report.json", "0" * 64)                   # another PROTECT report than the reviewed one
    with pytest.raises(SystemExit, match="PROTECT"):
        m.check_start(started, code=False)


@data
def test_live_rows_are_p_without_its_table():
    from devision.reading import HEADER, serialise
    src = {r["id"]: r for r in m.jsonl(m.TUNE_P)}
    for r in m.live_rows():
        full = src[r["id"]]["state_text"]
        lines = full.split("\n")
        table = [[ln.split(": ", 1)[0], *ln.split(": ", 1)[1].rsplit(" = ", 1)] for ln in lines[lines.index(HEADER) + 1:]]
        assert HEADER not in r["state_text"] and serialise(r["state_text"], table) == full


@data
def test_only_panels_without_reading_questions_are_reused(monkeypatch):
    assert set(m.reused_records()) == set(m.REUSED)
    monkeypatch.setattr(m, "REUSED", m.REUSED + ("science",))
    with pytest.raises(SystemExit, match="cannot be reused"):
        m.reused_records()


class FakeDecider:
    def __init__(self, probs):
        self.probs = probs
        self.cfg = type("C", (), {"temperature": None, "temperature_by_options": None})()

    def decide(self, state, questions, image=None):
        return {"answers": {k: {"type": "choice", "probabilities": dict(self.probs[(state, q["instructions"], tuple(q["criteria"]))])}
                            for k, q in questions.items()}}


def registered_c():
    rows, inputs = m.pr.rows_inputs(m.P_K)
    out = {}
    for r in m.e57.read_set(m.PROTECT_OUT, "C_P_new", rows, m.NEW_SHA, inputs):
        row = rows[r["sample_id"]]
        q = row["questions"][r["qid"]]
        out[(row["state_text"], q["instructions"], tuple(q["criteria"]))] = r["probabilities"]
    return out


@data
def test_text_fidelity_against_protect(started):
    probs = registered_c()
    assert m.stage_fid_text(started, decider=FakeDecider(probs))["reproduced"]
    off = {k: {o: (v + 2e-5 if i == 0 else v - 2e-5) for i, (o, v) in enumerate(p.items())} for k, p in probs.items()}
    with pytest.raises(SystemExit, match="does not reproduce"):
        m.stage_fid_text(started, decider=FakeDecider(off))


class FakeReading:
    """Reads EXP56's table for a tune_P picture (else a fixed table), routes and builds each read question's state exactly
    as Reading.decide does (k_block for temperature questions). Answers by (picture, question, options): a dev_sqa question
    on its fixed wrong picture gets a wrong option, every other question its gold (policy "right"); policy "wrong" also
    answers the dev_sqa questions on their true pictures wrongly (the rest unchanged)."""
    TABLE = [["Sample A", "mass of each particle", "14"], ["Sample B", "mass of each particle", "8"],
             ["Sample A", "average particle speed", "1,360"], ["Sample B", "average particle speed", "1,900"],
             ["Pair 1", "distance between the magnets", "2"], ["Pair 2", "distance between the magnets", "3"]]

    def __init__(self, policy="right", tamper=False):
        from devision.reading import HEADER
        self.policy, self.tamper, self.k_rule, self.corrector = policy, tamper, True, object()
        self.unreadable: dict = {}                    # (picture, template) -> "unreadable": a real read failure
        self.decider = FakeDecider({})
        self.p_tables, self.gold, self.wrong = {}, {}, set()
        for r in m.jsonl(m.TUNE_P):
            lines = r["state_text"].split("\n")
            self.p_tables[r["image"]] = [[ln.split(": ", 1)[0], *ln.split(": ", 1)[1].rsplit(" = ", 1)] for ln in lines[lines.index(HEADER) + 1:]]
        for name, path in list(m.TRANSFER.items()) + [("science", m.GUARD["science"]), ("P", m.TUNE_P)]:
            for r in m.jsonl(path):
                for qid, q in r["questions"].items():
                    opts = tuple(q["criteria"]) if q["type"] == "choice" else ("false", "true")
                    key = (r["image"], q["instructions"], opts)
                    self.gold[key] = max(r["gold"][qid]["probabilities"], key=r["gold"][qid]["probabilities"].get)
                    if name.endswith("_mism") or (policy == "wrong" and name in ("sqa_temp", "sqa_force")):
                        self.wrong.add(key)

    def decide(self, state, questions, image=None, diagnostics=None):
        from devision.reading import TEMPLATES, domain_of, k_block, reads_for, serialise
        path = getattr(image, "_path")
        tmpls = reads_for(questions)
        source = {tuple(row[:2]): row[2] for row in self.p_tables.get(path, self.TABLE)}
        table = [[*TEMPLATES[t], self.unreadable.get((path, t), source[tuple(TEMPLATES[t])])] for t in tmpls]   # Reading.read's order
        plain = serialise(state, table)
        kb = k_block(table)
        states = {k: (plain + "\n" + kb if kb and domain_of(q["instructions"]) == "temperature" else plain)
                  for k, q in questions.items() if tmpls and domain_of(q["instructions"])}
        if self.tamper and states:
            states[next(iter(states))] += " tampered"
        if diagnostics is not None:
            reads = [{"question": t, "value": v, "valid": v != "unreadable", "decoded": v.replace(",", "")} for t, (_, _, v) in zip(tmpls, table)]
            diagnostics.update(route={k: ("read" if k in states else "ordinary") for k in questions}, reads=reads or None,
                               table=table, states=states, k_block=kb if any(domain_of(q["instructions"]) == "temperature" for q in questions.values()) else None)
        ans = {}
        for k, q in questions.items():
            opts = list(q["criteria"]) if q["type"] == "choice" else ["false", "true"]
            key = (path, q["instructions"], tuple(opts))
            g = self.gold[key]
            right = key not in self.wrong
            pick = g if right else next(o for o in opts if o != g)
            probs = {o: (0.9 if o == pick else 0.1 / (len(opts) - 1)) for o in opts}
            ans[k] = {"type": "noul", "noul": probs.get("true", 0.5)} if q["type"] == "noul" else {"type": "choice", "probabilities": probs}
        return {"answers": ans}


@pytest.fixture
def tagged(monkeypatch):
    def ask(reading, image_path, text, questions, data_root, pictures, diagnostics=None):
        import io

        from PIL import Image
        raw = open(os.path.join(data_root, image_path), "rb").read()
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        img._path = image_path
        return reading.decide(text, questions, image=img, diagnostics=diagnostics)
    monkeypatch.setattr(m.e58, "ask", ask)


def chain(out, reading, skip=()):
    m.stage_fid_text(out, decider=FakeDecider(registered_c()))
    for s in ("live_p", "transfer", "guard_science"):
        if s not in skip:
            m.stage_live(out, s, reading=reading)
    return m.report(out)


@data
@pytest.mark.parametrize("case,branch", [("met", "the K combination meets"), ("not_met", "the K combination does not meet"),
                                         ("missing", "incomplete"), ("tampered", "invalid")])
def test_report_branches(started, tagged, case, branch):
    reading = FakeReading("wrong" if case == "not_met" else "right", tamper=case == "tampered")
    res = chain(started, reading, skip=("guard_science",) if case == "missing" else ())
    assert res["branch"].startswith(branch), res["branch"]
    if case == "met":
        assert res["gates"]["transfer"] and res["gates"]["live_P"] and res["gates"]["protection"]
        assert res["protection"]["science"]["source"] == "recomputed here" and all("PROTECT-A reused" in res["protection"][n]["source"] for n in m.REUSED)
        assert res["live_P"]["with_K"] == 297 and res["live_P"]["live_equals_exp56_table"] == 1.0
    assert res["candidate_eligible"] is False


STUB = """#!/bin/bash
echo "uv $*" >> "${MARK_CALLS}"
case "$*" in
  *"watch.py"*) shift 2; exec python3 "$@";;
  *"livek.py start"*) mkdir -p runs/livek; exit 0;;
  *"livek.py fid_text"*) [ -n "${FAIL_FID:-}" ] && exit 3; exit 0;;
  *"livek.py report"*) echo "report $*" > runs/livek/report.txt; exit 0;;
  *"livek.py "*) exit 0;;
esac
exit 9
"""


def run_chain(tmp_path, extra):
    import stat as st_
    import subprocess
    root = tmp_path / "repo"
    for rel in ("scripts/livek/run.sh", "scripts/exp53/watch.py", "scripts/exp35/guard.sh", "scripts/exp38/steps.sh"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(os.path.join(ROOT, rel), root / rel)
    bdir = tmp_path / "bin"
    bdir.mkdir()
    for name, body in {"uv": STUB, "pgrep": '#!/bin/bash\n[ "$1" = "-fl" ] && exit 1\nexec /usr/bin/pgrep "$@"\n'}.items():
        (bdir / name).write_text(body)
        (bdir / name).chmod((bdir / name).stat().st_mode | st_.S_IEXEC)
    env = {**os.environ, "PATH": "%s:%s" % (bdir, os.environ["PATH"]), "MARK_CALLS": str(tmp_path / "calls"), "LIVEK_LIMIT": "60",
           "LIVEK_RESERVE": "4", **extra}
    p = subprocess.run(["bash", str(root / "scripts/livek/run.sh")], env=env, capture_output=True, text=True, timeout=120)
    return p, (tmp_path / "calls").read_text().splitlines()


def test_run_order_fidelity_stop_and_limit(tmp_path):
    p, calls = run_chain(tmp_path / "a", {})
    assert p.returncode == 0, p.stdout + p.stderr
    order = ["start", "fid_text", "live_p", "transfer", "guard_science", "report"]
    pos = [next(i for i, c in enumerate(calls) if "livek.py %s" % s in c) for s in order]
    assert pos == sorted(pos)
    p, calls = run_chain(tmp_path / "b", {"FAIL_FID": "1"})
    assert p.returncode == 3 and "report --finalize" in "\n".join(calls) and not any("live_p" in c for c in calls)
    s = open(os.path.join(ROOT, "scripts/livek/run.sh")).read()
    assert "${LIVEK_LIMIT:-1800}" in s and "${LIVEK_RESERVE:-120}" in s


def test_values_compare_without_thousands_separators():
    assert m.bare("1,270") == "1270" == m.bare("1270") and m.bare("unreadable") == "unreadable" and m.bare(None) is None


def rewrite_reads(out, name, edit):
    f = os.path.join(out, "reads_%s.jsonl" % name)
    lines = [json.loads(x) for x in open(f)]
    lines = edit(lines)
    with open(f, "w") as fh:
        fh.write("".join(json.dumps(x) + "\n" for x in lines))
    m.register(out, "reads_" + name, {"reads_%s.jsonl" % name: m.sha(f)})       # registered again: only the content is wrong


def first_read(lines):
    return next(i for i, x in enumerate(lines) if x["table"])


@data
@pytest.mark.parametrize("damage", ["empty", "missing_request", "duplicate", "wrong_text", "value_not_table", "missing_template", "k_not_table"])
def test_incomplete_or_inconsistent_reads_are_refused(started, tagged, damage):
    """LIVE-K-RUNNER-R1-01: the reading evidence is checked against the source's requests and itself, after registration."""
    chain(started, FakeReading("right"))
    name = "sqa_temp" if damage != "missing_request" else "science"

    def edit(lines):
        i = first_read(lines)
        if damage == "empty":
            return []
        if damage == "missing_request":
            return lines[:-1]
        if damage == "duplicate":
            return lines + [lines[i]]
        if damage == "wrong_text":
            lines[i]["text"] += " (edited)"
        elif damage == "value_not_table":
            lines[i]["reads"][0]["value"] = "999"
        elif damage == "missing_template":
            lines[i]["reads"] = lines[i]["reads"][1:]
        else:
            lines[i]["k_block"] = "Computed from the values above (mass times speed squared, no unit):\nSample A: x = 1\nSample B: x = 2"
        return lines
    rewrite_reads(started, name, edit)
    res = m.report(started)
    assert res["states"][name].startswith("invalid") and res["branch"].startswith("invalid"), (res["states"][name], res["branch"])


@data
def test_an_unreadable_read_stays_valid_and_in_the_denominator(started, tagged):
    reading = FakeReading("right")
    src = m.jsonl(m.TRANSFER["sqa_temp"])
    pic = src[0]["image"]
    reading.unreadable[(pic, "What is the average particle speed in sample A?")] = "unreadable"
    res = chain(started, reading)
    assert res["states"]["sqa_temp"] == "completed" and res["accuracy"]["sqa_temp"] is not None
    reads = [json.loads(x) for x in open(os.path.join(started, "reads_sqa_temp.jsonl"))]
    hit = next(x for x in reads if x["image"] == pic)
    assert any(r["value"] == "unreadable" for r in hit["reads"]) and hit["k_block"] is None   # no K, still decided and counted
