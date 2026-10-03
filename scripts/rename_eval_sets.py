"""Rename the evaluation sets to test_* (our held-out splits) and bench_* (external benchmarks).

    uv run python scripts/rename_eval_sets.py --root data --runs runs            # dry run: print every change
    uv run python scripts/rename_eval_sets.py --root data --runs runs --apply    # do it

  data/v2/eval_coco_<x>.jsonl, eval_gqa, eval_vqa_*   ->  data/v2/test_<x>.jsonl (LookFirst's config names)
  data/v2/eval_pope.jsonl                             ->  data/v2/bench_pope.jsonl
  data/lv_bench/                                      unchanged (image paths inside stay valid); its sets are
                                                      evaluated as bench_lv_<set>
  runs/**/eval/[v2eval_]<old set>.<suffix>            ->  runs/**/eval/<new set>.<suffix>  (.json, .details.jsonl,
                                                      .mismatched.*, .reversed.*); lv_<set> -> bench_lv_<set>;
                                                      the v2eval_ prefix is dropped (v2eval_val_mix -> val_mix)
  runs/**/eval/compare/*                              every dot-separated part renamed the same way
  configs/*.yaml, docs/**/*.md, CLAUDE.md, README.md  references rewritten
  data/v*/MANIFEST.json                               keys and paths rewritten (text_baselines etc.)

Data files stay in the directory of the build that made them; only names change. Everything is planned
and checked before anything is touched: a target that already exists with different content stops the
run. A target with the same content means the rename was half done: the old file is removed. Running it
again changes nothing.
"""
import argparse
import filecmp
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

# old set name -> new set name
TEST = {"eval_coco_exist": "test_exist", "eval_coco_position": "test_position",
        "eval_coco_relation": "test_relation", "eval_coco_size": "test_size", "eval_gqa": "test_gqa",
        "eval_vqa_choice": "test_vqa_choice", "eval_vqa_yesno": "test_vqa_yesno"}
BENCH = {"eval_pope": "bench_pope"}
LV_SETS = ("vqav2_yesno", "aokvqa", "scienceqa", "pope")
LV = {"lv_" + s: "bench_lv_" + s for s in LV_SETS}
SETS: Dict[str, str] = {**TEST, **BENCH, **LV}
OLD_PREFIX = "v2eval_"   # round 2's results on the data-v2 sets

_WORD = r"A-Za-z0-9_"
_SET_RE = re.compile(r"(?<![%s])(?:%s)?(%s)(?![%s])" % (_WORD, OLD_PREFIX, "|".join(sorted(SETS, key=len, reverse=True)),
                                                       _WORD))
_PREFIX_RE = re.compile(r"(?<![%s])%s" % (_WORD, OLD_PREFIX))
_LV_GLOB_RE = re.compile(r"(?<![%s])lv_\*" % _WORD)
_YAML_LV_PREFIX_RE = re.compile(r"^(\s*prefix:\s*)lv_(\s|$)", re.M)
_YAML_V2EVAL_PREFIX_RE = re.compile(r"^\s*prefix:\s*v2eval_\b.*\n", re.M)


def new_part(part: str) -> str:
    """One dot-separated part of a file name: a set name gets its new name, the v2eval_ prefix goes."""
    bare = part[len(OLD_PREFIX):] if part.startswith(OLD_PREFIX) else part
    return SETS.get(bare, bare)


def new_file_name(name: str) -> str:
    return ".".join(new_part(p) for p in name.split("."))


def _parts(path: str) -> List[str]:
    return os.path.normpath(path).split(os.sep)


def rename_target(path: str) -> Optional[str]:
    """Where a data file or evaluation output moves to (same directory), or None if it keeps its name.
    Covers <data>/v<N>/<file>, <runs>/**/eval/<file> and <runs>/**/eval/compare/<file>."""
    parts = _parts(path)
    if len(parts) < 2:
        return None
    parent = parts[-2]
    in_eval = parent == "eval" or (parent == "compare" and len(parts) >= 3 and parts[-3] == "eval")
    in_data_version = re.fullmatch(r"v\d+", parent) is not None
    if not (in_eval or in_data_version):
        return None
    if in_data_version and not parts[-1].startswith("eval_"):
        return None   # data-v2 files other than the eval_ sets (dev_*, pools, MANIFEST) keep their names
    new = new_file_name(parts[-1])
    if new == parts[-1]:
        return None
    return os.path.join(os.path.dirname(path), new)


def rewrite_text(text: str, yaml: bool = False) -> str:
    """References to the old names in a config, a document or a MANIFEST, rewritten."""
    if yaml:
        text = _YAML_V2EVAL_PREFIX_RE.sub("", text)
        text = _YAML_LV_PREFIX_RE.sub(r"\1bench_lv_\2", text)
    text = _SET_RE.sub(lambda m: SETS[m.group(1)], text)
    text = _PREFIX_RE.sub("", text)
    return _LV_GLOB_RE.sub("bench_lv_*", text)


@dataclass
class Op:
    kind: str            # "rename", "drop-duplicate" (target has the same content), "rewrite"
    path: str
    target: str = ""
    text: str = ""


def plan_renames(paths: Iterable[str]) -> List[Tuple[str, str]]:
    """(old, new) for every path that gets a new name; raises if two old files would land on one name."""
    out, seen = [], {}
    for p in sorted(paths):
        t = rename_target(p)
        if t is None:
            continue
        if t in seen:
            raise SystemExit("both %s and %s would become %s" % (seen[t], p, t))
        seen[t] = p
        out.append((p, t))
    return out


def _walk(top: str) -> Iterable[str]:
    for d, dirs, files in os.walk(top):
        dirs.sort()
        for f in sorted(files):
            yield os.path.join(d, f)


def _candidates(root: str, runs: str) -> List[str]:
    paths = []
    if os.path.isdir(root):
        for name in sorted(os.listdir(root)):
            d = os.path.join(root, name)
            if re.fullmatch(r"v\d+", name) and os.path.isdir(d):
                paths += [os.path.join(d, f) for f in sorted(os.listdir(d)) if os.path.isfile(os.path.join(d, f))]
    if os.path.isdir(runs):
        for d, dirs, files in os.walk(runs):
            dirs.sort()
            if os.path.basename(d) == "eval" or (os.path.basename(d) == "compare"
                                                 and os.path.basename(os.path.dirname(d)) == "eval"):
                paths += [os.path.join(d, f) for f in sorted(files)]
    return paths


def _text_files(repo: str, root: str) -> List[Tuple[str, bool]]:
    files = [(os.path.join(repo, "configs", f), True) for f in sorted(os.listdir(os.path.join(repo, "configs")))
             if f.endswith(".yaml")] if os.path.isdir(os.path.join(repo, "configs")) else []
    files += [(p, False) for p in _walk(os.path.join(repo, "docs")) if p.endswith(".md")]
    files += [(os.path.join(repo, f), False) for f in ("CLAUDE.md", "README.md") if os.path.exists(os.path.join(repo, f))]
    if os.path.isdir(root):
        files += [(os.path.join(root, d, "MANIFEST.json"), False) for d in sorted(os.listdir(root))
                  if re.fullmatch(r"v\d+", d) and os.path.exists(os.path.join(root, d, "MANIFEST.json"))]
    return files


def plan(root: str, runs: str, repo: str) -> Tuple[List[Op], List[str]]:
    """Every change, checked; plus warnings about files that look like old evaluation sets but are not known."""
    ops: List[Op] = []
    problems = []
    for old, new in plan_renames(_candidates(root, runs)):
        if not os.path.exists(new):
            ops.append(Op("rename", old, new))
        elif filecmp.cmp(old, new, shallow=False):
            ops.append(Op("drop-duplicate", old, new))
        else:
            problems.append("%s -> %s: the target exists with different content" % (old, new))
    if problems:
        raise SystemExit("nothing done:\n  " + "\n  ".join(problems))
    for path, yaml in _text_files(repo, root):
        with open(path) as f:
            text = f.read()
        new_text = rewrite_text(text, yaml=yaml)
        if new_text != text:
            ops.append(Op("rewrite", path, text=new_text))
    renamed = {o.path for o in ops if o.kind != "rewrite"}
    warnings = ["%s: an eval_ file this script does not know, left as it is" % p
                for p in _candidates(root, "") if os.path.basename(p).startswith("eval_") and p not in renamed]
    return ops, warnings


def _changed_lines(old: str, new: str) -> List[str]:
    a, b = old.splitlines(), new.splitlines()
    if len(a) != len(b):
        import difflib
        return [l for l in difflib.unified_diff(a, b, lineterm="", n=0) if l[:1] in "+-" and l[:3] not in ("+++", "---")]
    return ["-%s\n      +%s" % (x, y) for x, y in zip(a, b) if x != y]


def apply(ops: List[Op]) -> None:
    for op in ops:
        if op.kind == "rename":
            os.rename(op.path, op.target)
        elif op.kind == "drop-duplicate":
            os.remove(op.path)
        else:
            tmp = op.path + ".renaming"
            with open(tmp, "w") as f:
                f.write(op.text)
            os.replace(tmp, op.path)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data", help="data root (data/v2/eval_* live here)")
    p.add_argument("--runs", default="runs", help="runs directory (runs/<round>/eval/)")
    p.add_argument("--repo", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   help="repository whose configs/ and docs/ are rewritten")
    p.add_argument("--apply", action="store_true", help="make the changes (default: only print them)")
    a = p.parse_args(argv)
    ops, warnings = plan(a.root, a.runs, a.repo)
    for op in ops:
        if op.kind == "rename":
            print("rename   %s -> %s" % (op.path, os.path.basename(op.target)))
        elif op.kind == "drop-duplicate":
            print("remove   %s (same content as %s)" % (op.path, os.path.basename(op.target)))
        else:
            with open(op.path) as f:
                old = f.read()
            print("rewrite  %s" % op.path)
            for line in _changed_lines(old, op.text):
                print("      %s" % line)
    for w in warnings:
        print("warning  %s" % w, file=sys.stderr)
    counts = {k: sum(o.kind == k for o in ops) for k in ("rename", "drop-duplicate", "rewrite")}
    print("%d renames, %d duplicates removed, %d files rewritten%s" % (
        counts["rename"], counts["drop-duplicate"], counts["rewrite"], "" if a.apply else " (dry run; --apply to do it)"))
    if a.apply:
        apply(ops)
        print("done")


if __name__ == "__main__":
    main()
