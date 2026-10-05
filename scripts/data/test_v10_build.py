"""Round 10: the four diagram question types are named by their question text; everything else stays out."""
from v10_build import diagram_type


def test_the_four_types_and_nothing_else():
    assert diagram_type("Will these magnets attract or repel each other?") == "repel"
    assert diagram_type("Think about the magnetic force between the magnets in each pair. Which of the following "
                        "statements is true?") == "force"
    assert diagram_type("Compare the average kinetic energies of the particles in each sample. Which sample has the "
                        "higher temperature?") == "temp"
    assert diagram_type("Which solution has a higher concentration of  purple particles?") == "conc"
    assert diagram_type("Complete the text to describe the diagram. Solute particles moved in both directions") is None
    assert diagram_type("Which of these states is farthest north?") is None


def test_held_files_include_round_10s_monitoring_set(tmp_path):
    from v5_build import held_files
    for d in ("v2", "v3", "v4", "v6", "lv_bench", "v10"):
        (tmp_path / d).mkdir()
    (tmp_path / "v10" / "dev_sqa_diagram.jsonl").write_text("")
    (tmp_path / "v10" / "train_x.jsonl").write_text("")
    files = [p.replace(str(tmp_path), "") for p in held_files(str(tmp_path))]
    assert "/v10/dev_sqa_diagram.jsonl" in files and "/v10/train_x.jsonl" not in files


def test_repeat_fills_whole_passes_then_cuts_and_marks_repeats():
    import random
    from v10_focus import repeat
    rows = [{"id": "q%d" % i} for i in range(3)]
    out, passes = repeat(rows, 7, random.Random(0))
    assert len(out) == 7 and passes == 3
    ids = [r["id"] for r in out]
    assert sorted(ids[:3]) == ["q0", "q1", "q2"] and all(i.endswith("#r2") for i in ids[3:6])
    assert len(set(ids)) == 7 and ids[6].endswith("#r3")
